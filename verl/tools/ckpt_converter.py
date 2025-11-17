#!/usr/bin/env python3
import argparse
import json
import os
import re
import shutil
import sys
import warnings
from typing import Optional, Tuple


def _infer_world_size_and_pattern(ckpt_dir: str) -> Tuple[int, str]:
    """
    Infer world size from files like model_world_size_32_rank_0.pt and
    return the filename pattern with a placeholder for rank.
    """
    files = os.listdir(ckpt_dir)
    pat = re.compile(r"model_world_size_(\d+)_rank_(\d+)\.pt$")
    matches = [pat.match(f) for f in files]
    matches = [m for m in matches if m is not None]
    if not matches:
        raise FileNotFoundError(
            f"No model shard files found in {ckpt_dir}. Expected files like 'model_world_size_XX_rank_YY.pt'."
        )
    # Use the first match to get world size
    world_size = int(matches[0].group(1))
    # Build pattern with {rank}
    pattern = os.path.join(ckpt_dir, f"model_world_size_{world_size}_rank_{{rank}}.pt")
    return world_size, pattern


def _copy_hf_sidecar(src_hf_dir: str, dst_dir: str):
    os.makedirs(dst_dir, exist_ok=True)
    if not os.path.isdir(src_hf_dir):
        raise FileNotFoundError(f"HuggingFace sidecar directory not found: {src_hf_dir}")

    # copy tokenizer/config related files
    for name in os.listdir(src_hf_dir):
        src = os.path.join(src_hf_dir, name)
        dst = os.path.join(dst_dir, name)
        if os.path.isdir(src):
            if os.path.exists(dst):
                shutil.rmtree(dst)
            shutil.copytree(src, dst)
        else:
            shutil.copy2(src, dst)


def _is_vision2seq_config(hf_config) -> bool:
    try:
        from transformers import AutoModelForVision2Seq

        return type(hf_config) in AutoModelForVision2Seq._model_mapping.keys()
    except Exception:
        return False


def _build_model_and_fsdp(hf_dir: str, device: Optional[str] = None):
    import torch
    import torch.distributed as dist
    from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
    from torch.distributed.fsdp import MixedPrecision
    from torch.distributed.fsdp.api import ShardingStrategy
    from transformers import AutoConfig, AutoModelForCausalLM, AutoModelForVision2Seq

    # local imports from repo
    from verl.utils.fsdp_utils import get_fsdp_wrap_policy, init_fn

    trust_remote_code = False
    hf_config = AutoConfig.from_pretrained(hf_dir, trust_remote_code=trust_remote_code)

    # Create base HF model with config only (no weights)
    model_cls = AutoModelForVision2Seq if _is_vision2seq_config(hf_config) else AutoModelForCausalLM
    # Use fp32 init to avoid dtype issues; weights will be loaded from checkpoint
    base_model = model_cls.from_config(hf_config, trust_remote_code=trust_remote_code)

    # Build FSDP with a standard wrap policy based on the model's _no_split_modules
    auto_wrap_policy = get_fsdp_wrap_policy(module=base_model, config=None)

    # Mixed precision only affects communication; parameters are loaded from ckpt
    mixed_precision = MixedPrecision(param_dtype=torch.bfloat16, reduce_dtype=torch.float32, buffer_dtype=torch.float32)

    # Determine device id if CUDA available
    device_id = None
    if torch.cuda.is_available():
        device_id = torch.cuda.current_device() if device is None else device

    fsdp_model = FSDP(
        base_model,
        use_orig_params=False,
        auto_wrap_policy=auto_wrap_policy,
        device_id=device_id,
        sharding_strategy=ShardingStrategy.FULL_SHARD,  # zero3
        mixed_precision=mixed_precision,
        param_init_fn=init_fn,
        sync_module_states=True,
        forward_prefetch=False,
    )
    return base_model, fsdp_model


def _merge_fsdp_to_full_state_dict(fsdp_model, shard_path: str):
    import torch
    from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
    from torch.distributed.fsdp.api import FullStateDictConfig, ShardedStateDictConfig, StateDictType

    # Load sharded state into FSDP
    state_dict_cfg = ShardedStateDictConfig(offload_to_cpu=True)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        with FSDP.state_dict_type(fsdp_model, StateDictType.SHARDED_STATE_DICT, state_dict_cfg):
            model_state_dict = torch.load(shard_path, weights_only=False, map_location="cpu")
            fsdp_model.load_state_dict(model_state_dict)

    # Consolidate to full state dict (rank0_only)
    cfg = FullStateDictConfig(offload_to_cpu=True, rank0_only=True)
    with FSDP.state_dict_type(fsdp_model, StateDictType.FULL_STATE_DICT, cfg):
        full_state_dict = fsdp_model.state_dict()
    return full_state_dict


def convert_fsdp_ckpt_to_hf(src_dir: str, out_dir: Optional[str], max_shard_size: str = "5GB") -> str:
    """
    Convert an FSDP sharded checkpoint directory into a Hugging Face checkpoint
    with safetensors, plus copy tokenizer/config sidecar files.

    Requirements:
    - Run under torch.distributed with world size equal to the saved checkpoint world size.
      Example: torchrun --nproc_per_node=32 tools/ckpt_converter.py --src <.../actor> --out <dst>
    """
    import torch
    import torch.distributed as dist
    from transformers import PreTrainedModel

    # Validate source
    if not os.path.isdir(src_dir):
        raise NotADirectoryError(f"Input directory does not exist: {src_dir}")

    world_size, pattern = _infer_world_size_and_pattern(src_dir)
    src_hf_dir = os.path.join(src_dir, "huggingface")
    if out_dir is None:
        out_dir = os.path.join(src_dir, "huggingface_merged")

    # Initialize distributed if not
    if not dist.is_initialized():
        # Try to initialize from environment (torchrun)
        backend = "nccl" if torch.cuda.is_available() else "gloo"
        try:
            dist.init_process_group(backend=backend)
        except Exception as e:
            raise RuntimeError(
                "Failed to initialize torch.distributed. Please run with torchrun, e.g.\n"
                f"  torchrun --nproc_per_node={world_size} tools/ckpt_converter.py --src {src_dir} --out {out_dir}\n"
                f"Underlying error: {e}"
            )

    rank = dist.get_rank()
    ws = dist.get_world_size()
    if ws != world_size:
        raise RuntimeError(
            f"Distributed world size {ws} does not match checkpoint world size {world_size}.\n"
            f"Run with: torchrun --nproc_per_node={world_size} tools/ckpt_converter.py --src {src_dir} --out {out_dir}"
        )

    if torch.cuda.is_available():
        local_rank = int(os.environ.get("LOCAL_RANK", "0"))
        torch.cuda.set_device(local_rank)
        device = local_rank
    else:
        device = None

    base_model, fsdp_model = _build_model_and_fsdp(src_hf_dir, device=device)


    # Load this rank's shard and consolidate
    shard_path = pattern.format(rank=rank)
    if not os.path.exists(shard_path):
        raise FileNotFoundError(f"Missing shard for rank {rank}: {shard_path}")

    full_state_dict = _merge_fsdp_to_full_state_dict(fsdp_model, shard_path)

    # Only rank 0 writes files
    if rank == 0:
        os.makedirs(out_dir, exist_ok=True)

        # Save model weights as safetensors via HF
        # base_model: transformers.PreTrainedModel
        if not isinstance(base_model, PreTrainedModel):
            # Some custom models still subclass; otherwise, rely on save_pretrained on wrapper
            raise TypeError("The base HF model is not a transformers.PreTrainedModel instance.")

        # Enforce safetensors save by setting safe_serialization=True
        base_model.save_pretrained(
            out_dir,
            state_dict=full_state_dict,
            safe_serialization=True,
            max_shard_size=max_shard_size,
        )

        # Copy tokenizer/config sidecar files (tokenizer.json, merges.txt, etc.)
        _copy_hf_sidecar(src_hf_dir, out_dir)

    # Sync all ranks before returning
    dist.barrier()
    return out_dir


def main():
    parser = argparse.ArgumentParser(description="Convert FSDP checkpoint to HF safetensors + tokenizer/config")
    parser.add_argument("--src", required=True, help="Path to FSDP checkpoint dir (e.g., .../global_step_xxx/actor)")
    parser.add_argument(
        "--out",
        default=None,
        help="Output directory for merged HF checkpoint (defaults to <src>/huggingface_merged)",
    )
    parser.add_argument(
        "--max_shard_size",
        default="5GB",
        help="Max shard size for safetensors when saving via Hugging Face (e.g., '2GB', '5GB')",
    )

    args = parser.parse_args()
    try:
        out_dir = convert_fsdp_ckpt_to_hf(args.src, args.out, max_shard_size=args.max_shard_size)
    except Exception as e:
        # Print a concise error and exit non-zero
        print(f"[ckpt_converter] Error: {e}", file=sys.stderr)
        sys.exit(1)

    # Print the output directory path as the script result
    print(out_dir)


if __name__ == "__main__":
    main()
