#!/usr/bin/env python3
import argparse
import os
import sys
from typing import Optional


def _load_tokenizer(model_dir: str, trust_remote_code: bool):
    from transformers import AutoTokenizer, AutoProcessor

    tok = None
    err = None
    try:
        tok = AutoTokenizer.from_pretrained(model_dir, trust_remote_code=trust_remote_code)
        return tok
    except Exception as e:
        err = e
    # fallback to processor.tokenizer for multi-modal models
    try:
        proc = AutoProcessor.from_pretrained(model_dir, trust_remote_code=trust_remote_code)
        if hasattr(proc, "tokenizer") and proc.tokenizer is not None:
            return proc.tokenizer
    except Exception:
        pass
    raise RuntimeError(f"Failed to load tokenizer from {model_dir}: {err}")


def _load_model(model_dir: str, trust_remote_code: bool, use_device_map_auto: bool):
    import torch
    from transformers import AutoConfig, AutoModelForCausalLM, AutoModelForVision2Seq

    # dtype preference
    torch_dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32

    # determine class from config
    config = AutoConfig.from_pretrained(model_dir, trust_remote_code=trust_remote_code)
    model_cls = AutoModelForCausalLM
    try:
        from transformers import AutoModelForVision2Seq

        # If it's a Vision2Seq config, select the proper class
        if type(config) in AutoModelForVision2Seq._model_mapping.keys():
            model_cls = AutoModelForVision2Seq
    except Exception:
        pass

    kwargs = dict(trust_remote_code=trust_remote_code, torch_dtype=torch_dtype)
    if use_device_map_auto:
        kwargs.update(dict(device_map="auto", low_cpu_mem_usage=True))

    try:
        model = model_cls.from_pretrained(model_dir, **kwargs)
        return model
    except Exception as e:
        # Fallback: avoid device_map if it failed
        if use_device_map_auto:
            kwargs.pop("device_map", None)
            kwargs.pop("low_cpu_mem_usage", None)
            model = model_cls.from_pretrained(model_dir, **kwargs)
            return model
        raise e


def _prepare_prompt(tokenizer, prompt: str) -> str:
    # Use chat template if available
    apply_chat = getattr(tokenizer, "apply_chat_template", None)
    if callable(apply_chat) and getattr(tokenizer, "chat_template", None):
        messages = [{"role": "user", "content": prompt}]
        try:
            return apply_chat(messages, tokenize=False, add_generation_prompt=True)
        except Exception:
            pass
    return prompt


def validate(model_dir: str, prompt: str = "hello", max_new_tokens: int = 64, temperature: float = 0.7,
             top_p: float = 0.9, trust_remote_code: bool = False, device: Optional[str] = None,
             use_device_map_auto: bool = True) -> str:
    import torch
    from transformers import GenerationConfig

    if not os.path.isdir(model_dir):
        raise NotADirectoryError(f"Not a directory: {model_dir}")

    tokenizer = _load_tokenizer(model_dir, trust_remote_code)
    model = _load_model(model_dir, trust_remote_code, use_device_map_auto)

    # Resolve device for inputs
    if use_device_map_auto:
        # Put inputs to the device of input embeddings
        try:
            input_device = model.get_input_embeddings().weight.device
        except Exception:
            input_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        input_device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        model.to(input_device)

    # Prepare prompt text (with chat template if present)
    prompt_text = _prepare_prompt(tokenizer, prompt)

    # Tokenize
    inputs = tokenizer([prompt_text], return_tensors="pt")
    inputs = {k: v.to(input_device) for k, v in inputs.items()}

    # Generation config
    try:
        gen_config = GenerationConfig.from_pretrained(model_dir)
    except Exception:
        gen_config = GenerationConfig()
    gen_config.max_new_tokens = max_new_tokens
    gen_config.temperature = temperature
    gen_config.top_p = top_p
    gen_config.do_sample = True

    # Ensure pad token
    if tokenizer.pad_token_id is None and tokenizer.eos_token_id is not None:
        tokenizer.pad_token = tokenizer.eos_token

    # Generate
    with torch.no_grad():
        outputs = model.generate(**inputs, generation_config=gen_config)

    # Decode only the new tokens
    input_len = inputs["input_ids"].shape[1]
    new_tokens = outputs[0, input_len:]
    text = tokenizer.decode(new_tokens, skip_special_tokens=True)

    if text is None or len(text.strip()) == 0:
        raise RuntimeError("Empty generation result")

    return text.strip()


def main():
    parser = argparse.ArgumentParser(description="Validate a merged HF checkpoint by generating a reply to 'hello'")
    parser.add_argument("--model", required=True, help="Path to merged HF checkpoint directory")
    parser.add_argument("--prompt", default="hello", help="Prompt text to query (default: hello)")
    parser.add_argument("--max_new_tokens", type=int, default=64)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top_p", type=float, default=0.9)
    parser.add_argument("--trust_remote_code", action="store_true")
    parser.add_argument("--no_device_map_auto", action="store_true", help="Disable device_map='auto' loading")

    args = parser.parse_args()
    try:
        text = validate(
            model_dir=args.model,
            prompt=args.prompt,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            top_p=args.top_p,
            trust_remote_code=args.trust_remote_code,
            use_device_map_auto=not args.no_device_map_auto,
        )
    except Exception as e:
        print(f"[ckpt_validator] Error: {e}", file=sys.stderr)
        sys.exit(1)

    print(text)
    print("OK")


if __name__ == "__main__":
    main()

