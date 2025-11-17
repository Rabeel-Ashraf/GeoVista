#!/usr/bin/env bash
set -euo pipefail

# 源分片目录（所有机器都能访问到）
SRC=/mnt/lzy/DeepEyes/checkpoints_gy/OpenThinkIMG/1002_search_v2.1_n4/global_step_187/actor

# 输出目录（仅 master 节点写就可以）
OUT=/root/models/1002_search_v2.1_n4/actor_hf

export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export NCCL_DEBUG=INFO
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export NCCL_ASYNC_ERROR_HANDLING=1
# 可选：如需指定网卡，取消注释
export NCCL_IB_GID_INDEX=3
export NCCL_IB_SL=3
export NCCL_CHECKS_DISABLE=1
export NCCL_P2P_DISABLE=0
export NCCL_IB_DISABLE=0
export NCCL_LL_THRESHOLD=16384
export NCCL_IB_CUDA_SUPPORT=1
export NCCL_SOCKET_IFNAME=bond1
export UCX_NET_DEVICES=bond1
export NCCL_IB_HCA=mlx5_bond_1,mlx5_bond_5,mlx5_bond_3,mlx5_bond_7,mlx5_bond_4,mlx5_bond_8,mlx5_bond_2,mlx5_bond_6
export NCCL_COLLNET_ENABLE=0
export SHARP_COLL_ENABLE_SAT=0
export NCCL_NET_GDR_LEVEL=2
export NCCL_IB_QPS_PER_CONNECTION=4
export NCCL_IB_TC=160
export NCCL_PXN_DISABLE=0


torchrun --standalone --nproc_per_node=${GPU_COUNT} tools/ckpt_converter.py --src "$SRC" --out "$OUT"
