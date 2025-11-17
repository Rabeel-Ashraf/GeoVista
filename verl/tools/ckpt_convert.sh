#!/usr/bin/env bash
# 4-node (8 GPUs/node) FSDP shard merge via pdsh + torchrun

# set -euo pipefail

# ======== 用户配置 ========
# 主机列表（按顺序，0 号为 master）
HOSTS=("29.207.53.157" "29.210.132.39" "29.210.129.24" "29.210.132.205")

# 各节点上的仓库根目录（与当前项目一致）
REPO_DIR=/mnt/private/agent_workspace/hunyuan-o3/external/DeepEyes/verl

# 源分片目录（所有机器都能访问到）
SRC=/mnt/lzy/DeepEyes/checkpoints_gy/OpenThinkIMG/1002_search_v2.1_n4/global_step_187/actor

# 输出目录（仅 master 节点写就可以）
OUT=/root/models/1002_search_v2.1_n4/actor_hf

# Python 可执行程序
PYTHON_BIN="python3"

# 每节点进程数 / 总节点数
NPROC_PER_NODE=8
NNODES=${#HOSTS[@]}

# Master 地址与端口（默认取第一个主机）
MASTER_ADDR="${HOSTS[0]}"
MASTER_PORT=29500

# 可选：环境激活命令（如有 conda/venv）
# ENV_ACTIVATE="source ~/.bashrc && conda activate myenv"
ENV_ACTIVATE=""

# 可选：NCCL 参数（如需指定网卡/IB）
NCCL_SOCKET_IFNAME=bond1
export NCCL_IB_HCA=mlx5_bond_1,mlx5_bond_5,mlx5_bond_3,mlx5_bond_7,mlx5_bond_4,mlx5_bond_8,mlx5_bond_2,mlx5_bond_6

# ======== 结束用户配置 ========

# 组合 HOSTS 为 pdsh 的 -w 参数
join_by_comma() { local IFS=,; echo "$*"; }
PDSH_HOSTS="$(join_by_comma "${HOSTS[@]}")"

echo "[info] Hosts: ${HOSTS[*]}"
echo "[info] Master: ${MASTER_ADDR}:${MASTER_PORT}"
echo "[info] SRC: ${SRC}"
echo "[info] OUT: ${OUT}"
echo "[info] REPO: ${REPO_DIR}"

# 简单连通性/环境检查（可按需注释）
echo "[check] GPU & path sanity on all nodes..."
pdsh -R ssh -w "$PDSH_HOSTS" "bash -lc '
  hostname
  $PYTHON_BIN -c \"import torch; print(torch.cuda.device_count())\"
  test -d \"$REPO_DIR\" && echo \"repo=OK\" || echo \"repo=MISSING\"
  test -d \"$SRC\" && echo \"src=OK\" || echo \"src=MISSING\"
'"

# 清理函数：中断时尽量拉闸
cleanup() {
  echo "[warn] Caught signal, trying to kill remote torchrun..."
  pdsh -R ssh -w "$PDSH_HOSTS" "pkill -f 'tools/ckpt_converter.py' || true; pkill -f torchrun || true" || true
}
trap cleanup INT TERM

# 启动各节点
for i in "${!HOSTS[@]}"; do
  host="${HOSTS[$i]}"
  echo "[launch] $host (NODE_RANK=$i)"
  pdsh -R ssh -w "$host" "bash -lc '
    set -Eeuo pipefail
    $ENV_ACTIVATE
    cd \"$REPO_DIR\"
    mkdir -p logs
    export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
    export PYTHONUNBUFFERED=1
    export TORCH_DISTRIBUTED_DEBUG=DETAIL
    export NCCL_DEBUG=TRACE
    export NCCL_DEBUG_SUBSYS=INIT,GRAPH,COLL,NET
    export NCCL_ASYNC_ERROR_HANDLING=1
    export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
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
    export TORCHELASTIC_ERROR_FILE=logs/elastic_\$(hostname).log
    export NCCL_DEBUG_FILE=logs/nccl_%h_r%r.log

    stdbuf -oL -eL torchrun \
      --nnodes=$NNODES \
      --nproc_per_node=$NPROC_PER_NODE \
      --node_rank=$i \
      --master_addr=\"$MASTER_ADDR\" \
      --master_port=$MASTER_PORT \
      --redirects=3 --tee 3 --log_dir logs \
      tools/ckpt_converter.py \
      --src \"$SRC\" \
      --out \"$OUT\" 2>&1 | tee -a logs/launcher_${host}.log
  '" &
done

# 等所有节点完成
wait
echo "[done] Conversion finished. Output at: $OUT"
