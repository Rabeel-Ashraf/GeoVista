cd $(dirname $0)/..

SRC=/root/models/1002_search_v2.1_n4/actor
OUT=/root/models/1002_search_v2.1_n4/actor_hf

# OUT=/home/models/debug/OpenThinkIMG/1017_search_v3.0_debug/global_step_0/actor/huggingface

# mkdir -p $OUT
# export CUDA_VISIBLE_DEVICES=
# torchrun --standalone --nproc_per_node=32 \
#     tools/ckpt_converter.py \
#     --src $SRC \
#     --out $OUT

export CUDA_VISIBLE_DEVICES=0
python3 tools/ckpt_validator.py \
    --model $OUT \
    --trust_remote_code
