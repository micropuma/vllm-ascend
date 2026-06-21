# ============================================== 开启server ==============================================
export VLLM_WORKER_MULTIPROC_METHOD=spawn
export ASCEND_RT_VISIBLE_DEVICES=0

MODEL=/data/workspace/vllm-ascend-dly/tutorial/models/Qwen3-8B

vllm serve $MODEL \
  --dtype bfloat16 \
  --max-model-len 8192 \
  --max-num-batched-tokens 8192


