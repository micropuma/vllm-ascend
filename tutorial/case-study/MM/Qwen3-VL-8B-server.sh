export VLLM_USE_MODELSCOPE=true
export VLLM_WORKER_MULTIPROC_METHOD=spawn
export ASCEND_RT_VISIBLE_DEVICES=0

MODEL=/root/models/Qwen3-VL-8B-Instruct

vllm serve $MODEL \
  --dtype bfloat16 \
  --limit-mm-per-prompt '{"image": 1}' \
  --max-model-len 16384 \
  --max-num-batched-tokens 16384

