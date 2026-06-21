# ============================================== client测试 ==============================================
# 1. 使用sharedgpt做最简测试

# mkdir -p /root/datasets/sharegpt
# cd /root/datasets/sharegpt

# wget https://huggingface.co/datasets/anon8231489123/ShareGPT_Vicuna_unfiltered/resolve/main/ShareGPT_V3_unfiltered_cleaned_split.json


MODEL=/data/workspace/vllm-ascend-dly/tutorial/models/Qwen3-8B

vllm bench serve \
  --backend vllm \
  --model $MODEL \
  --endpoint /v1/completions \
  --dataset-name sharegpt \
  --dataset-path /root/datasets/sharegpt/ShareGPT_V3_unfiltered_cleaned_split.json \
  --num-prompts 10