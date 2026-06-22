# qwen3
modelscope download --model Qwen/Qwen3-30B-A3B \
  --local_dir /data/models/Qwen3-30B-A3B

# deepseek
modelscope download \
  --model deepseek-ai/DeepSeek-V2-Lite-Chat \
  --local_dir /data/models/DeepSeek-V2-Lite-Chat \
  --max-workers 16 \
  --include "*.safetensors" "*.json" "tokenizer*" "*.py"
  