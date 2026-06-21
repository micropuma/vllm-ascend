#!/usr/bin/env python3
import gc
import sys
import os

# 系统环境变量配置
os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"
os.environ["HCCL_INTRA_ROCE_ENABLE"] = "1"          # 允许跨平面通信

# 将各种缓存目录重定向到 /data（空间充足）
os.environ["TORCH_EXTENSIONS_DIR"] = "/data/torch_cache"
os.environ["VLLM_CACHE_DIR"] = "/data/vllm_cache"
os.environ["TRITON_CACHE_DIR"] = "/data/triton_cache"  
os.environ["HF_HOME"] = "/data/huggingface_cache"
os.environ["TMPDIR"] = "/data/tmp"
os.environ["TORCHINDUCTOR_CACHE_DIR"] = "/data/torch_inductor_cache"
os.environ["VLLM_COMPILE_CACHE_PATH"] = "/data/vllm_compile_cache"  

# 创建所有缓存目录
for d in ["/data/torch_cache", "/data/vllm_cache", "/data/triton_cache",
          "/data/huggingface_cache", "/data/tmp"]:
    os.makedirs(d, exist_ok=True)

# 现在可以安全导入 vLLM 和 torch
import torch
from vllm import LLM, SamplingParams
from vllm.distributed.parallel_state import (
    destroy_distributed_environment,
    destroy_model_parallel,
)

def clean_up():
    destroy_model_parallel()
    destroy_distributed_environment()
    gc.collect()
    if torch.npu.is_available():
        torch.npu.empty_cache()

def main():
    prompts = [
        "Hello, my name is",
        "The future of AI is",
    ]
    sampling_params = SamplingParams(temperature=0.6, top_p=0.95, top_k=40)

    llm = LLM(
        model="/mnt/moark-models/Qwen3.6-35B-A3B",
        tensor_parallel_size=4,
        distributed_executor_backend="mp",
        max_model_len=4096,
        enable_expert_parallel=True,
        enable_dbo=True,
    )

    outputs = llm.generate(prompts, sampling_params)
    for output in outputs:
        prompt = output.prompt
        generated_text = output.outputs[0].text
        print(f"Prompt: {prompt!r}, Generated text: {generated_text!r}")

    del llm
    clean_up()

if __name__ == "__main__":
    sys.exit(main())