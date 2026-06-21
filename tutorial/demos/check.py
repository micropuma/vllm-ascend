from vllm import LLM, SamplingParams

'''
    参考 https://github.com/vllm-project/vllm-ascend/issues/414
    * 设置 VLLM_WORKER_MULTIPROC_METHOD=spawn  
    * 将 LLM 对象的创建放在 if __name__ == "__main__": 下面
'''

def main():

    prompts = [
        "Hello, my name is",
        "The president of the United States is",
        "The capital of France is",
        "The future of AI is",
    ]

    # Create a sampling params object.
    sampling_params = SamplingParams(temperature=0.8, top_p=0.95)
    # Create an LLM.
    llm = LLM(model="/mnt/.cache/modelscope/Qwen3-0.6B")

    # Generate texts from the prompts.
    outputs = llm.generate(prompts, sampling_params)
    for output in outputs:
        prompt = output.prompt
        generated_text = output.outputs[0].text
        print(f"Prompt: {prompt!r}, Generated text: {generated_text!r}")

if __name__ == "__main__":
    main()