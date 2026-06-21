# vLLM-ascend环境搭建  

## <font color = brown>环境搭建流程</font>

### <font color = green>单卡环境搭建</font>

> 简单参数适配单卡即可完成，但是诸如 --dbo这种设计EP并行的MOE推理，需要多卡。本小节先在单卡环境上搭建好vllm-ascend插件，后续再搭建多卡通信环境。

1. 镜像环境检查  

   使用vllm-ascend项目自带的`check_env.py`检查环境，我当前的环境如下：  

   ```shell
   Ascend 910B NPU
   openEuler操作系统（容器自带）
   Python 3.11（容器自带）
   CANN 8.5.1（容器自带）
   torch 2.9.0（容器自带）
   torch-npu 2.9.0rc1（容器自带环境，后续需要重新install成torch-npu 2.9.0以解决兼容性问题）
   Triton-Ascend 3.2.0（需要自行安装）
   vLLM-Ascend `0.1.dev1+gce9effc33.d20260507`（最新github main分支的版本）
   vLLM `0.19.2rc1.dev17+gd886c26d4`（和vllm-ascend最新main分支兼容版本）
   ```

2. 虚拟环境搭建

   * 启动虚拟环境

     ```shell
     python3 -m venv --system-site-packages .venv   # 使用 --system-site-packages 继承系统环境
     ```

   * 基础工具链安装

     ```shell
     dnf install -y git gcc gcc-c++ cmake numactl-devel wget curl jq ninja-build python3-pip
     ```

     ```shell
     python -m pip install \
       cmake \
       ninja \
       packaging \
       "setuptools>=77,<81" \
       setuptools-scm
     ```

     ```shell
     # For torch-npu dev version or x86 machine
     pip config set global.extra-index-url "https://download.pytorch.org/whl/cpu/"
     ```

3. 源码安装vllm

   * clone并切到对应版本 

     ```shell
     git clone --recursive git@github.com:vllm-project/vllm.git
     git checkout d886c26d4d4fef7d079696beb4ece1cfb4b008a8   
     ```

     vllm版本具体参考 [vllm-ascend Dokerfile](https://github.com/vllm-project/vllm-ascend/blob/main/Dockerfile)。

   * 源码安装  

     这里的核心要点是，一定使用 `--no-deps --no-build-isolation`，以防止vllm 下载并安装默认torch（2.10.0版本，存在兼容性问题）。

     ```shell
     VLLM_TARGET_DEVICE=empty \
     python -m pip install -v --no-deps --no-build-isolation -e .
     ```

4. 源码安装vllm-ascend

   ```shell
   # 进入 vllm-ascend目录
   python -m pip install -v --no-deps --no-build-isolation -e .
   ```

5. 解决版本兼容性问题  

   * torch-npu必须是torch-npu2.9.0  

     ```shell
     python -m pip install --no-cache-dir --force-reinstall --no-deps \
       "https://vllm-ascend.obs.cn-north-4.myhuaweicloud.com/vllm-ascend/torch_npu-2.9.0.post1%2Bgitdc51c2d-cp311-cp311-manylinux_2_28_x86_64.whl"
     ```

   * triton-ascend必须是3.2.0  

     ```shell
     python -m pip install --no-cache-dir --force-reinstall --ignore-installed --no-deps \
       -i https://pypi.org/simple \
       "triton-ascend==3.2.0"
     ```

   

#### 最终测试  

使用如下`qwen3-0.6B`测试。

```python
from vllm import LLM, SamplingParams

prompts = [
    "Hello, my name is",
    "The president of the United States is",
    "The capital of France is",
    "The future of AI is",
]

# Create a sampling params object.
sampling_params = SamplingParams(temperature=0.8, top_p=0.95)
# Create an LLM.
llm = LLM(model="Qwen/Qwen3-0.6B")

# Generate texts from the prompts.
outputs = llm.generate(prompts, sampling_params)
for output in outputs:
    prompt = output.prompt
    generated_text = output.outputs[0].text
    print(f"Prompt: {prompt!r}, Generated text: {generated_text!r}")
```

运行 `check.py`测试是否能够跑通`qwen3-0.6B`的推理流程。

## <font color = brown>问题</font>

### <font color = green>镜像相关</font>

#### python 版本问题

检查 Python：

```bash
which python
python -V
python -m pip -V
```

期望 Python 来自：

```text
/usr/local/python3.11.14/bin/python
```

昇腾容器有两条python环境，需要自行export一下。

### <font color = green>OpenEuler相关</font>

```shell
dnf install -y 包名
yum install -y 包名
```

### <font color = green>vLLM相关</font>

#### vLLM或是vLLM-ascend出现module缺失  

这种情况就是版本没有对齐。参考 https://docs.vllm.ai/projects/vllm-ascend-cn/zh-cn/latest/community/versioning_policy.html 版本管理矩阵对齐。如果是用的github最新的main分支，那么vllm等依赖版本通过`Dockerfile`来对齐即可。

#### torch_air 缺失 bug

```shell
python - <<'PY'
import importlib.metadata as md

for p in ["torch", "torch-npu", "torchvision", "torchaudio", "vllm", "vllm-ascend"]:
    try:
        print(f"{p:12s} => {md.version(p)}")
    except Exception as e:
        print(f"{p:12s} => NOT FOUND: {e}")

print("\nChecking torchair API...")
try:
    import torch_npu
    from torch_npu.dynamo import torchair
    print("torch_npu:", torch_npu.__file__)
    print("torchair:", torchair)
    print("torchair file:", getattr(torchair, "__file__", None))
    print("has register_replacement:", hasattr(torchair, "register_replacement"))
    print("related symbols:", [x for x in dir(torchair) if "register" in x or "replace" in x])
except Exception as e:
    print("FAILED:", repr(e))
PY
```

运行上述脚本，`has register_replacement:`会返回false，原因是算力广场提供的pytorch镜像使用的是`torch-npu2.9.0rc1`，而不是官方要求的`torch-npu2.9.0`。重新构建好 `torch-npu2.9.0`即可。

#### 缺少triton

```shell
(.venv) [root@179581a576fc vllm-ascend-dly]# python check.py 
INFO 05-07 10:36:10 [__init__.py:44] Available plugins for group vllm.platform_plugins:
INFO 05-07 10:36:10 [__init__.py:46] - ascend -> vllm_ascend:register
INFO 05-07 10:36:10 [__init__.py:49] All plugins in this group will be loaded. Set `VLLM_PLUGINS` to control which plugins to load.
INFO 05-07 10:36:10 [__init__.py:239] Platform plugin ascend is activated
INFO 05-07 10:36:14 [importing.py:44] Triton is installed but 0 active driver(s) found (expected 1). Disabling Triton to prevent runtime errors.
INFO 05-07 10:36:14 [importing.py:68] Triton not installed or not compatible; certain GPU-related functions will not be available.
```

```shell
AttributeError: '_OpNamespace' 'vllm' object has no attribute 'qkv_rmsnorm_rope'
```

相关issue：https://github.com/vllm-project/vllm-ascend/issues/6737

重新构建triton-ascend：  

```shell
python -m pip install --no-cache-dir --force-reinstall --ignore-installed --no-deps \
  -i https://pypi.org/simple \
  "triton-ascend==3.2.0"
```

测试如下脚本，只要`qkv_rmsnorm_rope`存在即可。

```shell
python - <<'PY'
import torch
import importlib.util

print("torch:", torch.__version__)
try:
    import torch_npu
    print("torch_npu:", torch_npu.__version__)
except Exception as e:
    print("torch_npu import failed:", repr(e))

try:
    import vllm
    print("vllm:", vllm.__version__, vllm.__file__)
except Exception as e:
    print("vllm import failed:", repr(e))

try:
    import vllm_ascend
    print("vllm_ascend:", getattr(vllm_ascend, "__version__", "unknown"), vllm_ascend.__file__)
except Exception as e:
    print("vllm_ascend import failed:", repr(e))

print("triton spec:", importlib.util.find_spec("triton"))
print("triton_ascend spec:", importlib.util.find_spec("triton_ascend"))

for name in [
    "qkv_rmsnorm_rope",
    "triton_split_qkv_rmsnorm_rope",
    "triton_split_qkv_rmsnorm_mrope",
    "fused_qk_norm_rope",
]:
    try:
        op = getattr(torch.ops.vllm, name)
        print("[OK] torch.ops.vllm.%s exists: %s" % (name, op))
    except Exception as e:
        print("[MISS] torch.ops.vllm.%s: %s" % (name, repr(e)))
PY
```





