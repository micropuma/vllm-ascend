# 详尽的 性能benchmark测试任务 

1. 注意：本次benchmark任务原则是自动化覆盖尽量多的测试case，做详尽的消融实验。所以如果跑的过程，有case出bug，记录下来，继续跑别的case，先力求覆盖度。  
2. 注意：后续的测试矩阵，测试任务，都做了p0,p1,p2标记。其中p0是测试重点，搞定p0后再做p1和p2，依次优先级降低。我要求你尽量做完所有测试。   
3. 生成一个详尽测试报告在 /data/workspace/vllm-dbo-v0221/vllm-ascend/testbench/MOE/dbo/testbench/ 下，要求是简洁，清晰，数据展示足够直观。  
4. 具体运行指南，参考 /data/workspace/vllm-dbo-v0221/vllm-ascend/.agents/skills/dbo-e2e-validation

## 环境要求  

* 虚拟环境位置：/data/workspace/vllm-dbo-v0221/.venv-dbo/
* 环境脚本加载：/data/workspace/vllm-dbo-v0221/env.sh
* 启动测试，注意取消 localhost，影响server和test的端口监听    

## 测试流程  
1. 启动对应server  
2. 启动test  
3. 记录 TFTT, TPOT等信息，越详细越好，记录如下：
    * 参数配置记录  
    * 性能信息记录  
    * 汇总成表格  
    * 输出位置：/data/workspace/vllm-dbo-v0221/vllm-ascend/testbench/MOE/dbo/testbench/results  
4. 删除所有进程，进行下一组测试   

## 测试的workload  

参考 /data/workspace/vllm-dbo-v0221/vllm-ascend/testbench/MOE/dbo/demos的测试方法，以如下顺序测试：  

1. 先完成 500个 prefill4k任务（覆盖所有测试矩阵），优先测试好这个workload   [P0]
2. 测试 16个 prefill4k任务，检验dbo对于batch的要求，注意这种情况不用测试aicpu和aiv的矩阵了（后续可以深挖掘）  [P1]
3. 测试 decode，修改脚本，减小decode的dbo触发threshold，确保decode触发dbo，并观测decode是否有效果？（仅仅测试dbo即可）   [P2]


## 测试矩阵  

如下是详细的测试矩阵：

### DeepseekV2模型测试 :key: [P0]

* TP=2，EP启动
    * dbo启动 / baseline  
    * flashcomm1(aicpu) / baseline 
    * flashcomm1(aiv) / baseline  
    * flashcomm1(aicpu) + dbo / baseline  
    * flashcomm1(aiv) + dbo / baseline  
    * flashcomm1 + flashcomm2(aiv) + dbo / baseline 
* DP=2, EP启动（默认用aicpu）
    * dbo启动 / baseline  
* shared expert multistream参数  
    * 该参数 / baseline  
    * dbo + 该参数 / baseline  (先只做dp，TP太多了先不做)


### Qwen3-30B  [P1]

同Deepseekv2一样，测试一遍（除了shared expert参数，这个是deepseek模型特性）  

> 注意，由于qwen3比较大，如果两卡出现存储溢出，可以自行调节gpu-memory-utilization等参数配置。但是注意记录即可。  

