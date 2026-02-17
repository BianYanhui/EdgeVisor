# EdgeVisor: 异构分布式大模型推理框架

## 项目概述

EdgeVisor 是一个针对异构设备环境的分布式大语言模型推理框架，实现了 Qwen3 模型的分布式推理。该框架的核心创新在于支持**异构张量并行**和**动态负载均衡**，能够在不同计算能力的设备上高效运行大模型推理。

## 核心创新点

相较于传统分布式推理方案，本项目具有以下创新特性：

### 1. 异构张量并行 (Heterogeneous Tensor Parallelism)

传统张量并行假设所有设备具有相同的计算能力，将模型参数均匀分割到各设备。本项目实现了**非均匀分割**：

- **按比例分配注意力头**：通过 `set_split_by_ratio()` 方法，可以根据设备算力动态分配不同数量的 Query Head 给各个设备
- **灵活的 FFN 分割**：Feed-Forward Network 的隐藏层维度也可以按比例非均匀分割
- **示例配置**：Stage 0 使用 2 个设备（TP=2），Stage 1 使用 3 个设备（TP=3），实现了跨阶段的异构配置

```python
# 示例：按 3:2 比例分配工作负载
ratios = [3, 2]  # 设备0承担60%，设备1承担40%
block.att.set_split_by_ratio(ratios)
```

### 2. 冗余 KV 维护机制 (Redundant KV Maintenance)

针对 GQA (Grouped Query Attention) 架构的创新设计：

- **维护头超集**：每个设备维护一个 Query Head 超集的 KV Cache，而非仅维护活跃头
- **动态活跃头切换**：可以在不重新计算 KV 的情况下，动态切换活跃计算的 Query Head
- **KV 索引映射**：通过 `maintained_kv_indices` 和 `active_q_heads_indices` 实现灵活的注意力计算

```python
# 设置维护的 Q Head 超集
block.att.set_maintained_heads([0, 1, 2, 3])
# 动态设置当前活跃的 Q Head 子集
block.att.set_active_heads([0, 2])
```

### 3. 流水线并行与张量并行混合 (Hybrid PP + TP)

- **两阶段流水线**：24 层模型分为 Stage 0（层 0-11）和 Stage 1（层 12-23）
- **阶段内张量并行**：每个阶段内部使用张量并行，支持不同 TP 度
- **P2P 通信**：阶段间通过 `dist.send/recv` 传递激活值

### 4. 动态负载均衡接口

框架预留了动态负载均衡的接口：

```python
def update_load_balance_policy(self, layer_idx, ratios):
    # 运行时动态调整各设备的负载比例
    block.att.set_split_by_ratio(ratios)
    block.ff.set_split_by_ratio(ratios)
```

## 项目结构

```
EdgeVisor/
├── distributed_qwen3.py    # 核心分布式模型实现
├── test_distributed_qwen3.py  # 测试用例
├── infer_config.py         # 模型配置推断工具
├── check_bias.py           # 权重偏置检查工具
└── README.md               # 本文档
```

## 环境要求

- Python 3.8+
- PyTorch 2.0+
- safetensors
- transformers (用于 tokenizer)

安装依赖：

```bash
pip install torch safetensors transformers
```

## 使用方法

### 1. 准备模型

下载 Qwen3-0.6B 模型到本地路径，并修改 `test_distributed_qwen3.py` 中的 `model_path`：

```python
model_path = "/path/to/your/Qwen-3-0.6B"
```

### 2. 运行分布式推理测试

```bash
python test_distributed_qwen3.py
```

该测试会：
1. 启动 5 个进程模拟 5 个设备
2. 配置：Stage 0 使用 2 设备（Rank 0-1），Stage 1 使用 3 设备（Rank 2-4）
3. 加载模型权重并执行分布式推理
4. 与单进程参考模型对比验证正确性

### 3. 自定义分布式配置

修改 `DistributedConfig` 参数：

```python
# 定义阶段和设备分配
stage_ranks = [[0, 1], [2, 3, 4]]  # Stage 0: 2设备, Stage 1: 3设备
tp_ranks_per_stage = [2, 3]         # 对应的 TP 度

dist_config = DistributedConfig(rank, world_size, stage_ranks, tp_ranks_per_stage)
```

### 4. 动态调整负载比例

```python
# 在推理过程中动态调整
model.update_load_balance_policy(layer_idx=5, ratios=[3, 2])  # 适用于 Stage 0
model.update_load_balance_policy(layer_idx=15, ratios=[2, 2, 1])  # 适用于 Stage 1
```

## 测试结果

测试用例验证了分布式推理的正确性：

```
Rank 2: Max Difference = 0.00022363662719726562
Rank 2: SUCCESS! Output matches reference.
Rank 2: Generated IDs: [108230, 235, 3837, 1694, 279, 39635, 102168, 99222, 61213, 90840]
```

最大误差约为 2.2e-4，在浮点精度范围内，证明分布式实现与单进程参考模型输出一致。

## 测试用例分析

测试代码覆盖以下场景：

1. **权重加载正确性**：从 HuggingFace safetensors 格式加载权重到分布式模型
2. **分布式通信正确性**：验证 AllReduce、Broadcast、Send/Recv 操作
3. **流水线执行正确性**：验证两阶段流水线的激活值传递
4. **自回归生成**：测试连续生成 10 个 token 的完整流程
5. **数值一致性**：分布式输出与单进程参考模型对比

## 架构说明

### 分布式配置示例

```
┌─────────────────────────────────────────────────────────┐
│                    Stage 0 (Layers 0-11)                │
│  ┌─────────────┐         ┌─────────────┐               │
│  │   Rank 0    │  TP=2   │   Rank 1    │               │
│  │  (TP=0)     │◄───────►│  (TP=1)     │               │
│  │  Q: 0-3     │ AllReduce│  Q: 4-6    │               │
│  └─────────────┘         └─────────────┘               │
└──────────────────────────┬──────────────────────────────┘
                           │ P2P Send/Recv
                           ▼
┌─────────────────────────────────────────────────────────┐
│                   Stage 1 (Layers 12-23)                │
│  ┌─────────┐    ┌─────────┐    ┌─────────┐            │
│  │ Rank 2  │    │ Rank 3  │    │ Rank 4  │   TP=3     │
│  │(TP=0)   │    │(TP=1)   │    │(TP=2)   │            │
│  │Q: 0-2   │    │Q: 3-4   │    │Q: 5-6   │            │
│  └─────────┘    └─────────┘    └─────────┘            │
│        ◄─────────── AllReduce ───────────►             │
└─────────────────────────────────────────────────────────┘
```

## 注意事项

1. 当前实现使用 `gloo` 后端，适用于 CPU 测试；生产环境建议使用 `nccl` 后端
2. 模型路径需要根据实际环境修改
3. 动态负载均衡算法需要根据实际监控数据实现

## 许可证

MIT License
