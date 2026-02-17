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

### 4. 自定义层划分 (Custom Layer Partitioning)

支持非均匀的层划分，可以根据各 Stage 设备的计算能力灵活分配不同数量的层：

```python
# Stage 0: 负责 layers 0-9 (10层) - 算力较弱的设备
model_stage0 = DistributedQwen3Model(cfg, dist_config, managed_layers_range=(0, 10))

# Stage 1: 负责 layers 10-35 (26层) - 算力较强的设备
model_stage1 = DistributedQwen3Model(cfg, dist_config, managed_layers_range=(10, 36))
```

如果不指定 `managed_layers_range`，则默认使用均匀分配策略。

### 5. Stage 内动态负载均衡算法 (Intra-Stage Load Balancing)

框架实现了 **Stage 内部** 的动态负载均衡算法，能够根据各设备的实时执行时间自动调整 **Attention Heads** 和 **FFN Dims** 的分配：

#### 算法核心逻辑

1. **瓶颈识别**：找到执行时间最长的设备（瓶颈设备）
2. **Heads 迁移计算**：基于单位 Head 执行时间，计算需要向左右邻居迁移的 Head 数量
3. **FFN 比例迁移**：FFN Dims 按 Head 迁移比例同步调整
4. **KV Cache 约束**：确保 Head 迁移后的范围不超过设备已缓存的 KV 范围

#### 使用示例

```python
from rebalance_algo import DeviceStatus, rebalance_intra_stage

# 定义 Stage 内设备状态
devices = [
    DeviceStatus(
        device_id=0,
        execution_time_ms=100.0,       # 瓶颈设备
        current_head_start=0,
        current_head_end=8,            # 当前负责 8 个 Heads
        current_ffn_start=0,
        current_ffn_end=5504,          # 当前负责 5504 个 FFN Dims
        kv_head_holding_start=0,
        kv_head_holding_end=8          # KV Cache 覆盖的 Head 范围
    ),
    DeviceStatus(
        device_id=1,
        execution_time_ms=50.0,        # 较快设备
        current_head_start=8,
        current_head_end=16,           # 当前负责 8 个 Heads
        current_ffn_start=5504,
        current_ffn_end=11008,         # 当前负责 5504 个 FFN Dims
        kv_head_holding_start=6,       # 有 2 个 Head 的冗余 KV
        kv_head_holding_end=16
    )
]

# 执行 Stage 内负载均衡
new_head_counts, new_ffn_counts = rebalance_intra_stage(devices)
# 结果: Heads [6, 10], FFN [4128, 6880]
# 从设备0迁移 2 个 Heads 和 1376 个 FFN Dims 到设备1
```

#### 算法特点

- **Stage 内部迁移**：迁移单位是 Attention Heads 和 FFN Dims，而非 Transformer Layers
- **比例同步**：FFN Dims 按 Head 迁移比例同步调整，保持计算负载均衡
- **KV Cache 约束感知**：Head 迁移受 KV Cache 边界限制，避免缓存失效
- **双向迁移**：瓶颈设备可同时向左右邻居迁移负载

## 项目结构

```
EdgeVisor/
├── distributed_qwen3.py    # 核心分布式模型实现
├── rebalance_algo.py       # 动态负载均衡算法
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
--- Testing Rebalance Algorithm (Intra-Stage Heads/FFN) ---
Initial Devices: [DeviceStatus(device_id=0, execution_time_ms=100.0, ...), DeviceStatus(device_id=1, execution_time_ms=50.0, ...)]
Rebalanced Head Counts: [6, 10]
Rebalanced FFN Counts: [4128, 6880]
Algorithm Verification SUCCESS: Heads and FFN rebalanced proportionally.

Rank 2: Max Difference (Shifted [10, 14]) = 0.0002288818359375
Rank 2: SUCCESS! Output matches reference.
Rank 2: Generated IDs: [99226, 102168, 108645, 112116, 71138, 29285, 11, 220, 4018, 28946]
```

测试验证了：
1. **Stage 内负载均衡算法正确性**：Heads 从 [8, 8] 迁移到 [6, 10]，FFN 按比例同步迁移
2. **分布式推理一致性**：最大误差约 2.3e-4，在浮点精度范围内
3. **动态层切换**：运行时成功切换层分配配置（[12, 12] → [10, 14]）
4. **KV Cache 约束**：Head 迁移受 KV Cache 边界限制（设备1 可接受 Head 6-16）

## 测试用例分析

测试代码覆盖以下场景：

1. **Stage 内负载均衡算法验证**：测试 `rebalance_intra_stage()` 算法在瓶颈场景下的 Heads/FFN 迁移计算
2. **权重加载正确性**：从 HuggingFace safetensors 格式加载权重到分布式模型
3. **分布式通信正确性**：验证 AllReduce、Broadcast、Send/Recv 操作
4. **流水线执行正确性**：验证两阶段流水线的激活值传递
5. **动态层切换**：运行时从 [12, 12] 切换到 [10, 14] 层分配
6. **自回归生成**：测试连续生成 10 个 token 的完整流程
7. **数值一致性**：分布式输出与单进程参考模型对比

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
3. 动态负载均衡算法需要配合设备执行时间监控使用
4. KV Cache 冗余范围决定了层迁移的边界约束

## 许可证

MIT License
