# EdgeVisor: 异构分布式大模型推理框架

## 项目概述

EdgeVisor 是一个针对异构设备环境的分布式大语言模型推理框架，实现了 Qwen3 模型的分布式推理。该框架的核心创新在于支持**异构张量并行**、**动态负载均衡**和**智能初始化算法**，能够在不同计算能力的设备上高效运行大模型推理。

## 核心创新点

相较于传统分布式推理方案（如 distributed-llama），本项目具有以下创新特性：

### 1. 智能初始化算法 (Intelligent Initialization)

框架提供了完整的初始化算法套件，能够根据异构设备的计算能力、内存容量和网络拓扑自动生成最优的分布式配置：

#### 1.1 RRAGC: 设备分组算法 (Root Recognition and Adaptive Group Clustering)

将异构设备自动划分为多个 Virtual Group (VG)，每个 VG 对应一个 Pipeline Stage：

- **Phase 1 - Root 识别**：基于带宽和计算能力的综合评分，选择各 VG 的 Root 节点
- **Phase 2 - 星型聚类**：将 Worker 节点分配到带宽最高的 Root
- **Phase 3 - 可行性约束**：确保每个 VG 满足最小计算/内存约束，必要时迁移节点
- **Phase 4 - Pipeline 排序**：使用 Max-Min Bottleneck 策略确定 Stage 执行顺序

```python
from init_algorithm import RRAGC, Device, Link, RRAGCConfig

devices = [
    Device(id=0, compute=100.0, memory=16.0),
    Device(id=1, compute=50.0, memory=8.0),
    Device(id=2, compute=90.0, memory=16.0),
]

links = [
    Link(0, 1, 10.0),  # 10 Gbps
    Link(1, 0, 10.0),
    Link(0, 2, 5.0),
]

config = RRAGCConfig(K=2, P_min=50.0, M_min=8.0)
result = RRAGC.solve(devices, links, config)
# result.vg_roots: [0, 2] - Root 节点
# result.device_to_vg_map: {0: 0, 1: 0, 2: 1} - 设备分组
# result.pipeline_order: [0, 1] - Stage 执行顺序
```

#### 1.2 CCWF: Stage 内负载分配算法 (Concurrent Compute-Weighted Fair)

在 Stage 内部，根据各 Worker 的计算能力和网络带宽，计算最优的负载分配比例：

- **输入**：各 Worker 的计算能力 (FLOPS)、带宽 (bps)、内存约束
- **输出**：各 Worker 的负载分配比例 (alpha)
- **目标**：最小化 Stage 内的总延迟

```python
from init_algorithm import solve_ccwf, WorkerProfile, LayerTask

workers = [
    WorkerProfile(dev_id=0, compute_flops=100e12, bandwidth_bps=10e9, max_alpha_mem=1.0),
    WorkerProfile(dev_id=1, compute_flops=50e12, bandwidth_bps=10e9, max_alpha_mem=1.0),
]

task = LayerTask(input_bytes=1e6, output_bytes=1e6, total_flops=1e12)
result = solve_ccwf(workers, task)
# result.alphas: [0.67, 0.33] - 负载分配比例
# result.estimated_latency: 0.015 - 预估延迟 (秒)
```

#### 1.3 OLP: Stage 间层划分算法 (Optimal Layer Partition)

使用动态规划算法，在 Stage 之间最优分配 Transformer 层数：

- **输入**：各 VG 的单层处理时间、内存容量、Stage 间带宽
- **输出**：各 Stage 分配的层数
- **目标**：最小化 Pipeline 瓶颈延迟

```python
from init_algorithm import solve_layer_partition, VGProfile, ModelConfig

vgs = [
    VGProfile(vg_id=0, unit_time_ms=15.0, max_layers_capacity=20, next_link_bw_gbps=10.0),
    VGProfile(vg_id=1, unit_time_ms=20.0, max_layers_capacity=15, next_link_bw_gbps=5.0),
]

model = ModelConfig(total_layers=24, activation_size_gb=0.1)
allocation = solve_layer_partition(vgs, model)
# allocation: [14, 10] - Stage 0 分配 14 层，Stage 1 分配 10 层
```

### 2. 异构张量并行 (Heterogeneous Tensor Parallelism)

传统张量并行假设所有设备具有相同的计算能力，将模型参数均匀分割到各设备。本项目实现了**非均匀分割**：

- **按比例分配注意力头**：通过 `set_split_by_ratio()` 方法，可以根据设备算力动态分配不同数量的 Query Head 给各个设备
- **灵活的 FFN 分割**：Feed-Forward Network 的隐藏层维度也可以按比例非均匀分割
- **示例配置**：Stage 0 使用 2 个设备（TP=2），Stage 1 使用 3 个设备（TP=3），实现了跨阶段的异构配置

```python
ratios = [3, 2]
block.att.set_split_by_ratio(ratios)
```

### 3. 冗余 KV 维护机制 (Redundant KV Maintenance)

针对 GQA (Grouped Query Attention) 架构的创新设计：

- **维护头超集**：每个设备维护一个 Query Head 超集的 KV Cache，而非仅维护活跃头
- **动态活跃头切换**：可以在不重新计算 KV 的情况下，动态切换活跃计算的 Query Head
- **KV 索引映射**：通过 `maintained_kv_indices` 和 `active_q_heads_indices` 实现灵活的注意力计算

```python
block.att.set_maintained_heads([0, 1, 2, 3])
block.att.set_active_heads([0, 2])
```

### 4. 流水线并行与张量并行混合 (Hybrid PP + TP)

- **多阶段流水线**：支持任意数量的 Pipeline Stage
- **阶段内张量并行**：每个阶段内部使用张量并行，支持不同 TP 度
- **P2P 通信**：阶段间通过 `dist.send/recv` 传递激活值

### 5. 自定义层划分 (Custom Layer Partitioning)

支持非均匀的层划分，可以根据各 Stage 设备的计算能力灵活分配不同数量的层：

```python
model_stage0 = DistributedQwen3Model(cfg, dist_config, managed_layers_range=(0, 10))
model_stage1 = DistributedQwen3Model(cfg, dist_config, managed_layers_range=(10, 36))
```

### 6. Stage 内动态负载均衡 (Intra-Stage Dynamic Load Balancing)

框架实现了 **Stage 内部** 的动态负载均衡，能够根据各设备的实时执行时间自动调整 **Attention Heads** 和 **FFN Dims** 的分配：

#### 自动化流程

1. **时间统计**：每个 Worker 统计自己的执行时间
2. **Root 汇聚**：Root 节点通过 `all_gather` 收集所有 Worker 的时间
3. **算法执行**：Root 运行 `rebalance_intra_stage` 算法
4. **策略广播**：Root 通过 `broadcast` 将新策略发送给所有 Worker
5. **策略应用**：下一次 forward 时自动应用新策略

```python
from rebalance_algo import DeviceStatus, rebalance_intra_stage

devices = [
    DeviceStatus(
        device_id=0,
        execution_time_ms=100.0,
        current_head_start=0,
        current_head_end=8,
        current_ffn_start=0,
        current_ffn_end=5504,
        kv_head_holding_start=0,
        kv_head_holding_end=8
    ),
    DeviceStatus(
        device_id=1,
        execution_time_ms=50.0,
        current_head_start=8,
        current_head_end=16,
        current_ffn_start=5504,
        current_ffn_end=11008,
        kv_head_holding_start=6,
        kv_head_holding_end=16
    )
]

new_head_counts, new_ffn_counts = rebalance_intra_stage(devices)
```

#### 算法特点

- **Stage 内部迁移**：迁移单位是 Attention Heads 和 FFN Dims
- **比例同步**：FFN Dims 按 Head 迁移比例同步调整
- **KV Cache 约束感知**：Head 迁移受 KV Cache 边界限制
- **双向迁移**：瓶颈设备可同时向左右邻居迁移负载

### 7. Stage 间动态负载均衡 (Inter-Stage Dynamic Load Balancing)

框架实现了 **Stage 之间** 的动态层迁移，能够根据各 Stage 的实时执行时间自动调整 **Transformer Layers** 的分配：

#### IBSA: Inter-VG Bottleneck Smoothing Algorithm

IBSA 算法通过在相邻 Stage 之间迁移 Layer 来降低 Pipeline 瓶颈延迟：

1. **识别瓶颈**：找到执行时间最长的 Stage
2. **邻居搜索**：找到瓶颈 Stage 的最快邻居
3. **收益预测**：模拟迁移 1 层后的新延迟
4. **滞后检查**：只有当收益超过阈值时才执行迁移，避免振荡

```python
from intervg_dynamic import solve_ibsa

current_layers = [10, 10, 10]
execution_times = [100.0, 200.0, 120.0]  # Stage 1 是瓶颈

new_layers, changed, src, dst = solve_ibsa(
    current_layers,
    execution_times,
    threshold_ratio=0.05
)

if changed:
    print(f"Moved 1 layer from Stage {src} to Stage {dst}")
    print(f"New layers: {new_layers}")  # [10, 9, 11]
```

#### 算法特点

- **Stage 间迁移**：迁移单位是 Transformer Layers
- **相邻迁移**：只允许相邻 Stage 之间迁移，减少通信开销
- **滞后机制**：通过 `threshold_ratio` 避免频繁迁移导致的振荡
- **单位成本估算**：使用 `mu = time / layers` 估算单层处理时间

## 项目结构

```
EdgeVisor/
├── distributed_qwen3.py       # 核心分布式模型实现
├── init_algorithm.py          # 初始化算法 (RRAGC, CCWF, OLP)
├── rebalance_algo.py          # Stage 内动态负载均衡算法
├── intervg_dynamic.py         # Stage 间动态负载均衡算法 (IBSA)
├── test_distributed_qwen3.py  # 测试用例
├── infer_config.py            # 模型配置推断工具
├── check_bias.py              # 权重偏置检查工具
└── README.md                  # 本文档
```

## 文件说明

| 文件 | 功能 |
|------|------|
| `distributed_qwen3.py` | 分布式 Qwen3 模型实现，包含 DistributedConfig、DistributedGroupedQueryAttention、DistributedFeedForward、DistributedTransformerBlock、DistributedQwen3Model 等核心组件 |
| `init_algorithm.py` | 初始化算法套件：RRAGC（设备分组）、CCWF（Stage内负载分配）、OLP（Stage间层划分） |
| `rebalance_algo.py` | Stage 内动态负载均衡算法，迁移单位为 Attention Heads 和 FFN Dims |
| `intervg_dynamic.py` | Stage 间动态负载均衡算法 (IBSA)，迁移单位为 Transformer Layers |
| `test_distributed_qwen3.py` | 端到端测试用例，验证分布式推理正确性和负载均衡算法 |
| `infer_config.py` | 从模型权重推断模型配置参数 |
| `check_bias.py` | 检查模型权重中是否存在 bias 参数 |

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

### 3. 使用初始化算法

```python
from init_algorithm import run_initialization, Device, Link, RRAGCConfig, LayerTask, ModelConfig

devices = [
    Device(id=0, compute=100.0, memory=16.0),
    Device(id=1, compute=50.0, memory=8.0),
    Device(id=2, compute=90.0, memory=16.0),
    Device(id=3, compute=40.0, memory=4.0),
    Device(id=4, compute=30.0, memory=4.0),
]

links = [
    Link(0, 1, 10.0), Link(1, 0, 10.0),
    Link(0, 2, 5.0), Link(2, 0, 5.0),
    Link(2, 3, 8.0), Link(3, 2, 8.0),
    Link(1, 4, 6.0), Link(4, 1, 6.0),
]

rragc_config = RRAGCConfig(K=2, P_min=50.0, M_min=4.0)
layer_task = LayerTask(input_bytes=1e6, output_bytes=1e6, total_flops=1e12)
model_config = ModelConfig(total_layers=24, activation_size_gb=0.1)

result = run_initialization(devices, links, rragc_config, layer_task, model_config)

print(f"VG Roots: {result.rragc_result.vg_roots}")
print(f"Device to VG: {result.rragc_result.device_to_vg_map}")
print(f"Pipeline Order: {result.rragc_result.pipeline_order}")
print(f"Layer Allocation: {result.intervg_layer_allocation}")
```

### 4. 自定义分布式配置

```python
stage_ranks = [[0, 1], [2, 3, 4]]
tp_ranks_per_stage = [2, 3]

dist_config = DistributedConfig(rank, world_size, stage_ranks, tp_ranks_per_stage)
```

## 测试结果

```
--- Testing Rebalance Algorithm (Intra-Stage Heads/FFN) ---
Rebalanced Head Counts: [6, 10]
Rebalanced FFN Counts: [4128, 6880]
Algorithm Verification SUCCESS

Rank 2: Max Difference (Shifted [10, 14]) = 0.0002288818359375
Rank 2: SUCCESS! Output matches reference.
```

## 架构说明

```
┌─────────────────────────────────────────────────────────┐
│                    Stage 0 (Layers 0-11)                │
│  ┌─────────────┐         ┌─────────────┐               │
│  │   Rank 0    │  TP=2   │   Rank 1    │               │
│  │  (Root)     │◄───────►│  (Worker)   │               │
│  │  Q: 0-3     │ AllReduce│  Q: 4-6    │               │
│  └─────────────┘         └─────────────┘               │
└──────────────────────────┬──────────────────────────────┘
                           │ P2P Send/Recv (Root to Root)
                           ▼
┌─────────────────────────────────────────────────────────┐
│                   Stage 1 (Layers 12-23)                │
│  ┌─────────┐    ┌─────────┐    ┌─────────┐            │
│  │ Rank 2  │    │ Rank 3  │    │ Rank 4  │   TP=3     │
│  │ (Root)  │    │(Worker) │    │(Worker) │            │
│  │Q: 0-2   │    │Q: 3-4   │    │Q: 5-6   │            │
│  └─────────┘    └─────────┘    └─────────┘            │
│        ◄─────────── AllReduce ───────────►             │
└─────────────────────────────────────────────────────────┘
```

## Root 节点职责

每个 Stage 内的 Root 节点（`tp_rank=0`）承担以下职责：

1. **Stage 间通信**：接收上一个 Stage 的激活值，发送到下一个 Stage
2. **时间收集**：通过 `all_gather` 收集所有 Worker 的执行时间
3. **负载均衡决策**：运行 `rebalance_intra_stage` 算法
4. **策略广播**：将新的负载分配策略广播给所有 Worker
5. **结果输出**：最后一个 Stage 的 Root 负责输出最终 logits

## 注意事项

1. 当前实现使用 `gloo` 后端，适用于 CPU 测试；生产环境建议使用 `nccl` 后端
2. 模型路径需要根据实际环境修改
3. 动态负载均衡算法需要配合设备执行时间监控使用
4. KV Cache 冗余范围决定了 Head 迁移的边界约束

## 许可证

MIT License
