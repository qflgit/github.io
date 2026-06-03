# 大模型推理优化技术全景解析

本文系统梳理大语言模型（LLM）推理阶段的核心优化技术，涵盖 KV Cache 管理、投机解码、并行加速、量化压缩、通信优化等多个维度，从算法原理到工程实现进行全面解析。

---

## 一、推理优化的核心挑战

### 1.1 推理阶段的两个阶段

大模型推理分为两个截然不同的计算阶段，优化策略各有侧重：

**预填充阶段（Prefill）**
- 一次性处理整个输入提示词（Prompt）
- 计算密集的矩阵乘法，GPU 计算单元利用率接近 100%
- 主要瓶颈：计算吞吐量（Throughput）

**解码阶段（Decode）**
- 自回归生成，每次只生成 1 个 token
- 内存带宽密集型，GPU 大部分时间等待显存数据加载
- 主要瓶颈：内存带宽（Memory Bandwidth）

```
Prefill:  [Prompt: 1000 tokens]  -->  计算密集，可并行
          ↓
Decode:   [T1] -> [T2] -> [T3] -> ...  内存带宽密集，串行
          ↑
    每次只生成1个token，KV Cache不断增长
```

### 1.2 核心性能指标

| 指标 | 定义 | 优化目标 |
|------|------|----------|
| **TTFT** (Time To First Token) | 从请求到第一个输出生成的时间 | 降低 Prefill 延迟 |
| **TPOT** (Time Per Output Token) | 每个输出 token 的生成时间 | 降低 Decode 延迟 |
| **Throughput** | 每秒处理的请求数或 token 数 | 提升整体吞吐 |
| **KV Cache 占用** | 存储历史 KV 的显存大小 | 减少显存压力 |

---

## 二、KV Cache 优化：推理的显存命脉

### 2.1 KV Cache 的本质

在自回归解码中，为了避免重复计算历史 token 的 Key 和 Value，模型会将它们缓存起来：

```
标准 Attention 计算：
  Attention(Q_t, K_all, V_all) = softmax(Q_t · K_all^T / sqrt(d)) · V_all

其中 K_all = [K_1, K_2, ..., K_t]  需要缓存所有历史 K
      V_all = [V_1, V_2, ..., V_t]  需要缓存所有历史 V
```

**KV Cache 体积公式**：
```
KV Cache (bytes) = 2 × num_layers × num_heads × head_dim × seq_len × batch_size × precision_bytes

示例：Llama-3-8B, batch=1, seq_len=4096, FP16
  = 2 × 32层 × 8头 × 128维 × 4096 × 1 × 2字节
  ≈ 512 MB
```

当 batch_size=32, seq_len=128K 时，KV Cache 可达 **16 GB+**，成为显存的主要消耗者。

### 2.2 多查询注意力（MQA）与分组查询注意力（GQA）

#### 2.2.1 MQA（Multi-Query Attention）

```
传统 MHA：每个注意力头有独立的 K, V 投影
  K_h = X · W_K_h    (h = 1, 2, ..., num_heads)
  V_h = X · W_V_h

MQA：所有头共享同一组 K, V
  K_shared = X · W_K    (仅1组)
  V_shared = X · W_V    (仅1组)

  每个头的 Q 仍独立：Q_h = X · W_Q_h
  但 Attention(Q_h, K_shared, V_shared) 共享 KV
```

**效果**：KV Cache 减少为原来的 **1/num_heads**（如 8 头则减少 8 倍）

**代价**：模型表达能力轻微下降，需通过训练补偿

**代表模型**：PaLM、ChatGLM

#### 2.2.2 GQA（Grouped-Query Attention）

```
折中方案：将 num_heads 个 Q 头分为 num_groups 组
  每组共享 1 组 K, V

  例如：32 个 Q 头分为 8 组，每组 4 个 Q 头共享 1 组 KV
  KV Cache 减少为原来的 1/4
```

**效果**：在 MHA 和 MQA 之间取得平衡，KV Cache 减少 4~8 倍，性能损失极小

**代表模型**：Llama-2/3、Qwen、Mistral

### 2.3 KV Cache 量化

#### 2.3.1 精度降级

```
FP16 (2字节) → FP8 (1字节) → FP4 (0.5字节)

FP8 方案（主流）：
  - E4M3 格式：1位符号 + 4位指数 + 3位尾数
  - 动态缩放：每通道（per-channel）或每 token（per-token）缩放因子
  - H100/B200 原生支持，零额外开销

FP4 方案（前沿）：
  - 1位符号 + 2位指数 + 1位尾数
  - 需要量化感知训练（QAT）补偿精度损失
  - DeepSeek V4 的 Lightning Indexer 使用 FP4 QK 计算
```

**效果**：FP8 减少 50% 显存，FP4 减少 75% 显存

#### 2.3.2 KV Cache 变换编码（KVTC）

2026 年提出的新方法，借鉴视频压缩思想：

```
1. 将相邻 token 的 KV 视为"帧序列"
2. 对 KV 序列做 DCT/小波变换
3. 保留低频分量（携带主要语义信息）
4. 丢弃高频分量（多为噪声）
5. 需要时逆变换还原
```

**效果**：2:1 压缩几乎无损，4:1 压缩损失可控

### 2.4 KV Cache 稀疏化与驱逐

#### 2.4.1 H2O（Heavy Hitter Oracle）

核心观察：80% 的注意力权重集中在 20% 的 token 上

```
策略：
1. 维护一个"重要 token"集合（Heavy Hitters）
2. 只保留这些 token 的完整 KV
3. 其他 token 的 KV 丢弃或压缩存储

判定标准：
  - 累计 attention 得分最高的 token
  - 近期新生成的 token（保留局部性）
  - 特殊 token（如系统提示词、文档标题）
```

**效果**：保留 20% KV，精度损失 < 2%

#### 2.4.2 SnapKV

```
策略：
1. 观察窗口（Observation Window）：分析最近 N 个 token 的 attention 模式
2. 识别"关键 token"：对未来生成影响最大的历史 token
3. 只保留关键 token 的 KV，其余驱逐
4. 关键 token 数量可配置（如保留 20%）
```

**优势**：无需修改模型，纯推理时优化

#### 2.4.3 前缀缓存（Prefix Caching）

```
系统级缓存策略：

System Cache：
  - 缓存系统提示词（System Prompt）的 KV
  - 多用户/多轮对话共享
  - 避免重复计算相同的系统指令

Document Cache：
  - 缓存上传文档的 KV
  - 后续查询直接复用，无需重新编码

Conversation Cache：
  - 缓存历史对话轮次的 KV
  - 新轮次只需计算新增 token
```

**效果**：TTFT 降低 5~10 倍，特别适合多轮对话和 RAG 场景

### 2.5 PagedAttention（vLLM）

#### 2.5.1 核心思想

借鉴操作系统虚拟内存的"分页"思想管理 KV Cache：

```
传统方式：
  每个序列预分配连续显存块 [max_seq_len × kv_size]
  问题：内存碎片、无法共享前缀、利用率低

PagedAttention：
  将 KV Cache 划分为固定大小的"页"（Page，如 16 tokens）
  用块表（Block Table）管理虚拟到物理的映射

  序列 A: [Page_0] -> [Page_1] -> [Page_2]  (物理上可不连续)
  序列 B: [Page_0] -> [Page_3] -> [Page_4]  (共享 Page_0，Copy-on-Write)
```

#### 2.5.2 关键特性

| 特性 | 说明 | 效果 |
|------|------|------|
| **按需分配** | 不预分配最大长度，用多少分配多少 | 减少显存浪费 50%+ |
| **内存共享** | 多个序列共享公共前缀（如系统提示） | 多请求并行时显存节省显著 |
| **Copy-on-Write** | 分叉序列共享页直到某一方修改 | 树形解码（如 Beam Search）高效支持 |
| **碎片整理** | 后台合并空闲页 | 长期运行稳定性提升 |

**效果**：吞吐量提升 2~4 倍，支持更高的 batch size

### 2.6 多级缓存系统（IMPRESS）

2026 年提出的 GPU-CPU-磁盘三级缓存架构：

```
GPU 显存（HBM）：
  - 存放当前活跃请求的 KV
  - 高频访问的共享前缀

CPU 内存（DRAM）：
  - 存放近期使用过的 KV
  - 等待处理的请求队列

磁盘存储（SSD/NVMe）：
  - 存放历史会话的完整 KV
  - 大规模文档的编码结果

智能迁移策略：
  - 根据 attention 权重、访问频率、时效性自动分层
  - 需要时从下层加载到上层
```

**适用场景**：超长文档 RAG、多用户会话管理、企业级部署

---

## 三、投机解码（Speculative Decoding）：加速自回归生成

### 3.1 核心思想

自回归生成的瓶颈在于**每次只生成 1 个 token**，而 GPU 的计算能力远未被充分利用。投机解码的核心策略是：

```
用轻量"草稿模型"（Draft Model）快速生成多个候选 token
用大型"目标模型"（Target Model）一次性验证这些候选
接受正确的，拒绝错误的，从错误处重新生成

效果：用 1 次目标模型前向传播换取 3~5 个 token 的生成
```

### 3.2 标准投机解码流程

```
输入：前缀 P = [t1, t2, ..., tn]

Step 1: 草稿模型生成 K 个候选 token
  d1 = DraftModel(P)
  d2 = DraftModel(P + d1)
  d3 = DraftModel(P + d1 + d2)
  ...
  候选序列 D = [d1, d2, d3, ..., dK]

Step 2: 目标模型并行验证
  将前缀 P 和候选 D 拼接，一次性输入目标模型
  得到概率分布：TargetModel(P + D)

  对每个位置 i：
    如果 TargetModel 的分布与 DraftModel 一致（或概率足够高）：
      接受 token di
    否则：
      拒绝 token di，从 TargetModel 的分布中重新采样
      停止验证，返回已接受的 token + 新采样的 token

Step 3: 更新前缀，重复
```

### 3.3 草稿模型的选择

| 方案 | 草稿模型 | 适用场景 | 加速比 |
|------|----------|----------|--------|
| **独立小模型** | 参数量为主模型的 1/10~1/100 | 通用场景 | 2~3× |
| **自我投机** | 主模型自身（早退出层） | 无额外模型 | 1.5~2× |
| **多头投机** | 主模型的额外预测头 | 训练时增加少量参数 | 2~3× |
| **MTP（多Token预测）** | 主模型输出层扩展 | 与模型架构深度集成 | 3~5× |

### 3.4 多 Token 预测（MTP）

#### 3.4.1 核心机制

```
传统输出层：
  hidden_state -> Linear -> logits -> token_1

MTP 输出层：
  hidden_state -> Linear -> logits -> token_1
              -> Linear_2 -> logits -> token_2
              -> Linear_3 -> logits -> token_3
              -> Linear_4 -> logits -> token_4

  共享大部分计算（直到最后几层才分离）
  训练时同时预测未来 N 个 token
```

#### 3.4.2 推理时的树形验证

```
草稿模型生成多个候选路径：
  路径 1: [A, B, C, D]
  路径 2: [A, B, C, E]
  路径 3: [A, B, F, G]

目标模型用"树形注意力"（Tree Attention）一次性验证所有路径
接受最长有效路径
```

**代表实现**：
- **SGLang**：支持 MTP 投机解码，与专家并行叠加
- **DeepSeek V4**：MTP + MoE 专家并行，解码吞吐进一步提升

### 3.5 接受率优化

投机解码的加速效果取决于**接受率**（目标模型接受草稿 token 的比例）：

```
加速比 ≈ 1 / (1 - 接受率 + 验证开销)

提升接受率的策略：
1. 温度调整：降低采样温度使分布更集中
2. 草稿-目标对齐训练：用目标模型的输出蒸馏草稿模型
3. 动态 K 调整：根据接受率动态调整候选数量 K
4. 上下文感知：简单上下文用大胆猜测，复杂上下文保守验证
```

---

## 四、并行加速：压榨硬件极限

### 4.1 张量并行（Tensor Parallelism, TP）

#### 4.1.1 核心思想

将单层神经网络的权重矩阵切分到多个 GPU 上，每个 GPU 只存储部分权重：

```
线性层 Y = X · W

按列切分（Column-wise）：
  GPU 0: W[:, 0:half] -> Y0 = X · W[:, 0:half]
  GPU 1: W[:, half:]  -> Y1 = X · W[:, half:]
  Y = Concat(Y0, Y1)

按行切分（Row-wise）：
  GPU 0: W[0:half, :] -> Y0 = X[:, 0:half] · W[0:half, :]
  GPU 1: W[half:, :]  -> Y1 = X[:, half:] · W[half:, :]
  Y = Y0 + Y1
```

#### 4.1.2 Attention 层的并行

```
多头注意力天然适合张量并行：
  每个 GPU 负责一部分注意力头

  GPU 0: heads 0~3 -> Q0, K0, V0 -> Attention0 -> O0
  GPU 1: heads 4~7 -> Q1, K1, V1 -> Attention1 -> O1

  输出投影层 Concat(O0, O1)
```

**通信量**：每层只需要 2 次 AllReduce（输出投影前后），通信开销小

**适用场景**：单节点多 GPU（NVLink 高速互联），适合 70B 以下模型

### 4.2 流水线并行（Pipeline Parallelism, PP）

#### 4.2.1 核心思想

将模型的不同层分配到不同 GPU 上，形成流水线：

```
GPU 0: Layer 1~8   -> 处理完传递给 GPU 1
GPU 1: Layer 9~16  -> 处理完传递给 GPU 2
GPU 2: Layer 17~24 -> 处理完传递给 GPU 3
GPU 3: Layer 25~32 -> 输出
```

#### 4.2.2 气泡问题与缓解

```
朴素流水线：
  GPU 0: [F1] -> [F2] -> [F3] -> ...
  GPU 1:        [F1] -> [F2] -> [F3] -> ...
  GPU 2:               [F1] -> [F2] -> [F3] -> ...

  问题：GPU 空闲时间（气泡）严重

GPipe（微批次）：
  将 batch 拆分为多个微批次（micro-batch）
  流水线填充：
    GPU 0: [F1a][F1b][F1c] -> [F2a][F2b][F2c] -> ...
    GPU 1:       [F1a][F1b][F1c] -> [F2a][F2b][F2c] -> ...

  效果：气泡减少，吞吐量提升

1F1B（One Forward One Backward）：
  前向和反向交错，进一步减少气泡
```

**适用场景**：跨节点部署（网络带宽受限），适合超大模型（100B+）

### 4.3 专家并行（Expert Parallelism, EP）

#### 4.3.1 MoE 模型的专属并行策略

在 MoE 模型中，专家网络数量巨大（如 32,000 个），需要将专家分布到不同 GPU：

```
All-to-All 通信模式：

前向传播：
  1. Token 路由决策（Router 计算）
  2. All-to-All Dispatch：将 token 发送到目标专家所在 GPU
  3. 各 GPU 并行计算其持有的专家
  4. All-to-All Combine：收集各专家的输出

反向传播：
  1. All-to-All 分发梯度
  2. 各专家计算参数梯度
  3. All-to-All 收集梯度
```

#### 4.3.2 EP 规模与硬件匹配

```
EP 规模（Expert Parallelism Size）决定专家分布：
  EP=8：  8 个 GPU，每个 GPU 持有 1/8 的专家
  EP=32： 32 个 GPU，每个 GPU 持有 1/32 的专家
  EP=256：256 个 GPU（前沿配置）

匹配原则：
  - EP 规模应与 GPU 数量匹配
  - 专家数量应能被 EP 规模整除
  - 通信带宽决定 EP 上限（EP 越大，All-to-All 通信量越大）
```

**DeepSeek V4 配置**：
- EP=32~256（高带宽互联环境下）
- 细粒度通信-计算重叠：将 All-to-All 与专家计算融合为流水线
- 效果：通信延迟被计算完全隐藏

### 4.4 序列并行（Sequence Parallelism, SP）

#### 4.4.1 核心思想

将输入序列切分到多个 GPU，每个 GPU 处理一部分 token：

```
输入序列：512K tokens

SP=4 切分：
  GPU 0: tokens 0~128K
  GPU 1: tokens 128K~256K
  GPU 2: tokens 256K~384K
  GPU 3: tokens 384K~512K

Ring Attention（环注意力）：
  各 GPU 按环状传递 KV Cache
  GPU 0 计算完本地 attention 后，将 KV 传给 GPU 1
  GPU 1 结合本地 Q 和收到的 KV 计算 attention
  以此类推，绕环一圈后得到完整 attention 结果
```

**优势**：突破单卡显存限制，支持超长序列（1M+ tokens）

**代价**：通信量大，需要高速互联（NVLink/InfiniBand）

### 4.5 3D 并行组合

实际部署中通常组合多种并行策略：

```
3D 并行 = TP × PP × EP

示例：DeepSeek V4 训练配置
  - TP=8（张量并行，单节点内 8 卡）
  - PP=16（流水线并行，16 个阶段）
  - EP=128（专家并行，128 个 GPU 分担专家）
  - 总 GPU 数 = 8 × 16 × 128 = 16,384 张 H100
```

**调度原则**：
- TP 优先（通信最少，利用 NVLink）
- PP 次之（层间切分，适合跨节点）
- EP 根据 MoE 规模配置
- SP 用于超长序列场景

---

## 五、量化与压缩：降低计算与显存开销

### 5.1 权重量化

#### 5.1.1 训练后量化（PTQ）

```
FP16 权重 → INT8 / INT4 / FP4

对称量化：
  scale = max(abs(W)) / 127
  W_int8 = round(W / scale)
  W_dequant = W_int8 × scale

非对称量化：
  scale = (max(W) - min(W)) / 255
  zero_point = round(-min(W) / scale)
  W_uint8 = round(W / scale) + zero_point
```

**主流方案**：

| 方案 | 位宽 | 精度损失 | 速度提升 | 代表 |
|------|------|----------|----------|------|
| INT8 | 8 bit | < 1% | 1.5~2× | TensorRT, ONNX Runtime |
| INT4 | 4 bit | 2~5% | 2~3× | GPTQ, AWQ |
| FP8 | 8 bit | < 1% | 1.5~2× | H100/B200 原生 |
| FP4 | 4 bit | 3~7% | 2~4× | DeepSeek V4（QAT） |

#### 5.1.2 GPTQ（梯度感知量化）

```
逐层量化，最小化输出误差：

对于每层权重 W：
  1. 计算该层的 Hessian 矩阵（输出对权重的二阶导数）
  2. 按重要性排序权重（Hessian 大的权重更敏感）
  3. 逐列量化，重要权重用更高精度
  4. 量化误差通过后续权重补偿
```

**效果**：4-bit 量化下精度损失 < 3%

#### 5.1.3 AWQ（激活感知权重量化）

```
核心观察：不是所有权重同等重要
  与显著激活（salient activation）相乘的权重更重要

策略：
  1. 分析激活分布，识别"显著"通道
  2. 显著通道的权重用更高精度（或更大缩放）
  3. 其他通道正常量化
```

**效果**：4-bit 下精度优于 GPTQ，推理速度更快

### 5.2 量化感知训练（QAT）

```
训练时模拟低精度：

前向传播：
  W_quant = quantize(W_fp32)   # 模拟量化误差
  Y = X · W_quant              # 用量化权重计算

反向传播：
  梯度用 FP32 精度计算
  更新 FP32 权重

效果：模型学会在低精度下保持精度
```

**DeepSeek V4 应用**：
- MoE 专家权重 FP4 量化
- 通过 QAT 训练，精度损失极小
- 显存节省 2 倍，推理速度提升

### 5.3 剪枝（Pruning）

```
结构化剪枝：
  移除整个神经元/通道/注意力头
  优点：实际减少计算量，加速明显
  缺点：需要重新训练，可能破坏模型结构

非结构化剪枝：
  将接近 0 的权重置零
  优点：简单，压缩率高（可达 90%）
  缺点：需要稀疏计算支持，实际加速有限
```

**稀疏注意力剪枝**：
- 移除不重要的注意力头（如部分 GQA 实现）
- 动态稀疏：根据输入动态决定哪些头参与计算

---

## 六、通信优化：隐藏延迟的艺术

### 6.1 通信-计算重叠

#### 6.1.1 原理

```
GPU 执行时间线：

无重叠：
  [计算] -> [等待通信] -> [计算] -> [等待通信]

有重叠：
  [计算] + [通信在后台进行]

关键：通信时间 < 计算时间
→ 通信被完全隐藏在计算之下
→ 有效零通信开销
```

#### 6.1.2 DeepSeek V4 的融合调度

```
MoE 层的通信-计算流水线：

时间轴 ->

GPU 0: [Dispatch通信] -> [Linear-1计算] -> [Combine通信] -> [Linear-2计算]
GPU 1: [Dispatch通信] -> [Linear-1计算] -> [Combine通信] -> [Linear-2计算]
GPU 2: [Dispatch通信] -> [Linear-1计算] -> [Combine通信] -> [Linear-2计算]

融合内核：
  将 All-to-All 通信与矩阵乘法融合为单个 CUDA 内核
  通信和计算在同一个内核中交错执行
  最大化 GPU 利用率
```

**效果**：端到端延迟降低 30~50%

### 6.2 集合通信优化

#### 6.2.1 AllReduce 算法选择

```
Ring AllReduce：
  适合大消息、高带宽环境
  每个 GPU 只需与邻居通信

Tree AllReduce：
  适合小消息、高延迟环境
  分层聚合，减少跳数

Hierarchical AllReduce：
  节点内 Ring，节点间 Tree
  适合多节点集群
```

#### 6.2.2 NCCL 优化

```
NVIDIA Collective Communications Library：
  - 自动选择最优通信算法
  - 支持 NVLink、InfiniBand、TCP
  - 拓扑感知路由

优化策略：
  - 合并小消息（Message Fusion）
  - 双缓冲（Double Buffering）
  - 自适应路由（Adaptive Routing）
```

### 6.3 内存带宽优化

#### 6.3.1 FlashAttention

```
核心思想：IO-Aware 算法，减少 HBM 访问

标准 Attention：
  1. 加载 Q, K, V 到 HBM
  2. 计算 S = QK^T，写回 HBM
  3. 加载 S，计算 P = softmax(S)，写回 HBM
  4. 加载 P, V，计算 O = PV，写回 HBM

  HBM 读写次数：O(N)

FlashAttention：
  1. 将 Q, K, V 分块（Tile）
  2. 每次加载一小块到 SRAM（高速缓存）
  3. 在 SRAM 中完成全部计算
  4. 只写回最终结果 O

  HBM 读写次数：O(1)
```

**效果**：
- 内存带宽需求降低 5~10 倍
- 实际速度提升 2~4 倍（受 SRAM 容量限制）
- 支持更长序列（受 HBM 容量限制而非带宽）

**演进**：
- FlashAttention-2：减少非矩阵乘法 FLOPs，更好的并行性
- FlashAttention-3：Hopper 架构优化，异步拷贝，FP8 支持
- FlashDecoding：针对 Decode 阶段的优化，拆分 KV Cache

#### 6.3.2 PageAttention（续）

```
内存带宽优化：
  - 非连续内存访问优化
  - 共享页减少重复加载
  - Copy-on-Write 避免不必要拷贝
```

---

## 七、编译器与调度优化

### 7.1 图优化

#### 7.1.1 算子融合

```
融合前：
  [Linear] -> [LayerNorm] -> [Activation] -> [Dropout]
  4 个内核启动，4 次 HBM 读写

融合后：
  [FusedLinearNormActDropout]
  1 个内核启动，1 次 HBM 读写
```

**常见融合模式**：
- Linear + Bias + Activation
- Attention QKV 投影融合
- MoE 的 Router + Dispatch 融合
- LayerNorm + Residual 融合

#### 7.1.2 常量折叠与死代码消除

```
编译时优化：
  - 预计算常量表达式
  - 移除训练时遗留但推理时不用的代码
  - 简化控制流
```

### 7.2 动态批处理（Dynamic Batching / Continuous Batching）

#### 7.2.1 核心思想

```
传统批处理：
  等待 N 个请求到达，一起处理
  问题：短请求等长请求，延迟不稳定

Continuous Batching（Orca）：
  请求随时加入/离开批次
  每完成一个 token 的生成，立即调度新请求

  时间轴：
    T1: 请求 A (prompt=100) 开始预填充
    T2: 请求 B (prompt=50) 加入，一起预填充
    T3: 请求 A 开始解码，请求 B 继续预填充
    T4: 请求 C 加入，请求 A/B/C 同时解码（不同位置）
```

**效果**：
- GPU 利用率提升 5~10 倍
- 吞吐量提升 10~20 倍
- 延迟更稳定（短请求不被长请求阻塞）

#### 7.2.2 分块预填充（Chunked Prefill）

```
长提示词切分为多个块（Chunk），与解码请求交错执行：

  时间片 1: 处理 Chunk 1 of Prompt A
  时间片 2: 处理 Token 1 of Request B
  时间片 3: 处理 Chunk 2 of Prompt A
  时间片 4: 处理 Token 2 of Request B

效果：
  - 避免长预填充阻塞所有解码请求
  - 平衡计算密集和带宽密集任务
  - 整体延迟更平滑
```

### 7.3 请求调度策略

#### 7.3.1 优先级调度

```
优先级队列：
  P0（最高）：交互式对话（用户等待）
  P1：后台任务（文档分析）
  P2：批处理任务（离线生成）

抢占策略：
  - 高优先级请求可抢占低优先级的 KV Cache
  - 被抢占的请求状态保存到 CPU/磁盘
  - 高优先级完成后恢复低优先级请求
```

#### 7.3.2 负载均衡

```
多实例部署：
  - 请求按 KV Cache 大小、预计生成长度分发
  - 避免单个实例过载
  - 考虑实例间的前缀缓存共享
```

---

## 八、硬件协同优化

### 8.1 GPU 架构适配

#### 8.1.1 Ampere / Hopper 特性利用

```
Tensor Core：
  - FP16/BF16 矩阵乘法加速
  - Hopper 支持 FP8，吞吐量翻倍

Async Copy：
  - Hopper 支持异步内存拷贝
  - 计算与数据传输重叠

TMA（Tensor Memory Accelerator）：
  - 硬件级矩阵转置和重排
  - FlashAttention-3 充分利用

Multi-Stream：
  - 并行执行多个 CUDA Stream
  - 预填充和解码流并行
```

#### 8.1.2 HBM 带宽最大化

```
内存访问模式优化：
  - 连续访问（Coalesced Access）
  - 避免 bank conflict
  - 数据预取（Prefetching）
```

### 8.2 多节点优化

#### 8.2.1 InfiniBand / NVLink

```
NVLink（节点内）：
  - 带宽：900 GB/s（NVLink 4）
  - 延迟：微秒级
  - 适合 TP、EP

InfiniBand（节点间）：
  - 带宽：400 Gb/s（NDR）
  - 延迟：1~2 微秒
  - 适合 PP、DP

RDMA：
  - 直接内存访问，绕过 CPU
  - 零拷贝传输
```

#### 8.2.2 网络拓扑感知

```
Dragonfly+ 拓扑：
  - 节点分组为交换机
  - 组间全连接
  - 路由算法优化减少拥塞
```

---

## 九、技术选型与组合建议

### 9.1 按场景选型

**场景 1：单卡/单机部署（7B~13B 模型）**

```
推荐组合：
- 量化：AWQ INT4 或 FP8（H100）
- KV Cache：GQA + FP8 KV
- 注意力：FlashAttention-2/3
- 解码：Greedy / Sampling（无投机）
- 框架：vLLM / TensorRT-LLM

预期：7B 模型单卡 A100 可达 100+ tokens/s
```

**场景 2：多卡服务部署（70B 模型）**

```
推荐组合：
- 并行：TP=8（单节点）
- 量化：FP8 权重量化 + FP8 KV Cache
- KV Cache：PagedAttention + Prefix Caching
- 批处理：Continuous Batching
- 解码：MTP 投机解码（多头版本）
- 框架：vLLM / SGLang

预期：8×A100 可达 500+ tokens/s 总吞吐
```

**场景 3：超大规模 MoE 部署（1T+ 参数）**

```
推荐组合：
- 并行：TP=8 × PP=16 × EP=128
- 量化：FP4 MoE 专家（QAT）+ FP8 稠密层
- KV Cache：MLA 压缩 + CSA/HCA 稀疏注意力
- 通信：通信-计算重叠 + 融合内核
- 解码：MTP + 投机解码
- 框架：Megatron-LM / DeepSeek 自定义框架

预期：16K GPU 集群支持 1M 上下文，数万并发
```

**场景 4：超长文本处理（1M+ tokens）**

```
推荐组合：
- 模型：DeepSeek V4 / Kimi K2.6（原生长上下文）
- 注意力：CSA + HCA 混合稀疏注意力
- KV Cache：MLA 压缩 + 前缀缓存
- 系统：IMPRESS 多级缓存
- 智能体：RLM 递归处理（>10M tokens）

预期：1M 上下文 TTFT < 10s，检索准确率 97%+
```

### 9.2 优化效果量化对比

| 优化技术 | 适用阶段 | 吞吐提升 | 延迟降低 | 显存节省 |
|----------|----------|----------|----------|----------|
| GQA | Decode | - | - | 4~8× |
| KV 量化 (FP8) | Decode | - | - | 2× |
| PagedAttention | Decode | 2~4× | - | 2× |
| Prefix Caching | Prefill | - | 5~10× | - |
| FlashAttention | Prefill | 2~4× | - | - |
| 投机解码 | Decode | 2~3× | 2~3× | - |
| MTP | Decode | 3~5× | 3~5× | - |
| Continuous Batching | Both | 10~20× | - | - |
| 权重量化 (INT4) | Both | 2~3× | - | 4× |
| 通信-计算重叠 | Both | 1.3~1.5× | 30~50% | - |

---

## 十、总结

大模型推理优化是一项跨算法、系统、硬件的综合性工程。核心优化维度包括：

1. **KV Cache 优化**：通过 GQA、量化、稀疏化、分页、前缀缓存等手段，将显存占用降低 4~20 倍
2. **投机解码**：用草稿模型或 MTP 将串行生成变为并行验证，解码速度提升 2~5 倍
3. **并行加速**：TP/PP/EP/SP 的组合策略，将模型分布到数千 GPU，突破单卡极限
4. **量化压缩**：INT4/FP4/FP8 权重量化，在精度损失可控前提下实现 2~4 倍加速
5. **通信优化**：通信-计算重叠、融合内核、拓扑感知路由，将分布式开销降至最低
6. **编译调度**：算子融合、Continuous Batching、动态批处理，榨取最后一滴硬件性能
7. **硬件协同**：充分利用 Tensor Core、Async Copy、TMA 等架构特性

这些技术的组合应用，使得当前大模型推理已从"实验室玩具"进化为"生产级基础设施"——7B 模型可在单卡上实时响应，万亿参数 MoE 模型可在万卡集群上支持百万级上下文和数万并发。随着稀疏注意力、动态推理、端侧部署等方向的持续突破，推理效率仍将保持快速提升。
