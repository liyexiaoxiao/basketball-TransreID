# 篮球持球人重识别项目 PPT 大纲 v2.0

> **数据来源**：严格基于以下开发者文档
> - `development_logs_and_summary.docx` — 开发日志（21周，4阶段，4轮调优，6条技术教训）
> - `technical_scheme.docx` — 技术方案（完整架构、数据流、参数表）
> - `budget.md` — 成本估算（101,750 RMB）

---

## 📊 PPT 结构：24页

---

### 第1页 | 封面

| 内容 | 来源 |
|------|------|
| Basketball Ball-Handler Re-Identification System | development_logs |
| 基于TransReID的篮球持球人身份重识别算法开发 | technical_scheme P68 |
| Project: Basketball Ball-Handler Re-Identification Algorithm in Complex Motion Scenes | technical_scheme P69 |
| Team: Ye Li · Chengyou Long · Siyu Lei | development_logs P34 |
| Duration: January 21 – June 12, 2026 (21 weeks) | development_logs P33 |

---

### 第2页 | 目录 CONTENTS

```
01  Project Overview              项目概览与背景
02  System Architecture            系统架构与数据流
03  Core Module Design             核心模块设计
04  Loss Function Design           损失函数设计
05  Training Strategy              训练策略与超参调优
06  Data Augmentation Pipeline     篮球专项数据增强
07  Inference & Evaluation         推理与评估
08  GUI Application                桌面应用
09  Performance & Lessons Learned  结果与经验教训
10  Project Management             项目管理
```

---

### 第3页 | 分隔页 — PART 01: Project Overview

---

### 第4页 | 项目背景与三大挑战

> 来源：technical_scheme §1.1, Table 0

**业务目标：**
在不依赖连续轨迹追踪的前提下，仅凭外观特征跨时刻、跨摄像头检索特定持球人。

**为什么选 TransReID（vs CNN）：**
| 优势 | 说明 |
|------|------|
| 自注意力机制 | 天然处理长距离依赖（关联跨遮挡的身体部位） |
| 多头注意力 | 保留细粒度局部特征，优于CNN的层次池化 |
| 位置嵌入 | 提供空间结构感知，无需显式部件模型 |

**三大挑战（Table 0）：**

| 挑战 | 描述 |
|------|------|
| **相似球衣区分** | 队友穿相同球衣，需提取号码、体态、护具、发型、鞋色等细粒度特征 |
| **严重遮挡与姿态多变** | 防守压力下的多人遮挡；运球/投篮导致极端身体形变 |
| **环境干扰** | 室内木地板反光、室外复杂背景、夜间灯光、高速运动模糊 |

**性能目标（Table 1）：**

| 指标 | 基线 | 赛题要求 |
|------|------|---------|
| mAP | 91.1% | ≥ 91.5% |
| Rank-1 | 93.6% | ≥ 94% |

---

### 第5页 | 系统架构总览

> 来源：technical_scheme §2.1-2.3

**训练管线：**
```
Data Loader → Model Builder → Loss Function → Optimization & Scheduling
    ↓              ↓               ↓                    ↓
BallShow     ViT-Base/16     ID Loss (Softmax)    SGD (momentum=0.9)
PK Sampler   + JPM (5分支)   Triplet (Hard Mine)  Cosine LR + Warmup
Augmentation + SIE (Camera)  Center (Global)      AMP + EMA (0.9995)
```

**推理管线：**
```
Gallery Feature Extract → Query Feature Extract → Feature Norm + TTA → Distance Matrix + Re-Ranking
```

**完整数据流（训练）：**
```
BallShow Dataset
→ RandomIdentitySampler (16 IDs × 4 instances = 64 batch)
→ 12步增强管线 (含篮球专项)
→ ViT-Base/16 (overlapping patch, stride=12) → 210 patches
→ SIE Embedding (camera-aware, λ=3.0)
→ 12 Transformer Blocks (drop_path=0.1)
→ JPM (1 global + 4 local, shuffle+shift+divide)
→ BNNeck (BatchNorm1d, 768-dim)
→ Loss: ID(LabelSmooth ε=0.1) + Triplet(SoftMargin HardMine) + 0.001×Center(Global)
→ AMP Backward (GradScaler)
→ EMA Update (decay=0.9995)
```

**完整数据流（推理）：**
```
Query + Gallery Images
→ Val Transforms (Resize→[256,128], ToTensor, Normalize)
→ TTA: Forward(original) + Forward(flipped) → Average → L2-Normalize
→ Multi-Branch Concat: 5×768 = 3840-dim (各分支独立L2归一化后拼接)
→ Euclidean Distance Matrix
→ k-Reciprocal Re-Ranking (k1=20, k2=6, λ=0.3)
→ Ranked Results → mAP / CMC
```

---

### 第6页 | 分隔页 — PART 02: Core Module Design

---

### 第7页 | ViT Backbone + Overlapping Patches

> 来源：technical_scheme §3.1, Table 2, Table 13

**ViT-Base/16 TransReID 结构：**
```
TransReID(
  patch_embed: PatchEmbed_overlap  -- Conv2d stride=12 < kernel=16
  cls_token:   learnable [1, 1, 768]
  pos_embed:   learnable [1, N+1, 768]
  sie_embed:   learnable [cameras, 1, 768]   -- SIE可选
  pos_drop:    Dropout(p=drop_rate)
  blocks:      12× Transformer Block (Norm→Attn→DropPath→Norm→MLP→DropPath)
  norm:        LayerNorm(768)
  fc:          Linear(768, num_classes)
)
```

**Overlapping Patch Embedding 关键计算：**
- 输入 256×128, kernel=16, stride=12
- num_y = (256-16)//12 + 1 = **21**
- num_x = (128-16)//12 + 1 = **10**
- Total: **21×10 = 210 patches**（vs stride=16的 16×8=128，**+64%**）
- 效果：50%重叠 → 更细空间粒度 → 捕捉球衣号码、护具等小特征

**架构变体（Table 2）：**

| 变体 | Layers | Heads | Embed Dim | Params | 用途 |
|------|--------|-------|-----------|--------|------|
| vit_base | 12 | 12 | 768 | ~86M | 主力模型 ✅ |
| vit_small | 8 | 8 | 768 | ~48M | 轻量 |
| deit_small | 12 | 6 | 384 | ~22M | 快速迭代 |

---

### 第8页 | SIE + JPM 核心模块

> 来源：technical_scheme §3.2-3.4

**SIE (Side Information Embedding)：**
```
Token = patch_embed(x) + pos_embed + λ × SIE_embed[camera_id]
```
- λ = SIE_COE = 3.0（强侧信息贡献）
- SIE_CAMERA=True, SIE_VIEW=False（篮球场景视角多样，关闭视角嵌入）
- 目的：解耦身份特征与摄像头相关伪影

**JPM (Jigsaw Patch Module) — 三步流程：**

| 步骤 | 操作 | 参数 |
|------|------|------|
| **Step 1: Shuffle + Shift** | circular_shift(offset=5) → group_shuffle(groups=2) | SHIFT_NUM=5, SHUFFLE_GROUP=2 |
| **Step 2: Divide** | 210 patches均分为4组 (52+52+52+54) → 每组+CLS → 共享Block+LN | DEVIDE_LENGTH=4 |
| **Step 3: Global** | 全部token序列 → Block+LN → global_feat | — |

**输出：**
- 训练时：5个分类分数 + 5个特征向量
- 推理时：concat([norm(global), norm(lf1-4)]) = **5×768 = 3840-dim**
- 关键设计：各分支独立L2归一化后拼接（非拼接后归一化），保留分支互补性

**JPM参数（Table 3）：**

| 参数 | 值 | 说明 |
|------|-----|------|
| SHIFT_NUM | 5 | Patch循环移位偏移 |
| SHUFFLE_GROUP | 2 | Patch洗牌组数 |
| DEVIDE_LENGTH | 4 | 局部分支数 |
| RE_ARRANGE | True | 启用shuffle+shift |

**设计理由（来自文档）：**
- 4个分支提供不同身体部位的互补局部特征
- 共享Block+LN确保特征一致性并减少参数
- 独立L2归一化保持分支互补性

**BNNeck：**
- global_feat(768-dim, raw) → BatchNorm1d → feat(768-dim, normalized)
- BN前 → Triplet Loss / BN后 → ID Loss
- 推理时使用 BN后特征 (NECK_FEAT='after')

**Model Factory（3种模型类）：**
| 模型类 | 结构 | 用途 |
|--------|------|------|
| Backbone | ResNet50 + BNNeck | CNN基线 |
| build_transformer | ViT/DeiT + SIE + BNNeck | 无JPM |
| build_transformer_local | ViT/DeiT + SIE + JPM(5分支) + BNNeck(5个独立颈层) | 主力 ✅ |

---

### 第9页 | 损失函数设计

> 来源：technical_scheme §5.1-5.5, development_logs §3.2

**总体公式：**
```
L_total = λ_id × L_id + λ_triplet × L_triplet + λ_center × L_center
       = 1.0 × L_id + 1.0 × L_triplet + 0.001 × L_center
```

**ID Loss — CrossEntropy with Label Smoothing（实际使用）：**
```
L_id = -(1-ε) × log(p_y) - ε/K × Σ log(p_k)    (ε=0.1)
```
- 应用于全部5个分支（JPM模式）
- 多分支加权：L_id = 0.5×L_id_global + 0.5×avg(L_id_lf1~4)

**Triplet Loss — Soft Margin + Hard Mining：**
```
L_triplet = softplus(d_an - d_ap)    (NO_MARGIN=True)
```
- Batch-hard mining：每个anchor选最难正样本 + 最难负样本
- 作用于BNNeck之前的特征空间
- 多分支加权：与ID Loss相同（50% global + 50% local平均）

**Center Loss：**
```
L_center = 1/2 × Σ ||f_i - c_{y_i}||²
```
- 仅作用于global分支特征（关键发现：JPM分支特征来自不同身体部位，语义分布不同，不应拉向同一中心）
- 独立SGD优化器（lr=0.5），梯度缩放1/CENTER_LOSS_WEIGHT
- 最终：weight=0.001, global only

**完整Loss流（JPM模式）：**
```
scores = [cls_global, cls_lf1, cls_lf2, cls_lf3, cls_lf4]
feats  = [feat_global, feat_lf1, feat_lf2, feat_lf3, feat_lf4]

ID_LOSS     = 0.5×CE(scores[0]) + 0.5×mean(CE(scores[1:5]))
TRI_LOSS    = 0.5×Triplet(feats[0]) + 0.5×mean(Triplet(feats[1:5]))
CENTER_LOSS = CenterLoss(feats[0], target)

total_loss = ID_LOSS + TRI_LOSS + 0.001×CENTER_LOSS
```

**损失函数演化（v1→v4）：**
| 版本 | 公式 |
|------|------|
| v1.0 (Jan) | CE + SoftMargin Triplet |
| v3.0 (May) | LabelSmooth(ε=0.1) or ArcFace(s=30,m=0.5) + Triplet + Gradient Clip |
| v4.0 (Jun) | LabelSmooth + Triplet + **0.001×Center(global only)** ← 最终 |

**可选ID Loss类型（Table 7）：代码均已实现：**
| 类型 | Margin | Scale | 状态 |
|------|--------|-------|------|
| Softmax + LabelSmooth | — | — | ✅ 使用中 |
| ArcFace | m=0.5 | s=30 | ⚠️ 可用未训练 |
| CosFace | m=0.3 | s=30 | ⚠️ 可用未训练 |
| AMSoftmax | m=0.3 | s=30 | ⚠️ 可用未训练 |
| CircleLoss | m=0.25 | s=256 | ⚠️ 可用未训练 |

---

### 第10页 | 训练策略与超参调优

> 来源：technical_scheme §6, development_logs §4

**优化配置（Table 9）：**

| 参数 | 值 | 说明 |
|------|-----|------|
| 模型优化器 | SGD, momentum=0.9 | — |
| Center优化器 | SGD, lr=0.5 | 独立于模型优化器 |
| Base LR | 0.008 | Cosine衰减 |
| Weight Decay | 1e-4 | 权重和偏置统一 |
| Bias LR Factor | 2x | 偏置学习率加倍 |
| Batch Size | 64 | 16 IDs × 4 instances (PK采样) |
| Max Epochs | 130 | 从120延长以收敛Center Loss |
| AMP | GradScaler (float16) | 降低显存, 加速~1.5x |

**LR Schedule — Cosine Annealing + Linear Warmup：**
```
Epochs 1-5 (Warmup):
  lr = 8e-5 + (epoch/5) × (0.008 - 8e-5)

Epochs 6-130 (Cosine):
  lr = 1.6e-5 + 0.5×(0.008-1.6e-5)×(1 + cos(π×epoch/130))
```
- 避免阶梯式衰减的性能突变

**EMA (Exponential Moving Average)：**
```
ema_param = 0.9995 × ema_param + (1-0.9995) × model_param
```
- 每次迭代后更新
- 评估时使用EMA模型
- 从0.9998调低 → 适应Center Loss带来的更频繁参数更新

**训练循环（do_train）：**
```
for epoch = 1 to 130:
  1. scheduler.step(epoch)
  2. for each batch:
     a. AMP autocast forward → loss
     b. scaler.scale(loss).backward()
     c. scaler.step(optimizer)  +  scaler.step(optimizer_center)
     d. scaler.update()
     e. ema_model.update(model)
  3. 每40 epoch保存: transformer_{epoch}.pth + transformer_ema_{epoch}.pth
  4. 每30 epoch评估: EMA模型验证 → mAP/R1/R5/R10
```

**Checkpoint策略：**
- 保存点：epoch 40, 80, 120, 130
- 每个点保存两份：标准模型 + EMA模型
- 最终推理使用 epoch 130 EMA 模型

**LR Schedule调优（development_logs Table 0）：**
| 参数 | 调优前 | 调优后 | 结论 |
|------|--------|--------|------|
| BASE_LR | 3e-4 | 0.008 | ViT+SGD需要显著更高的LR |
| LR Schedule | MultiStepLR | CosineLR+warmup | 平滑衰减改善收敛与最终mAP |

---

### 第11页 | 4轮Center Loss超参调优

> 来源：development_logs §4.2 (Table 1)

**Center Loss 调优全过程（June 12, 2026，一天内4轮迭代）：**

| 轮次 | Weight | 作用范围 | EMA Decay | Epochs | 结论 |
|------|--------|---------|-----------|--------|------|
| 初始基线 | 0.0005 | Global only | 0.9998 | 120 | 初始集成 |
| **Tune 1** | 0.0005 | **全部5分支** | 0.9995 | 120 | ❌ 多分支Center Loss降低性能 |
| **Tune 2** | **0.001** | Global only | 0.9995 | 120 | ✅ 加倍权重+全局有效 |
| **Tune 3** | 0.0015 | Global only | 0.9995 | 120 | ❌ 过度约束，降低特征区分性 |
| **最终** | **0.001** | **Global only** | **0.9995** | **130** | ✅ 最佳平衡 |

**关键洞察（来自文档原文）：**
> "Center loss on JPM branches is counterproductive because each local branch learns features from semantically different body regions. Pulling features from different body parts toward the same class center forces incompatible representations together."

**EMA Decay 调优（Table 2）：**
| Decay值 | 上下文 | 结论 |
|---------|--------|------|
| 0.9998 | 无Center Loss | 标准120 epoch训练适用 |
| **0.9995** | **含Center Loss** | **模型参数变化更快，需更快EMA追踪** |

**增强概率调优（Table 3）：**
| 参数 | 初始 | 调优后 | 理由 |
|------|------|--------|------|
| CJ_PROB | 0.5 | **0.2** | 过多颜色变化减慢服务器端收敛 |
| MB_PROB | 0.5 | **0.2** | 过多运动模糊遮盖区分队友所需的号码细节 |

**Re-Ranking 参数（Table 4）：**
| 参数 | 值 | 来源 |
|------|-----|------|
| k1 | 20 | CVPR 2017 原论文 |
| k2 | 6 | CVPR 2017 原论文 |
| λ | 0.3 | 30% Jaccard + 70% Euclidean平衡 |

---

### 第12页 | 数据增强流水线

> 来源：technical_scheme §4.3 (Table 5, Table 6)

**12步完整增强序列：**

| # | 增强 | 概率 | 篮球场景用途 |
|---|------|------|-------------|
| 1 | Resize [256, 128] | 1.0 | 输入标准化 |
| 2 | RandomHorizontalFlip | 0.5 | 球场左右对称 |
| 3 | Pad(10)+RandomCrop | 1.0 | 平移不变性+尺度增强 |
| 4 | ColorJitter(b=0.3,c=0.3,s=0.3) | 0.2 | 不同场馆光照变化 |
| 5 | RandomDirectionalLighting | 0.05 | 球馆聚光灯/室外太阳角度 |
| 6 | RandomColorTemperature | 0.05 | 室内暖光↔室外冷日光 |
| 7 | RandomISONoise | 0.05 | 昏暗室内高ISO传感器噪声 |
| 8 | RandomMotionBlur(kernel 5-15) | 0.1 | 快攻/上篮/扣篮中高速移动 |
| 9 | GaussianBlur(k=5,σ=0.1-2.0) | 0.025 | 相机自动对焦导致的散焦模糊 |
| 10 | RandomBackgroundBlur | 0.05 | 浅景深(bokeh)仿真 |
| 11 | ToTensor+Normalize([0.5]*3) | 1.0 | ViT标准归一化 |
| 12 | RandomErasing(max_count=3) | 0.5 | 篮球场景多人遮挡(多区域擦除) |

**5项篮球专项增强算法详情（Table 6）：**

| 增强 | 算法 | 篮球场景理由 |
|------|------|-------------|
| **MotionBlur** | cv2.line 抗锯齿运动核，随机角度0-360° | 快攻/上篮/扣篮中高速移动球员 |
| **DirectionalLighting** | 线性渐变强度遮罩，方向随机，强度0.4-0.8 | 球馆聚光灯、室外定向阳光 |
| **ColorTemperature** | R-B通道偏移(±30) | 室内钠灯/卤素灯暖光↔室外冷日光 |
| **ISONoise** | 高斯噪声 σ=10-25 | 昏暗室内体育场馆高ISO传感器噪声 |
| **BackgroundBlur** | 椭圆中心遮罩(清晰)+边缘高斯模糊(kernel 15-25) | 浅景深隔离球员与球场背景 |

**TTA（测试时增强）：**
```
feat_final = L2_normalize((feat_original + feat_flipped) / 2)
```
- 训练评估和最终推理均使用
- +0.3-0.5% Rank-1，计算成本极低（2x forward）

---

### 第13页 | 推理与评估

> 来源：technical_scheme §7

**特征提取管线：**
```
Image → Resize[256,128] → ToTensor → Normalize([0.5]³) → Model Forward → L2-Norm
  + TTA: feat = normalize(feat_original + feat_flipped) / 2
  + JPM: concat([norm(global), norm(lf1-4)]) = 3840-dim
  (各分支独立归一化 → 保留分支互补性)
```

**距离计算：**
- **Euclidean Distance（主力）**：dist(q,g) = ‖q‖² + ‖g‖² - 2·q·gᵀ
  - L2归一化后等价于 2-2cos(q,g)
- Cosine Distance（备选）

**k-Reciprocal Re-Ranking（CVPR 2017）：**
```
1. 计算初始Euclidean距离矩阵
2. 对每个样本找 top-k1 (k1=20) 近邻
3. 计算互惠近邻 + 扩展 (k2=6)
4. 计算Jaccard距离
5. 最终距离 = 0.3×Jaccard + 0.7×Original (λ=0.3)
```
- 效果：+1~3% mAP
- 篮球场景价值：抑制视觉相似但不在互惠近邻集中的同球衣队友误匹配

**评估指标（Table 10）：**
| 指标 | 公式 | 说明 |
|------|------|------|
| CMC Rank-k | Σ 1{correct_in_top_k} / N_query | 正确匹配出现在前k的概率 |
| mAP | Σ AP_i / N_query | 综合查准率与查全率 |

---

### 第14页 | PyQt5 GUI 桌面应用

> 来源：technical_scheme §8

**架构：**
```
run_gui.py → gui/app.py (QApplication) → MainWindow
  ├── Left Sidebar: Model Config + Query Panel
  └── Right Panel (QTabWidget):
        ├── Tab 1: Single Image Search
        ├── Tab 2: Dataset Explorer
        └── Tab 3: Batch Evaluation
```

**3个Tab功能（Table 11）：**

| Tab | 功能 | 描述 |
|-----|------|------|
| **Single Image Search** | Query-by-image检索 | 选择查询图→搜索→Top-K结果(相似度%+颜色编码匹配状态) |
| **Dataset Explorer** | 浏览查询集目录 | 缩略图网格+ID搜索过滤；双击即设为查询并自动搜索 |
| **Batch Evaluation** | 全量指标评估 | 运行标准评估协议；实时显示mAP/R1/R5/R10指标卡 |

**4个后台线程：**
| 线程类 | 职责 |
|--------|------|
| ModelLoaderThread | 加载config + dataset + model + weights |
| GalleryFeatureExtractorThread | 预提取所有gallery特征(含TTA) |
| QuerySearchThread | 提取query特征(含TTA)→计算距离→返回Top-K |
| BatchEvaluationThread | 处理所有query+gallery→计算mAP和CMC |

**关键实现细节：**
- 线程并发：pyqtSignal更新UI不阻塞；closeEvent正确终止线程
- Gallery特征缓存：首次提取后缓存在内存 → 后续查询即时检索
- TTA全覆盖：gallery提取和query搜索均含水平翻转TTA
- 预训练权重检测：自动跳过backbone-only文件的分类层加载

---

### 第15页 | 分隔页 — PART 03: Results & Lessons

---

### 第16页 | 最终结果与性能分析

> 来源：development_logs §5, technical_scheme Table 17

**最终结果（June 12, 2026）：**

| 指标 | 实际结果 | 赛题要求 | 状态 |
|------|---------|---------|------|
| **mAP** | **91.8%** | ≥ 91.5% | ✅ 超额完成 |
| **Rank-1** | **94.8%** | ≥ 94% | ✅ 超额完成 |

**6大关键技术成果：**
1. Transformer-based ReID adapted to basketball domain — 解决三大核心挑战
2. JPM + overlapping patches — stride=12+5分支提供细粒度空间+鲁棒部件特征
3. Basketball-domain augmentation suite — 5种物理驱动的篮球场景增强
4. Center loss with systematic 4-iteration tuning — 揭示多分支Center Loss的关键限制
5. Complete production inference pipeline — Re-Ranking+TTA+3840-dim特征
6. Full-featured PyQt5 GUI — 非技术用户可操作的完整桌面应用

**各优化技术贡献估算（Table 17）：**

| 技术 | 估算贡献 | 机制 |
|------|---------|------|
| Overlapping patches (stride=12) | +0.5~1.0% mAP | 更细空间粒度 |
| JPM with shuffle+shift | +1.0~2.0% mAP | 鲁棒部件级特征 |
| SIE camera embedding | +0.3~0.5% mAP | 视角不变性 |
| Center loss (weight=0.001) | +0.5~1.0% mAP | 更紧类内聚类 |
| Basketball-domain augmentations | +0.5~1.5% mAP | 运动模糊/光照/噪声鲁棒 |
| EMA (decay=0.9995) | +0.2~0.5% mAP | 更平滑权重收敛 |
| k-Reciprocal Re-Ranking | +1.0~3.0% mAP | 后处理检索精度提升 |
| TTA (horizontal flip) | +0.3~0.5% mAP | 左右变化鲁棒 |
| Label Smoothing (ε=0.1) | +0.2~0.5% mAP | 对未见ID更好泛化 |
| 130 epochs (vs 120) | +0.2~0.5% mAP | Center Loss更充分收敛 |

---

### 第17页 | 经验教训 (Lessons Learned)

> 来源：development_logs §6.1-6.2

**6条技术教训 (Technical Lessons)：**

| # | 教训 | 详细说明 |
|---|------|---------|
| 1 | **Overlapping patches对细粒度ReID至关重要** | stride=12 vs stride=16：50%重叠提供冗余覆盖，对遮挡鲁棒；对相对较小的bounding box(256×128)中的号码、护具等小特征至关重要 |
| 2 | **Center Loss作用范围必须匹配特征语义** | 应用于语义异质的特征（不同身体部位）会产生反效果；每个局部分支学习不同身体区域的特征，不应被拉向同一中心 |
| 3 | **EMA decay需随新损失项调整** | 添加Center Loss改变了优化动力学；降低decay(0.9998→0.9995)确保EMA追踪更频繁的参数更新 |
| 4 | **领域专项增强需要概率校准** | 过多(p>0.3)减慢收敛；过少(p<0.05)正则化不足；经验最优区间0.1-0.2 |
| 5 | **YACS配置使系统实验成为可能** | 类型化YACS+YAML覆盖使6+架构变体无需代码修改即可实验；每个消融配置独立文件，结果可复现可比 |
| 6 | **TTA以低成本提供持续收益** | 水平翻转TTA仅2x forward开销，持续改善Rank-1 0.3-0.5%；应在所有推理配置中默认开启 |

**4条过程教训 (Process Lessons)：**

| # | 教训 |
|---|------|
| 1 | **Feature branch + PR review**：所有开发在feature分支进行，强制PR review后合并dev，确保代码质量并创建清晰审计轨迹 |
| 2 | **Single-variable systematic tuning**：4轮Center Loss调优每轮仅改变一个参数（范围→权重→权重→epochs），隔离每次变化的效果 |
| 3 | **Ablation-ready configurations**：JPM-only/SIE-only/Base ViT/DeiT等独立配置文件使得每个架构组件的贡献可被隔离验证而无需修改代码 |
| 4 | **Regular evaluation during training**：EVAL_PERIOD=30+EMA评估，提供定期进度反馈且不产生过多开销 |

---

### 第18页 | Future Work

> 来源：development_logs §6.3

**短期（立即改善潜力）：**
- 启用 ArcFace loss（代码已实现，切换到ArcFace s=30,m=0.5 估计 +0.3~0.8% mAP）
- 尝试更高分辨率 (384×128 config 已就绪)
- 延长训练到150-200 epochs

**中期（架构改进）：**
- ViT-Large backbone (307M params, 1024-dim)
- 外部数据预训练 (MSMT17 4,101 IDs, CUHK03)
- 利用时序元数据（帧索引，当前未使用）
- CNN+ViT Ensemble（ResNet50 + ViT 特征互补）

**长期（高级技术）：**
- Knowledge Distillation（ViT-Base → DeiT-Small）
- Online hard example mining with memory bank
- 系统化超参搜索（Grid/Bayesian）
- 跨数据集泛化研究

---

### 第19页 | 分隔页 — PART 04: Project Management

---

### 第20页 | 开发时间线与双周报告

> 来源：development_logs §1 Development Timeline

**完整开发时间线（21周，Jan 21 – Jun 12, 2026）：**

```
2026-01-21  [Init]         Project scaffold, TransReID baseline
     |      Phase 1        克隆官方仓库、搭建环境、ViT-Base+ImageNet预训练、模块化结构
     |                     交付：可运行的训练/测试管线 + YACS配置系统 + ResNet50备选
     |
2026-04-07  [Data]         Data folder setup, .gitignore
2026-04-09  [Augmentation]  5项篮球专项增强实现 + visualize_aug.py
2026-04-15  [Config]       服务器适配：CJ_PROB/MB_PROB降至0.2
2026-04-21  [Dataset]      BallShow dataset parser (regex解析) + 7个YAML配置文件
     |      Phase 2        增强集成到DataLoader、PK采样、训练/验证分割
     |
2026-05-27  [Frontend]     PyQt5 GUI (727行main_window + 377行model_thread + 258行widgets)
2026-05-30  [Inference]    Re-ranking优化 + AQE + Multi-scale特征提取
2026-06-04  [Model]        ArcFace loss option + Re-Ranking + gradient clipping
     |      Phase 3        3-Tab界面、4线程后台、Gallery特征缓存、最佳checkpoint追踪
     |
2026-06-05  [Loss]         Center Loss集成（独立SGD optimizer, lr=0.5）
2026-06-12  [Tuning]       4轮Center Loss超参调优（单日内完成）
     |      Phase 4        
2026-06-12  [Final]        mAP 91.8%, Rank-1 94.8% → 双指标超额完成
```

**4个开发阶段：**
| Phase | 时间 | 核心工作 |
|-------|------|---------|
| Phase 1: Init | Jan 21, 2026 | 项目脚手架、TransReID基线、环境搭建 |
| Phase 2: Data+Aug | Apr 7-21, 2026 | 5项篮球增强、BallShow解析器、7个YAML配置、服务器适配 |
| Phase 3: GUI+Inference | May 27-Jun 4, 2026 | PyQt5应用、Re-Ranking、AQE、Multi-scale、ArcFace选项 |
| Phase 4: Loss+Tuning | Jun 5-12, 2026 | Center Loss集成、4轮调优、双指标达标 |

**⚠️ 你需要补充的内容：**
- [ ] 各阶段的双周报告PPT/文档截图
- [ ] 里程碑评审会议纪要
- [ ] 关键决策记录（为什么1月到4月有3个月间隔？）

---

### 第21页 | 项目当前状态

**已有定量数据：**

| 指标 | 数值 | 来源 |
|------|------|------|
| 项目周期 | 21周 (Jan 21 – Jun 12, 2026) | development_logs P33 |
| 团队人数 | 3人 | P34 |
| 代码文件 | 45个.py源文件 | Appendix B |
| 代码行数 | ~5,000+行 | P315 |
| 配置文件 | 7个YAML | Appendix B |
| 开发阶段 | 4个Phase | §2 |
| 调优迭代 | 4轮 (Center Loss单日) | §4.2 |
| 最终mAP | 91.8% | §5 |
| 最终Rank-1 | 94.8% | §5 |
| 分支策略 | Feature branch → PR review → dev | P36 |
| Git | 有GitHub remote | Table 7 |

**⚠️ 需要你补充：**
- [ ] 燃烧图/燃尽图（PV/EV/AC数据）
- [ ] CPI/SPI 绩效指数
- [ ] 挣值分析表
- [ ] 各成员实际工时统计

---

### 第22页 | GitHub提交历史

**从文档可知：**
- Repository: basketball-TransreID（有GitHub remote）
- Branch Strategy: Feature branches → dev via Pull Request review
- 已知分支: master, dev, dev-lee, feature/reranking, feature/tune, occlusion
- 代码质量工具: ruff (Python linter)

**⚠️ 需要你提供：**
- [ ] `git log --oneline --all --graph` 截图
- [ ] GitHub Insights → Contributors / Code Frequency
- [ ] 各分支Commit数统计
- [ ] PR列表截图

---

### 第23页 | 定量统计

**已有数据（可直接使用）：**

| 统计项 | 数值 |
|--------|------|
| Python源文件 | 45个 |
| 代码行数 | ~5,000+ |
| YAML配置 | 7个 |
| 核心模块 | 9个 (config/datasets/model/loss/solver/processor/utils/gui/doc) |
| 开发阶段 | 4个Phase |
| 模型架构版本 | 4代 (v1→v4) |
| 损失函数版本 | 3代 (v1→v3→v4) |
| 增强版本 | 2代 (v1标准→v2篮球专项→v2.1概率平衡) |
| 基础设施版本 | 3代 (CLI→Multi-config→GUI) |
| 调优迭代 | 4轮 (单日) |
| 调优参数数 | 5类 (LR/Center Loss/EMA/增强概率/Re-Ranking) |

**⚠️ 需要你补充：总工作时间（人时）、WBS任务数、各成员贡献统计**

---

### 第24页 | 规划与实际对比

**已知对比：**

| 维度 | 计划 (budget.md) | 实际 (development_logs) |
|------|-----------------|------------------------|
| 项目周期 | 11周 | **21周** |
| 开发模型 | 6个WBS阶段 | **4个Phase** |
| 调优 | 多轮超参搜索+消融实验 | **4轮Center Loss调优（消融未执行）** |
| ArcFace | 预算中考虑为备选方案 | **代码已实现，未训练（列为Future Work）** |
| 团队 | 3人 (Li/Long/Lei) | 3人 ✅ |

**⚠️ 需要你补充：WBS甘特图对比、SV偏差分析、偏差原因（为什么21周 vs 11周？1月-4月gap？）**

---

### 第25页 | 风险缓解方法

**从budget.md识别的风险 + 实际缓解记录：**

| 风险 | 预算缓解 | 实际结果 |
|------|---------|---------|
| mAP不达标 (<91.5%) | 应急储备9,250 RMB | ✅ 4轮调优后mAP 91.8%达标 |
| 推理延迟超标 (>40ms) | TTA可选关闭、缓存 | ✅ Gallery特征缓存+GPU推理 |
| 数据集质量 | 数据清洗2,000 RMB | ✅ BallShow parser+junk过滤 |
| GPU资源不足 | AutoDL弹性扩容4,500 RMB | ✅ RTX 4090云服务器 |

**实际遇到并解决的4个问题：**
| 问题 | 缓解措施 |
|------|---------|
| 服务器训练收敛过慢 | CJ_PROB/MB_PROB 0.5→0.2 |
| Center Loss多分支降低性能 | 4轮迭代 → global-only |
| EMA不适应Center Loss | decay 0.9998→0.9995 |
| 增强概率过高遮盖特征 | 各专项增强精细调至0.025-0.2 |

**⚠️ 需要你补充：完整风险登记册、风险燃尽图、应急储备实际使用金额**

---

### 第26页 | 项目规模估计

**已有数据：**
| 指标 | 数值 |
|------|------|
| 源文件 | 45个.py + 7个.yml + 3个.md |
| 代码行数 | ~5,000+ Python |
| 预训练模型 | 1.6 GB ViT-Base |
| 依赖 | 6个核心包 |
| 架构参数 | ~86M (ViT-Base) |
| 推理特征维度 | 3840-dim (5×768) |

**⚠️ 需要你补充：FPA功能点分析、COCOMO估算、预估vs实际规模对比**

---

### 第27页 | 项目成本估算

> 来源：budget.md

| 类别 | 金额 (RMB) | 占比 |
|------|-----------|------|
| 人力 (3人×11周×5天×500/天=165人天) | 82,500 | 81.08% |
| 算力 (4090云服务器) | 4,500 | 4.42% |
| 数据与标注 | 2,000 | 1.97% |
| 间接与交付物 | 3,500 | 3.44% |
| 应急储备 (10%) | 9,250 | 9.09% |
| **总计** | **101,750** | **100%** |

**算力明细：**
- 密集训练 (W2-W7): 2台×6周×7天×16h×2.5 = 3,360 RMB
- 调试期 (W1+W8-W11): 1台×5周×7天×8h×2.5 = 700 RMB
- 存储+传输: 440 RMB

**⚠️ 需要你补充：实际支出对比、成本偏差分析**

---

### 第28页 | 经济评估

**⚠️ 全部需要你补充：** ROI分析、CBA成本效益分析、NPV/IRR、投资回收期、商业价值评估

---

### 第29页 | Thank You / Q&A

---

## 📋 与初版大纲的关键差异

| 修正项 | 原大纲 | 更新后 |
|--------|--------|--------|
| 输入尺寸 | [384, 128] | **[256, 128]** |
| 项目周期 | 未明确 | **21周 (Jan 21 – Jun 12, 2026)** |
| 增强概率 | 笼统CJ_PROB=0.2 | **精确到每个增强：MotionBlur=0.1, Lighting=0.05, etc.** |
| 补丁数 | 32×10=320 | **21×10=210** (来自technical_scheme准确计算) |
| LR Schedule | 仅提Cosine | **详细warmup公式+具体数值(8e-5→0.008)** |
| 增强序列 | 10项 | **12步完整序列(含Resize和Normalize)** |
| Loss公式 | 简略 | **完整JPM分支加权公式** |
| 调优记录 | 概述 | **4轮迭代详细表+每条的技术理由** |
| Lessons Learned | 无 | **6条技术+4条过程教训** |
| Future Work | 笼统 | **短/中/长期分类+每项估计增益** |
| 推理管线 | 简略 | **完整流程+3840-dim拼接理由** |
| BNNeck | 简单提及 | **BN前后用途+推理配置选择** |
