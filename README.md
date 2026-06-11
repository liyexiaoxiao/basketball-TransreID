# 🏀 篮球持球人身份重识别（Ball-Handler Re-Identification）

基于 **TransReID**（Transformer-based Object Re-Identification）的篮球场景持球人重识别算法，针对复杂运动场景中的球衣相似、遮挡、运动模糊等挑战进行专项优化。

## 目录

- [🏀 篮球持球人身份重识别（Ball-Handler Re-Identification）](#-篮球持球人身份重识别ball-handler-re-identification)
  - [目录](#目录)
  - [赛题背景](#赛题背景)
    - [核心组件](#核心组件)
    - [损失函数](#损失函数)
  - [项目结构](#项目结构)
  - [环境搭建](#环境搭建)
    - [依赖](#依赖)
    - [安装](#安装)
  - [数据准备](#数据准备)
  - [预训练权重](#预训练权重)
  - [使用流程](#使用流程)
    - [完整工作流总览](#完整工作流总览)
    - [Step 1: 训练模型](#step-1-训练模型)
    - [Step 2: 迁移旧 Checkpoint（可选）](#step-2-迁移旧-checkpoint可选)
    - [Step 3: 测试评估](#step-3-测试评估)
    - [Step 4: 多模型 Ensemble 测试（可选）](#step-4-多模型-ensemble-测试可选)
    - [Step 5: GUI 可视化推理（可选）](#step-5-gui-可视化推理可选)
  - [配置文件说明](#配置文件说明)
  - [关键优化点](#关键优化点)
  - [实验结果](#实验结果)
  - [后续优化路线](#后续优化路线)
  - [引用](#引用)

---

## 赛题背景

在真实的"球秀"业务场景中，需要跨时刻检索特定持球人。相较于传统行人重识别，篮球场景面临三大核心挑战：

| 挑战 | 描述 |
|------|------|
| **相似球衣区分** | 队友穿着完全相同的球衣，需依赖号码（常被遮挡）、体态、护具、发型、鞋色等细粒度特征 |
| **严重遮挡与姿态多变** | 多人包夹重叠、运球/上篮/投篮等剧烈形变，与常规站立姿态差异巨大 |
| **环境干扰** | 室内木地板强反光、室外复杂背景、夜间灯光、运动模糊丢失纹理细节 |

在不依赖连续轨迹追踪的前提下，仅凭外观特征在跨时刻图像库中精准找回目标球员。

---

### 核心组件

| 组件 | 说明 |
|------|------|
| **Backbone** | ViT-Base/16，12层 Transformer，768维特征 |
| **Overlapping Patches** | stride=12×12 的滑动窗口 patch embedding，保留更细粒度的局部信息 |
| **SIE (Side Information Embedding)** | 将摄像头编号编码为可学习的 embedding，注入到 Transformer 输入中 |
| **JPM (Jigsaw Patch Module)** | 将最后一层的 patch tokens 均匀切分为4组，每组与 CLS token 一起送入共享 Block 提取局部特征 |
| **ArcFace (ID Loss)** | 加性角度边际损失，5个分支各有独立 ArcFace 分类器，增强类内紧凑性和类间分离性 |
| **Triplet + Center Loss** | 度量学习层面：Triplet 挖掘难样本，Center 拉近同类特征到类别中心 |
| **BNNeck** | BatchNorm 颈层，缓解 ID loss 和 Triplet loss 在特征空间的优化冲突 |
| **EMA** | 指数移动平均，在测试时使用平滑后的模型权重（decay=0.9998） |
| **TTA** | 测试时水平翻转增强，原始+翻转特征平均 |
| **Re-Ranking** | k-reciprocal 重排序，利用查询-库图结构精排检索结果 |

### 损失函数

```
Total Loss = 1.0 × ID_Loss(ArcFace) + 1.0 × Triplet_Loss + 0.0005 × Center_Loss

ID_Loss: 5个分支各计算 ArcFace，50%权重给全局分支，50%给4个局部分支均值
Triplet_Loss: Soft Margin Triplet，5个分支各计算，同权重分配
Center_Loss: 仅在全局分支上计算
```

---

## 项目结构

```
TransReID/
├── config/                        # 配置系统
│   ├── defaults.py                # 所有默认配置项定义
│   └── __init__.py
├── configs/
│   └── BallShow/                  # 篮球数据集配置文件
│       ├── vit_transreid_stride.yml    # ★ 推荐主配置 (JPM+ArcFace+SIE+Stride)
│       ├── vit_transreid_stride_384.yml
│       ├── vit_transreid.yml
│       ├── vit_transreid_384.yml
│       ├── vit_jpm.yml
│       ├── vit_sie.yml
│       ├── vit_base.yml
│       └── deit_transreid_stride.yml
├── data/
│   └── BallShow/                  # 数据集目录
│       ├── bounding_box_train/    # 训练集
│       ├── bounding_box_test/     # 测试集（gallery）
│       └── query/                 # 查询集
├── datasets/                      # 数据加载
│   ├── ballshow.py                # BallShow 数据集解析
│   ├── make_dataloader.py         # DataLoader 构建
│   ├── sampler.py                 # 采样器
│   ├── preprocessing.py           # 数据增强
│   └── bases.py
├── loss/                          # 损失函数
│   ├── arcface.py                 # ArcFace, CircleLoss（分类层）
│   ├── metric_learning.py         # Arcface, Cosface, AMSoftmax, CircleLoss
│   ├── softmax_loss.py            # CrossEntropy + LabelSmooth
│   ├── triplet_loss.py            # Triplet Loss (Hard Mining)
│   ├── center_loss.py             # Center Loss
│   └── make_loss.py               # Loss 组装工厂
├── model/                         # 模型定义
│   ├── make_model.py              # ★ 模型构建（Backbone/Transformer/JPM）
│   └── backbones/
│       ├── vit_pytorch.py         # ViT 实现（TransReID 扩展版）
│       └── resnet.py              # ResNet50 backbone（备选）
├── processor/
│   └── processor.py               # 训练/推理循环（EMA、AMP、TensorBoard）
├── solver/                        # 优化器和学习率调度
│   ├── make_optimizer.py          # SGD/AdamW 优化器构建
│   ├── scheduler_factory.py       # 学习率调度策略
│   ├── cosine_lr.py
│   └── lr_scheduler.py
├── utils/                         # 工具
│   ├── metrics.py                 # mAP / Rank-K / Re-Ranking 评估
│   ├── ema.py                     # 指数移动平均
│   ├── migrate_checkpoint.py      # ★ 旧 checkpoint 迁移脚本
│   ├── logger.py
│   └── meter.py
├── gui/                           # PyQt5 图形化推理界面
│   ├── app.py
│   ├── main_window.py
│   ├── model_thread.py
│   └── widgets.py
├── train.py                       # ★ 训练入口
├── test.py                        # ★ 单模型测试入口
├── ensemble_test.py               # ★ 多模型集成测试入口
├── visualize_aug.py               # 数据增强可视化
├── run_gui.py                     # GUI 入口
├── requirements.txt
└── README.md
```

---

## 环境搭建

### 依赖

- Python 3.8+
- PyTorch 1.12+ (推荐 2.0+)
- CUDA 11.6+ (推荐 12.1)
- NVIDIA GPU (推荐 RTX 4090 / 24GB+ 显存)

### 安装

```bash
# 创建虚拟环境
python -m venv .venv
.venv\Scripts\activate      # Windows
# source .venv/bin/activate # Linux/Mac

# 安装 PyTorch（根据 CUDA 版本选择）
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121

# 安装其余依赖
pip install yacs timm opencv-python PyQt5 tensorboard
```

---

## 数据准备

将比赛数据集解压到 `data/BallShow/`，目录结构如下：

```
data/BallShow/
├── bounding_box_train/     # 训练集图片
├── bounding_box_test/      # 测试集（gallery）图片
└── query/                  # 查询集图片
```

**图片命名规范**：`{PID}_c{CameraID}_{...}.jpg`

- `PID`：球员身份 ID（-1 表示无效样本）
- `CameraID`：摄像头编号（从 1 开始）

---

## 预训练权重

下载 ViT-Base/16 在 ImageNet 上的预训练权重：

```bash
# 下载到项目根目录
wget https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-vitjx/jx_vit_base_p16_224-80ecf9dd.pth
```

> 配置文件默认从项目根目录读取 `./jx_vit_base_p16_224-80ecf9dd.pth`，也可修改配置中的 `MODEL.PRETRAIN_PATH` 为你的实际路径。

---

## 使用流程

### 完整工作流总览

```
┌──────────┐    ┌──────────────┐    ┌──────────┐    ┌──────────────┐    ┌────────────┐
│ 1. Train │───►│ 2. Migrate   │───►│ 3. Test  │───►│ 4. Ensemble  │───►│ 5. GUI     │
│  train.py│    │  (if needed) │    │  test.py │    │  ensemble    │    │  run_gui.py│
└──────────┘    └──────────────┘    └──────────┘    └──────────────┘    └────────────┘
```

### Step 1: 训练模型

使用推荐的 `vit_transreid_stride` 配置开始训练：

```bash
python train.py \
    --config_file configs/BallShow/vit_transreid_stride.yml \
    MODEL.DEVICE_ID "('0')"
```

**关键训练特性（自动启用）：**
- ✅ EMA（验证时自动使用 EMA 模型评估）
- ✅ AMP 混合精度训练（加速 + 省显存）
- ✅ 自动保存最优模型（`transformer_best.pth` / `transformer_ema_best.pth`）
- ✅ TensorBoard 日志（`SOLVER.TB_LOG: True`）
- ✅ 梯度裁剪（`SOLVER.GRAD_CLIP: 1.0`）
- ✅ Warmup 学习率（5 epochs linear warmup）

**训练输出（`logs/BallShow_vit_transreid_stride/`）：**

| 文件 | 说明 |
|------|------|
| `transformer_{epoch}.pth` | 定期保存的检查点 |
| `transformer_ema_{epoch}.pth` | EMA 版本的检查点 |
| `transformer_best.pth` | 验证集 mAP 最高的模型 |
| `transformer_ema_best.pth` | 验证集 mAP 最高的 EMA 模型 |

**常用命令行覆盖参数：**

```bash
# 使用特定 GPU
MODEL.DEVICE_ID "('3')"

# 调整训练 epoch 数
SOLVER.MAX_EPOCHS 200

# 关闭 TensorBoard
SOLVER.TB_LOG False

# 增大 batch size
SOLVER.IMS_PER_BATCH 128

# 不保存最优模型
SOLVER.EVAL_BEST False
```

### Step 2: 迁移旧 Checkpoint（可选）

> **何时需要**：如果你有在「方案1修复」之前训练的旧模型权重，直接加载会因缺少 `classifier_1~4.weight` 而报错。

**① 先预览（推荐）**：

```bash
python utils/migrate_checkpoint.py \
    --input logs/BallShow_vit_transreid_stride/transformer_120.pth \
    --dry-run
```

输出会告诉你哪些 key 将被创建。

**② 迁移单个文件**：

```bash
python utils/migrate_checkpoint.py \
    --input logs/old/transformer_120.pth \
    --output logs/new/transformer_120.pth
```

**③ 批量迁移整个目录**：

```bash
python utils/migrate_checkpoint.py \
    --input-dir logs/BallShow_vit_transreid_stride/ \
    --output-dir logs/BallShow_vit_transreid_stride_migrated/
```

> **原理**：将已训练好的全局分支 `classifier.weight`（ArcFace 权重矩阵）复制到缺失的4个局部分支 `classifier_1~4.weight`。全局分类器已经学到了有意义的类中心，这比随机初始化好得多，fine-tune 几轮即可适应。

### Step 3: 测试评估

**单模型测试（含 TTA 水平翻转 + Re-Ranking）：**

```bash
python test.py \
    --config_file configs/BallShow/vit_transreid_stride.yml \
    MODEL.DEVICE_ID "('0')" \
    TEST.WEIGHT "logs/BallShow_vit_transreid_stride/transformer_ema_best.pth"
```

> **推荐**：使用 `transformer_ema_best.pth`（EMA 版本），通常比普通模型高 0.2-0.5% mAP。

**多尺度测试**：

```bash
python test.py \
    --config_file configs/BallShow/vit_transreid_stride.yml \
    TEST.MULTI_SCALE True \
    TEST.SCALES "([256,128],[384,128],[384,192])" \
    TEST.WEIGHT "logs/.../transformer_ema_best.pth"
```

**开启 Query Expansion**：

```bash
python test.py \
    --config_file configs/BallShow/vit_transreid_stride.yml \
    TEST.QE True \
    TEST.QE_K 10 \
    TEST.WEIGHT "logs/.../transformer_ema_best.pth"
```

### Step 4: 多模型 Ensemble 测试（可选）

融合不同 epoch 的 EMA 模型，进一步榨取性能：

```bash
python ensemble_test.py \
    --config_file configs/BallShow/vit_transreid_stride.yml \
    MODEL.DEVICE_ID "('0')" \
    --weights logs/BallShow_vit_transreid_stride/transformer_ema_120.pth,logs/BallShow_vit_transreid_stride/transformer_ema_140.pth,logs/BallShow_vit_transreid_stride/transformer_ema_best.pth
```

> 多个模型的特征在 score-level 融合（距离矩阵取平均），通常可提升 0.3-1% mAP。

### Step 5: GUI 可视化推理（可选）

```bash
python run_gui.py
```

提供图形化界面：加载模型权重 → 选择查询图片 → 浏览检索结果。

---

## 配置文件说明

| 配置文件 | Backbone | JPM | SIE | Stride | ID Loss | 适用场景 |
|----------|----------|-----|-----|--------|---------|----------|
| **[vit_transreid_stride.yml](configs/BallShow/vit_transreid_stride.yml)** | ViT-B/16 | ✅ | ✅ Camera | 12×12 | ArcFace | **★ 推荐** |
| [vit_jpm.yml](configs/BallShow/vit_jpm.yml) | ViT-B/16 | ✅ | ❌ | 16×16 | Softmax | 消融实验 |
| [vit_sie.yml](configs/BallShow/vit_sie.yml) | ViT-B/16 | ❌ | ✅ Camera | 16×16 | Softmax | 消融实验 |
| [vit_transreid.yml](configs/BallShow/vit_transreid.yml) | ViT-B/16 | ✅ | ✅ Camera | 16×16 | Softmax | 基础 TransReID |
| [vit_transreid_384.yml](configs/BallShow/vit_transreid_384.yml) | ViT-B/16 | ✅ | ✅ Camera | 16×16 | Softmax | 384×128 输入 |
| [vit_transreid_stride_384.yml](configs/BallShow/vit_transreid_stride_384.yml) | ViT-B/16 | ✅ | ✅ Camera | 12×12 | Softmax | 大分辨率 |
| [vit_base.yml](configs/BallShow/vit_base.yml) | ViT-B/16 | ❌ | ❌ | 16×16 | Softmax | 最简 Baseline |
| [deit_transreid_stride.yml](configs/BallShow/deit_transreid_stride.yml) | DeiT-S/16 | ✅ | ✅ Camera | 12×12 | Softmax | 轻量（384维） |

---

## 关键优化点

本实现相比原始 TransReID 做了以下增强：

| # | 优化项 | 说明 | 预期收益 |
|---|--------|------|----------|
| 1 | **JPM 全分支 ArcFace** | 修复局部分支无 ID Loss 的 Bug，5个分支各独立 ArcFace | mAP +2~4% |
| 2 | **Overlapping Stride (12×12)** | 比默认 16×16 保留更细粒度 patch 信息 | mAP +0.5~1% |
| 3 | **EMA 模型验证** | 训练时维护指数移动平均权重，测试用 EMA 模型 | mAP +0.2~0.5% |
| 4 | **AMP 混合精度** | 加速训练 1.5-2×，显存节省 ~30% | 训练效率 |
| 5 | **TTA 水平翻转** | 测试时原图+翻转特征平均 | mAP +0.3~0.8% |
| 6 | **Re-Ranking** | k-reciprocal 重排序 | mAP +1~3% |
| 7 | **Label Smoothing** | 防止过拟合，提升泛化 | mAP +0.3~0.5% |
| 8 | **Gradient Clipping** | 稳定训练，防止梯度爆炸 | 训练稳定性 |
| 9 | **最优模型保存** | 自动追踪并保存验证集最佳 mAP 的模型 | 工程便利 |
| 10 | **多模型 Ensemble** | 不同 epoch EMA 模型的分数级融合 | mAP +0.3~1% |

---

## 实验结果

| 评估指标 | Baseline | 赛题要求 |
|:--------:|:--------:|:--------:|
| **mAP** | 91.1% | ≥ 91.5% |
| **Rank-1 Accuracy** | 93.6% | ≥ 94% |

> 上述结果为原始 TransReID Baseline。最后优化的实验结果待训练完成后更新。

---

## 后续优化路线

按优先级排列的待实施方案：

| 优先级 | 方案 | 说明 |
|--------|------|------|
| ⭐1 | ✅ JPM 全分支 ArcFace | 已完成 — 5个分支各有独立 ArcFace 监督 |
| ⭐2 | AdamW 优化器 | ViT 对 AdamW 更友好，预期 mAP +0.5~1.5% |
| ⭐3 | Circle Loss 替代 Triplet+Center | 统一类内/类间优化，预期 mAP +1~2% |
| ⭐4 | 特征维度压缩 (3840→512) | 满足 ≤40ms 推理延迟硬约束 |
| 5 | 更强数据增强 | Random Patch Drop、MixStyle、CutMix 等 |
| 6 | Multi-Scale Testing | 多尺度推理融合 |
| 7 | Query Expansion | 查询扩展增强召回 |
| 8 | ViT-Large Backbone | 更大模型容量，需权衡推理速度 |

---

## 引用

本实现基于 TransReID：

```bibtex
@InProceedings{He_2021_ICCV,
    author    = {He, Shuting and Luo, Hao and Wang, Pichao and Wang, Fan and Li, Hao and Jiang, Wei},
    title     = {TransReID: Transformer-Based Object Re-Identification},
    booktitle = {Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV)},
    month     = {October},
    year      = {2021},
    pages     = {15013-15022}
}
```

官方仓库：[https://github.com/damo-cv/TransReID](https://github.com/damo-cv/TransReID)

ViT 实现参考：[https://github.com/rwightman/pytorch-image-models](https://github.com/rwightman/pytorch-image-models)
