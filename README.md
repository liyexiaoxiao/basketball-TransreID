# 篮球持球人身份重识别（Ball-Handler Re-Identification）

基于 **TransReID**（Transformer-based Object Re-Identification）的篮球场景持球人重识别算法，针对复杂运动场景中的球衣相似、遮挡、运动模糊等挑战设计。

## 赛题背景

在真实的"球秀"业务场景中，需要在不依赖连续轨迹追踪的前提下，仅凭外观特征跨时刻检索特定持球人。相较于传统行人重识别，篮球场景面临三大核心挑战：

| 挑战 | 描述 |
|------|------|
| **相似球衣区分** | 队友穿着完全相同的球衣，需依赖号码（常被手臂或球体遮挡）、体态、护具、发型、鞋色等细粒度特征 |
| **严重遮挡与姿态多变** | 多人包夹重叠遮挡，运球/上篮/投篮等动作导致身体姿态剧烈形变 |
| **环境干扰** | 室内木地板强反光、室外复杂背景、夜间灯光、持球人高速移动导致的运动模糊 |

## 核心组件

| 组件 | 说明 |
|------|------|
| **Backbone** | ViT-Base/16，12层 Transformer，768维特征嵌入 |
| **Overlapping Patches** | stride=12×12 的滑动窗口 patch embedding（vs 标准 16×16），保留更细粒度的局部纹理 |
| **SIE (Side Information Embedding)** | 将摄像头编号编码为可学习 embedding，在 Transformer 输入端注入，帮助模型学习摄像头不变性 |
| **JPM (Jigsaw Patch Module)** | 将 patch tokens 均匀切分为4组，每组与 CLS token 一起送入共享 Block 提取局部细粒度特征。配合 Shift + Shuffle 增强鲁棒性 |
| **BNNeck** | 每个分支独立的 BatchNorm 颈层，缓解 ID Loss 与 Metric Loss 在特征空间的优化冲突 |
| **EMA (Exponential Moving Average)** | 训练时维护模型参数的平滑副本（decay=0.9995），测试时使用 EMA 模型提升 0.2~0.5% mAP |
| **TTA (Test-Time Augmentation)** | 测试时对原图+水平翻转提取特征并平均，充分利用双向信息 |
| **Re-Ranking** | k-reciprocal 重排序，利用查询-库图结构精排检索结果，提升 mAP 1~3% |

## 损失函数

支持两种 ID Loss，通过 `MODEL.ID_LOSS_TYPE` 切换：

### Softmax 模式（`ID_LOSS_TYPE: 'softmax'`）

```
Total Loss = ID_LOSS_WEIGHT × CrossEntropy + TRIPLET_LOSS_WEIGHT × Triplet + CENTER_LOSS_WEIGHT × Center
```

5个 JPM 分支各输出 softmax logits + 特征，ID/Triplet Loss 均以 50%全局 + 50%局部分支均值加权。

### ArcFace 模式（`ID_LOSS_TYPE: 'arcface'`）

```
Total Loss = ID_LOSS_WEIGHT × ArcFace + TRIPLET_LOSS_WEIGHT × Triplet + CENTER_LOSS_WEIGHT × Center
```

每个 JPM 分支拥有**独立的 ArcFace 分类器**（additive angular margin, s=30, m=0.5），全局与局部分支各算各的角边际损失后加权求和。相比 Softmax，ArcFace 在类内紧凑性和类间分离性上表现更强。

> 额外支持 `cosface`、`amsoftmax`、`circle`（分类器级 Circle Loss）作为 ID Loss。

## 项目结构

```
TransReID/
├── config/
│   └── defaults.py                    # 所有默认配置项
├── configs/BallShow/                  # 篮球数据集配置文件
│   ├── vit_transreid_stride.yml       # ★ 推荐：JPM+SIE+Stride
│   ├── vit_transreid.yml              # JPM+SIE 标准版
│   ├── vit_jpm.yml                    # 纯 JPM 消融
│   ├── vit_sie.yml                    # 纯 SIE 消融
│   └── vit_base.yml                   # 最简基线
├── data/BallShow/
│   ├── bounding_box_train/            # 训练集
│   ├── bounding_box_test/             # 测试集（gallery）
│   └── query/                         # 查询集
├── datasets/
│   ├── ballshow.py                    # BallShow 数据集解析
│   └── make_dataloader.py             # DataLoader + 增强 pipeline
├── loss/
│   ├── metric_learning.py             # Arcface / Cosface / AMSoftmax / CircleLoss (ID级)
│   ├── triplet_loss.py                # Triplet Loss (Hard Mining)
│   ├── center_loss.py                 # Center Loss
│   ├── softmax_loss.py                # CrossEntropy + Label Smoothing
│   └── make_loss.py                   # ★ 损失函数组装工厂
├── model/
│   ├── make_model.py                  # ★ 模型构建（Transformer + JPM）
│   └── backbones/
│       ├── vit_pytorch.py             # ViT 实现（TransReID 扩展：SIE + JPM）
│       └── resnet.py                  # ResNet50 备选 backbone
├── processor/
│   └── processor.py                   # 训练/推理循环（EMA + AMP + TTA）
├── solver/
│   ├── make_optimizer.py              # SGD / AdamW 优化器
│   ├── scheduler_factory.py           # Cosine Annealing + Warmup
│   └── cosine_lr.py
├── utils/
│   ├── metrics.py                     # mAP / Rank-K / Re-Ranking 评估
│   ├── ema.py                         # 指数移动平均
│   ├── logger.py
│   └── meter.py
├── train.py                           # ★ 训练入口
├── test.py                            # ★ 单模型测试入口
├── ensemble_test.py                   # ★ 多模型集成测试
├── visualize_aug.py                   # 数据增强可视化
└── requirements.txt
```

## 环境搭建

- Python 3.8+
- PyTorch 1.12+ (推荐 2.0+)
- CUDA 11.6+
- NVIDIA GPU (推荐 RTX 4090 / 24GB+)

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
pip install yacs timm opencv-python
```

## 数据准备

解压数据集到 `data/BallShow/`：

```
data/BallShow/
├── bounding_box_train/     # 训练集
├── bounding_box_test/      # 测试集（gallery）
└── query/                  # 查询集
```

图片命名规范：`{PID}_c{CameraID}_{...}.jpg`
- `PID`：球员身份 ID（-1 表示无效样本，自动过滤）
- `CameraID`：摄像头编号（从 1 开始）

## 预训练权重

```bash
wget https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-vitjx/jx_vit_base_p16_224-80ecf9dd.pth
```

放在项目根目录，配置中 `MODEL.PRETRAIN_PATH: './jx_vit_base_p16_224-80ecf9dd.pth'`。

---

## 使用流程

### 训练

```bash
python train.py \
    --config_file configs/BallShow/vit_transreid_stride.yml \
    MODEL.DEVICE_ID "('0')" \
    SOLVER.MAX_EPOCHS 130
```

**启用 ArcFace（推荐，需要方案1代码支持）：**

```bash
python train.py \
    --config_file configs/BallShow/vit_transreid_stride.yml \
    MODEL.DEVICE_ID "('0')" \
    MODEL.ID_LOSS_TYPE "('arcface')" \
    SOLVER.COSINE_SCALE 30 \
    SOLVER.COSINE_MARGIN 0.5 \
    SOLVER.MAX_EPOCHS 130
```

**切换到 AdamW（ViT 优化更友好）：**

```bash
python train.py \
    --config_file configs/BallShow/vit_transreid_stride.yml \
    MODEL.DEVICE_ID "('0')" \
    SOLVER.OPTIMIZER_NAME "('AdamW')" \
    SOLVER.BASE_LR 0.00035 \
    SOLVER.WEIGHT_DECAY 0.05 \
    SOLVER.BIAS_LR_FACTOR 1.0 \
    SOLVER.MAX_EPOCHS 130
```

**训练输出（`logs/BallShow_vit_transreid_stride/`）：**

| 文件 | 说明 |
|------|------|
| `transformer_{epoch}.pth` | 定期保存的检查点 |
| `transformer_ema_{epoch}.pth` | EMA 版本检查点（测试推荐使用） |

### 测试

**单模型（含 TTA 水平翻转 + Re-Ranking）：**

```bash
python test.py \
    --config_file configs/BallShow/vit_transreid_stride.yml \
    MODEL.DEVICE_ID "('0')" \
    TEST.WEIGHT "logs/BallShow_vit_transreid_stride/transformer_ema_120.pth"
```

### 多模型 Ensemble

融合不同 epoch 的 EMA 模型，在分数级（距离矩阵平均）集成：

```bash
python ensemble_test.py \
    --config_file configs/BallShow/vit_transreid_stride.yml \
    MODEL.DEVICE_ID "('0')" \
    --weights logs/BallShow_vit_transreid_stride/transformer_ema_80.pth,logs/BallShow_vit_transreid_stride/transformer_ema_120.pth
```

---

## 配置文件说明

| 配置文件 | SIE | JPM | Stride | 说明 |
|----------|-----|-----|--------|------|
| **[vit_transreid_stride.yml](configs/BallShow/vit_transreid_stride.yml)** | ✅ Camera | ✅ | 12×12 | **★ 推荐** |
| [vit_transreid.yml](configs/BallShow/vit_transreid.yml) | ✅ Camera | ✅ | 16×16 | 标准 TransReID |
| [vit_jpm.yml](configs/BallShow/vit_jpm.yml) | ❌ | ✅ | 16×16 | JPM 消融 |
| [vit_sie.yml](configs/BallShow/vit_sie.yml) | ✅ Camera | ❌ | 16×16 | SIE 消融 |
| [vit_base.yml](configs/BallShow/vit_base.yml) | ❌ | ❌ | 16×16 | 最简基线 |

### 关键配置项

```yaml
MODEL:
  ID_LOSS_TYPE: 'arcface'       # 'softmax' | 'arcface' | 'cosface' | 'amsoftmax' | 'circle'
  METRIC_LOSS_TYPE: 'triplet'   # 'triplet' | 'triplet_center'
  JPM: True                     # 启用 Jigsaw Patch Module（5分支）
  SIE_CAMERA: True              # 摄像头信息嵌入

SOLVER:
  OPTIMIZER_NAME: 'SGD'         # 'SGD' | 'AdamW'
  BASE_LR: 0.008                # SGD: 0.008; AdamW: 0.00035
  WEIGHT_DECAY: 1e-4            # SGD: 1e-4; AdamW: 0.05
  COSINE_SCALE: 30              # ArcFace scale s
  COSINE_MARGIN: 0.5            # ArcFace margin m

TEST:
  RE_RANKING: True              # k-reciprocal 重排序
```

---

## 实验结果

| 评估指标 | Baseline | 赛题要求 |
|:--------:|:--------:|:--------:|
| **mAP** | 91.1% | ≥ 91.5% |
| **Rank-1 Accuracy** | 93.6% | ≥ 94% |

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

ArcFace: [Deng et al., ArcFace: Additive Angular Margin Loss for Deep Face Recognition, CVPR 2019](https://arxiv.org/abs/1801.07698)
