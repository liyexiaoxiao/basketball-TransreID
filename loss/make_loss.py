# encoding: utf-8
"""
Loss function factory.

Supports:
  - ID Loss:   Softmax, ArcFace/Cosface/AMSoftmax/Circle (set in model via ID_LOSS_TYPE)
  - Metric Loss: Triplet, Triplet+Center, Circle
  - Label Smoothing (configurable via IF_LABELSMOOTH)
"""

import torch.nn.functional as F
from .softmax_loss import CrossEntropyLabelSmooth, LabelSmoothingCrossEntropy
from .triplet_loss import TripletLoss
from .center_loss import CenterLoss
from .circular_loss import CircleMetricLoss


def make_loss(cfg, num_classes):
    sampler = cfg.DATALOADER.SAMPLER

    # ── Auto-detect feature dimension ──────────────────────────────────────
    if cfg.MODEL.NAME == 'transformer':
        feat_dim = 768
        if cfg.MODEL.TRANSFORMER_TYPE and 'deit_small' in cfg.MODEL.TRANSFORMER_TYPE:
            feat_dim = 384
    else:
        feat_dim = 2048

    # ── Center Loss (only when explicitly requested) ────────────────────────
    center_criterion = CenterLoss(num_classes=num_classes, feat_dim=feat_dim, use_gpu=True)
    print(f"CenterLoss initialized with feat_dim={feat_dim}, num_classes={num_classes}")

    # ── Metric Loss ─────────────────────────────────────────────────────────
    metric_loss_type = cfg.MODEL.METRIC_LOSS_TYPE

    if 'circle' in metric_loss_type:
        circle_s = getattr(cfg.SOLVER, 'CIRCLE_S', 256.0)
        circle_m = getattr(cfg.SOLVER, 'CIRCLE_M', 0.25)
        metric_loss_fn = CircleMetricLoss(scale=circle_s, margin=circle_m)
        print(f"using Circle Loss (feature-level) with scale={circle_s}, margin={circle_m}")

    elif 'triplet' in metric_loss_type:
        if cfg.MODEL.NO_MARGIN:
            metric_loss_fn = TripletLoss()
            print("using soft triplet loss for training")
        else:
            metric_loss_fn = TripletLoss(cfg.SOLVER.MARGIN)
            print("using triplet loss with margin:{}".format(cfg.SOLVER.MARGIN))
    else:
        print('expected METRIC_LOSS_TYPE should be triplet, triplet_center, or circle'
              ' but got {}'.format(metric_loss_type))

    # ── Label Smoothing ─────────────────────────────────────────────────────
    if cfg.MODEL.IF_LABELSMOOTH == 'on':
        xent = CrossEntropyLabelSmooth(num_classes=num_classes)
        print("label smooth on, numclasses:", num_classes)

    # ── Helper: compute weighted ID loss across branches ────────────────────
    def _compute_id_loss(score, target):
        """Compute ID loss, handling both single-tensor and list-of-tensors."""
        if isinstance(score, list):
            if cfg.MODEL.IF_LABELSMOOTH == 'on':
                branch_losses = [xent(s, target) for s in score[1:]]
                avg_local = sum(branch_losses) / len(branch_losses)
                return 0.5 * avg_local + 0.5 * xent(score[0], target)
            else:
                branch_losses = [F.cross_entropy(s, target) for s in score[1:]]
                avg_local = sum(branch_losses) / len(branch_losses)
                return 0.5 * avg_local + 0.5 * F.cross_entropy(score[0], target)
        else:
            if cfg.MODEL.IF_LABELSMOOTH == 'on':
                return xent(score, target)
            else:
                return F.cross_entropy(score, target)

    # ── Helper: compute weighted metric loss across branches ────────────────
    def _compute_metric_loss(feat, target):
        """Compute metric loss, handling both single-tensor and list-of-tensors."""
        if isinstance(feat, list):
            branch_losses = [metric_loss_fn(f, target)[0] for f in feat[1:]]
            avg_local = sum(branch_losses) / len(branch_losses)
            return 0.5 * avg_local + 0.5 * metric_loss_fn(feat[0], target)[0]
        else:
            return metric_loss_fn(feat, target)[0]

    # ── Helper: compute center loss ─────────────────────────────────────────
    def _compute_center_loss(feat, target):
        """Center loss on global branch only (feat[0] for JPM, feat itself otherwise)."""
        if isinstance(feat, list):
            return center_criterion(feat[0], target)
        else:
            return center_criterion(feat, target)

    # ── Build loss function closure ─────────────────────────────────────────
    if sampler == 'softmax':
        def loss_func(score, feat, target):
            return F.cross_entropy(score, target)

    elif sampler == 'softmax_triplet':
        def loss_func(score, feat, target, target_cam):
            valid_types = ('triplet', 'triplet_center', 'circle', 'circle_center')

            if metric_loss_type in valid_types:
                # Core losses
                id_loss = _compute_id_loss(score, target)
                metric_loss = _compute_metric_loss(feat, target)

                total_loss = (cfg.MODEL.ID_LOSS_WEIGHT * id_loss +
                              cfg.MODEL.TRIPLET_LOSS_WEIGHT * metric_loss)

                # Optional center loss
                if 'center' in metric_loss_type:
                    center_loss = _compute_center_loss(feat, target)
                    total_loss = total_loss + cfg.SOLVER.CENTER_LOSS_WEIGHT * center_loss

                return total_loss
            else:
                print('expected METRIC_LOSS_TYPE in {triplet, triplet_center, circle, circle_center}'
                      ' but got {}'.format(metric_loss_type))

    else:
        print('expected sampler should be softmax or softmax_triplet'
              ' but got {}'.format(sampler))

    return loss_func, center_criterion
