"""
Circle Loss (Feature-Level Metric Learning)

Reference:
  Sun et al. "Circle Loss: A Unified Perspective of Pair Similarity Optimization"
  CVPR 2020, https://arxiv.org/abs/2002.10857

Circle Loss unifies class-level and pair-level optimization by adaptively
weighting each similarity score. It gives more gradient to:
  - Hard positive pairs (low similarity within same class)
  - Hard negative pairs (high similarity across different classes)

This replaces Triplet + Center Loss with a single, principled objective.
"""

import torch
import torch.nn.functional as F


def circle_loss(features, labels, scale=256.0, margin=0.25):
    """
    Compute feature-level Circle Loss for a batch.

    Args:
        features: [B, D] feature matrix (should be L2-normalized)
        labels:   [B]    ground-truth labels
        scale:    γ, scale factor (default 256, per paper)
        margin:   m, relaxation factor (default 0.25, per paper)

    Returns:
        loss: scalar tensor

    Formulation:
        For each anchor x with similarities to K positive pairs and L negative pairs:
            s_p^i = cos(x, pos_i)
            s_n^j = cos(x, neg_j)

        Adaptive weights:
            α_p^i = max(1+m - s_p^i, 0)    # high when positive is hard
            α_n^j = max(s_n^j + m, 0)      # high when negative is hard

        Optimal similarities:
            O_p = 1+m,  O_n = -m
            Δ_p = 1-m,  Δ_n = m

        L = softplus( log Σ exp(γ·α_n·(s_n - Δ_n)) + log Σ exp(γ·α_p·(Δ_p - s_p)) )

        Gradients are adaptively scaled — hard pairs dominate, easy pairs decay.
    """
    B = features.size(0)
    device = features.device

    # Cosine similarity matrix [B, B]
    sim_mat = features @ features.T

    # Positive / negative masks (exclude self-pairs)
    labels = labels.view(-1, 1)
    pos_mask = (labels == labels.T).float()
    neg_mask = (labels != labels.T).float()
    pos_mask.fill_diagonal_(0.0)

    with torch.no_grad():
        Op = 1.0 + margin
        On = -margin

        # Adaptive weights (detach to block gradient through weights)
        alpha_p = torch.clamp_min(Op - sim_mat, min=0.0)
        alpha_n = torch.clamp_min(sim_mat - On, min=0.0)

    # Target deltas
    Dp = 1.0 - margin
    Dn = margin

    # Weighted terms
    pos_terms = -scale * alpha_p * (sim_mat - Dp)   # -(Δ_p - s_p) negative for exp
    neg_terms = scale * alpha_n * (sim_mat - Dn)

    # Mask and compute log-sum-exp per anchor
    pos_terms = pos_terms * pos_mask
    neg_terms = neg_terms * neg_mask

    # To avoid -inf in log-sum-exp on empty sets, add a large negative fill
    LARGE_NEG = -1e9

    pos_terms = torch.where(pos_mask.bool(), pos_terms,
                            torch.tensor(LARGE_NEG, device=device))
    neg_terms = torch.where(neg_mask.bool(), neg_terms,
                            torch.tensor(LARGE_NEG, device=device))

    # log-sum-exp: log(Σ exp(x))
    log_pos = torch.logsumexp(pos_terms, dim=1)  # [B]
    log_neg = torch.logsumexp(neg_terms, dim=1)  # [B]

    # softplus(log_pos + log_neg) = log(1 + exp(log_pos + log_neg))
    loss = F.softplus(log_pos + log_neg)

    return loss.mean()


class CircleMetricLoss:
    """
    Circle Loss as a drop-in replacement for Triplet Loss.

    Usage identical to TripletLoss:
        criterion = CircleMetricLoss(scale=256.0, margin=0.25)
        loss = criterion(features, labels, normalize_feature=True)

    Returns (loss,) tuple for compatibility with the existing loss pipeline.
    """

    def __init__(self, scale=256.0, margin=0.25):
        self.scale = scale
        self.margin = margin

    def __call__(self, features, labels, normalize_feature=True):
        if normalize_feature:
            features = F.normalize(features, p=2, dim=1)
        loss = circle_loss(features, labels, scale=self.scale, margin=self.margin)
        return loss, None, None  # match TripletLoss return format: (loss, dist_ap, dist_an)
