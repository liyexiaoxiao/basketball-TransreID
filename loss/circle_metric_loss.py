"""
Circle Loss for Metric Learning (Feature-Level)

Reference:
    Sun et al. "Circle Loss: A Unified Perspective of Pair Similarity Optimization"
    CVPR 2020, https://arxiv.org/abs/2002.10857

Unlike the classifier-based CircleLoss in metric_learning.py (which replaces ArcFace),
this is a pairwise metric loss that operates directly on L2-normalized features,
replacing the Triplet + Center loss combination.

Key properties:
- Unifies within-class compactness (←) and between-class separation (→) under one objective
- Adaptive weighting: easy pairs get less gradient, hard pairs get more
- No separate margin for pos/neg — a single relaxation factor `m` controls both
- Mathematically: O_p = 1+m (target for positive), O_n = -m (target for negative)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class CircleLoss(nn.Module):
    """Circle Loss for deep metric learning.

    For each sample in the batch:
        L = log[1 + Σ exp(γ·α_p^j·(s_p^j - Δ_p)) · Σ exp(γ·α_n^k·(s_n^k - Δ_n))]

    where:
        s_p^j = cosine similarity to j-th positive sample
        s_n^k = cosine similarity to k-th negative sample
        α_p^j = max(0, 1+m - s_p^j)  — adaptive weight (positive)
        α_n^k = max(0, s_n^k + m)     — adaptive weight (negative)
        Δ_p = 1-m  — target for positive similarity
        Δ_n = m    — target for negative similarity

    Args:
        scale:   Scale factor γ (default 128). Higher → sharper decision boundary.
        margin:  Relaxation margin m (default 0.25). Controls the target similarity gap.
                 Δ_p - Δ_n = (1-m) - m = 1-2m. With m=0.25, gap=0.5.
    """

    def __init__(self, scale: float = 128.0, margin: float = 0.25):
        super(CircleLoss, self).__init__()
        self.scale = scale
        self.margin = margin
        # Optimal similarity targets
        self.O_p = 1.0 + margin   # target for positive pairs
        self.O_n = -margin        # target for negative pairs
        # Corresponding deltas (used in original formulation)
        self.Delta_p = 1.0 - margin
        self.Delta_n = margin

    def forward(self, features: torch.Tensor, labels: torch.Tensor):
        """
        Args:
            features: [B, D] — L2-normalized feature vectors
            labels:   [B]    — identity labels
        Returns:
            scalar loss
        """
        batch_size = features.size(0)
        device = features.device

        # 1. Cosine similarity matrix
        sim_mat = features @ features.T  # [B, B], values in [-1, 1]

        # 2. Build positive/negative masks
        labels = labels.view(-1, 1)
        pos_mask = labels.eq(labels.T).float()   # same identity
        neg_mask = 1.0 - pos_mask                 # different identity
        pos_mask.fill_diagonal_(0.0)              # remove self-pair

        # 3. Adaptive weighting factors (detached to avoid gradient through weights)
        #    α_p = max(0, 1+m - sim)   ;   α_n = max(0, sim + m)
        alpha_p = torch.clamp_min(self.O_p - sim_mat.detach(), min=0.0)
        alpha_n = torch.clamp_min(sim_mat.detach() - self.O_n, min=0.0)

        # 4. Weighted logits
        #    For positives: push sim → O_p (1+m)
        #    For negatives: push sim → O_n (-m)
        logit_p = -self.scale * alpha_p * (sim_mat - self.Delta_p) * pos_mask
        logit_n =  self.scale * alpha_n * (sim_mat - self.Delta_n) * neg_mask

        # 5. Log-sum-exp for numerical stability
        #    L = log[1 + exp(Σ_pos + Σ_neg)] = softplus(Σ_pos + Σ_neg)
        sum_p = torch.logsumexp(logit_p, dim=1)   # sum over positives for each anchor
        sum_n = torch.logsumexp(logit_n, dim=1)   # sum over negatives for each anchor

        # Handle anchors without positive or negative pairs gracefully
        has_pos = pos_mask.sum(dim=1) > 0
        has_neg = neg_mask.sum(dim=1) > 0

        loss = torch.zeros(batch_size, device=device)
        # Only compute loss for anchors that have BOTH positive and negative pairs
        valid = has_pos & has_neg
        if valid.sum() > 0:
            loss[valid] = F.softplus(sum_p[valid] + sum_n[valid])

        # For anchors with only positives (rare), encourage compactness
        only_pos = has_pos & ~has_neg
        if only_pos.sum() > 0:
            loss[only_pos] = F.softplus(sum_p[only_pos])

        # For anchors with only negatives (rare), encourage separation
        only_neg = ~has_pos & has_neg
        if only_neg.sum() > 0:
            loss[only_neg] = F.softplus(sum_n[only_neg])

        return loss.mean()


# Convenience aliases matching the naming convention in the project
circle_loss = CircleLoss


if __name__ == "__main__":
    # Quick smoke test
    B, D, C = 16, 768, 4  # batch, dim, classes
    feats = F.normalize(torch.randn(B, D), dim=1)
    labels = torch.arange(C).repeat(B // C)  # 4 samples per class

    criterion = CircleLoss(scale=128, margin=0.25)
    loss = criterion(feats, labels)
    print(f"Circle Loss (metric): {loss.item():.4f}")
    print(f"Scale: {criterion.scale}, Margin: {criterion.margin}")
    print(f"Target O_p: {criterion.O_p}, O_n: {criterion.O_n}")
