"""
MixStyle: Domain Generalization via Feature Statistics Mixing.

Reference:
    Zhou et al. "Domain Generalization with MixStyle", ICLR 2021.
    https://arxiv.org/abs/2104.02008

Applied after LayerNorm in ViT blocks to mix channel-wise statistics
between samples. This regularizes against domain-specific style biases
(e.g., indoor vs outdoor court lighting, different floor colors).

For basketball ReID: helps the model ignore court-specific visual styles
and focus on identity-discriminative features.
"""

import torch
import torch.nn as nn
import torch.distributions as dist


class MixStyle(nn.Module):
    """Mix feature statistics (mean, std) between samples in a batch.

    Args:
        p:     Probability of applying MixStyle in a forward pass.
        alpha: Beta distribution shape parameter (smaller = stronger mixing).
    """

    def __init__(self, p: float = 0.5, alpha: float = 0.1):
        super().__init__()
        self.p = p
        self.alpha = alpha

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B, N, C] or [B, C] feature tensor
        Returns:
            Mixed tensor of same shape
        """
        if not self.training or self.p <= 0:
            return x
        if torch.rand(1).item() > self.p:
            return x

        B = x.size(0)
        if B < 2:
            return x  # need at least 2 samples to mix

        # Reshape to [B, C, N] for easier statistics computation
        if x.dim() == 3:
            # [B, N, C] → [B, C, N]
            x_t = x.transpose(1, 2)
            spatial_dim = x.size(1)
        else:
            # [B, C] → [B, C, 1]
            x_t = x.unsqueeze(-1)
            spatial_dim = 1

        # Instance-level mean and std
        mu = x_t.mean(dim=-1, keepdim=True)   # [B, C, 1]
        var = x_t.var(dim=-1, keepdim=True)    # [B, C, 1]
        sig = (var + 1e-8).sqrt()

        # Shuffle statistics across batch
        idx = torch.randperm(B, device=x.device)
        mu_shuf = mu[idx]
        sig_shuf = sig[idx]

        # Mixing coefficient from Beta distribution
        lam = dist.Beta(self.alpha, self.alpha).sample((B, 1, 1)).to(x.device)
        # Clamp to avoid extreme values
        lam = lam.clamp(0.1, 0.9)

        # Mix
        mu_mix = lam * mu + (1 - lam) * mu_shuf
        sig_mix = lam * sig + (1 - lam) * sig_shuf

        # Normalize and rescale
        x_norm = (x_t - mu) / sig
        x_mixed = x_norm * sig_mix + mu_mix

        # Restore original shape
        if x.dim() == 3:
            return x_mixed.transpose(1, 2)  # [B, C, N] → [B, N, C]
        else:
            return x_mixed.squeeze(-1)      # [B, C, 1] → [B, C]
