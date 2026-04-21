"""
Exponential Moving Average (EMA) for model weights.

During training, maintains a smoothed copy of model parameters.
At test time, the EMA model typically outperforms the final model
by 0.2-0.5% because it averages out training noise.
"""

import torch
import copy


class ModelEMA:
    """
    Exponential Moving Average of model weights.

    Args:
        model: The model to track.
        decay: EMA decay factor. Higher = smoother, slower to update.
               0.9998 is standard for ~120 epoch training.
    """
    def __init__(self, model, decay=0.9998):
        self.ema = copy.deepcopy(model)
        self.ema.eval()
        self.decay = decay
        # Disable gradient for EMA model
        for p in self.ema.parameters():
            p.requires_grad_(False)

    @torch.no_grad()
    def update(self, model):
        """Update EMA parameters with current model parameters."""
        for ema_p, model_p in zip(self.ema.parameters(), model.parameters()):
            ema_p.data.mul_(self.decay).add_(model_p.data, alpha=1.0 - self.decay)

    def state_dict(self):
        return self.ema.state_dict()

    def load_state_dict(self, state_dict):
        self.ema.load_state_dict(state_dict)
