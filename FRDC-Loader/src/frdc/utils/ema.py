import torch
import torch.nn as nn


class EMA:
    def __init__(
        self,
        model: nn.Module,
        ema_model: nn.Module,
    ):
        self.model = model
        self.ema_model = ema_model

    def update(self, lr: float):
        """Update the EMA model with the current model's parameters.

        Args:
            lr: A fraction controlling how much should the EMA learn from the
                current model.
        """
        for param, ema_param in zip(
            self.model.parameters(), self.ema_model.parameters()
        ):
            if ema_param.dtype == torch.float32:
                ema_param.mul_(1 - lr)
                ema_param.add_(param * lr)
