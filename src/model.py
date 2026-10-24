"""二分类 MLP：4 -> 32 -> 16 -> 2。"""
from __future__ import annotations

import torch
import torch.nn as nn

from config import DROPOUT, HIDDEN_SIZES, N_CLASSES, N_FEATURES


class MLP(nn.Module):
    def __init__(
        self,
        n_features: int = N_FEATURES,
        hidden: tuple[int, ...] = HIDDEN_SIZES,
        n_classes: int = N_CLASSES,
        dropout: float = DROPOUT,
    ):
        super().__init__()
        layers: list[nn.Module] = []
        in_dim = n_features
        for index, width in enumerate(hidden):
            layers.append(nn.Linear(in_dim, width))
            layers.append(nn.ReLU())
            if dropout > 0 and index == 0:
                layers.append(nn.Dropout(dropout))
            in_dim = width
        layers.append(nn.Linear(in_dim, n_classes))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def build_model(cfg=None) -> MLP:
    if cfg is None:
        return MLP()
    return MLP(
        n_features=getattr(cfg, "N_FEATURES", N_FEATURES),
        hidden=tuple(getattr(cfg, "HIDDEN_SIZES", HIDDEN_SIZES)),
        n_classes=getattr(cfg, "N_CLASSES", N_CLASSES),
        dropout=getattr(cfg, "DROPOUT", DROPOUT),
    )


if __name__ == "__main__":
    from config import SEED
    from src.utils import set_seed

    set_seed(SEED)
    model = build_model()
    logits = model(torch.randn(4, N_FEATURES))
    print(model)
    print("output", tuple(logits.shape))
