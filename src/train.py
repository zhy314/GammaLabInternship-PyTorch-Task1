"""训练、验证、保存最优权重并记录 history.csv。"""
from __future__ import annotations

import csv

import torch
import torch.nn as nn

from config import (
    BEST_MODEL_PATH,
    EPOCHS,
    HISTORY_CSV,
    LR,
    SEED,
    WEIGHT_DECAY,
)
from src.dataset import build_dataloaders
from src.model import build_model
from src.utils import save_checkpoint, set_seed


def run_epoch(model, loader, criterion, optimizer, device) -> tuple[float, float]:
    training = optimizer is not None
    model.train(training)
    total_loss = 0.0
    correct = 0
    total = 0
    for features, targets in loader:
        features = features.to(device)
        targets = targets.to(device)
        if training:
            optimizer.zero_grad()
        logits = model(features)
        loss = criterion(logits, targets)
        if training:
            loss.backward()
            optimizer.step()
        total_loss += loss.item() * targets.size(0)
        correct += (logits.argmax(dim=1) == targets).sum().item()
        total += targets.size(0)
    return total_loss / total, correct / total


def main() -> None:
    set_seed(SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_loader, val_loader, splits = build_dataloaders()
    model = build_model().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)

    history: list[dict] = []
    best_acc = -1.0
    best_epoch = 0
    for epoch in range(1, EPOCHS + 1):
        train_loss, train_acc = run_epoch(model, train_loader, criterion, optimizer, device)
        with torch.no_grad():
            val_loss, val_acc = run_epoch(model, val_loader, criterion, None, device)
        history.append(
            {
                "epoch": epoch,
                "train_loss": train_loss,
                "train_acc": train_acc,
                "val_loss": val_loss,
                "val_acc": val_acc,
            }
        )
        print(
            f"epoch={epoch:03d} train_loss={train_loss:.4f} train_acc={train_acc:.4f} "
            f"val_loss={val_loss:.4f} val_acc={val_acc:.4f}"
        )
        if val_acc > best_acc:
            best_acc = val_acc
            best_epoch = epoch
            save_checkpoint(
                BEST_MODEL_PATH,
                {
                    "model_state": model.state_dict(),
                    "means": splits["means"],
                    "stds": splits["stds"],
                    "epoch": epoch,
                    "val_acc": val_acc,
                    "seed": SEED,
                    "loss": "CrossEntropyLoss",
                    "optimizer": "Adam",
                    "lr": LR,
                    "weight_decay": WEIGHT_DECAY,
                },
            )

    HISTORY_CSV.parent.mkdir(parents=True, exist_ok=True)
    with HISTORY_CSV.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["epoch", "train_loss", "train_acc", "val_loss", "val_acc"],
        )
        writer.writeheader()
        writer.writerows(history)
    print(f"best_epoch={best_epoch} best_val_acc={best_acc:.4f}")
    print(f"saved {BEST_MODEL_PATH}")
    print(f"saved {HISTORY_CSV}")


if __name__ == "__main__":
    main()
