"""对比 MLP 与 kNN 基线在留出验证集和测试预测上的结果。"""
from __future__ import annotations

import csv
from collections import Counter

import numpy as np
import torch

from config import BASELINE_SUBMISSION_CSV, BEST_MODEL_PATH, COMPARE_REPORT, KNN_K, SUBMISSION_CSV
from src.dataset import build_dataloaders
from src.model import build_model
from src.utils import load_checkpoint


def read_submission(path) -> dict[str, str]:
    with open(path, encoding="utf-8") as handle:
        return {row["id"]: row["prediction"] for row in csv.DictReader(handle)}


def knn_predict(x_train: np.ndarray, y_train: np.ndarray, x_query: np.ndarray, k: int) -> np.ndarray:
    predictions = []
    for row in x_query:
        distance = np.sum((x_train - row) ** 2, axis=1)
        neighbors = y_train[np.argsort(distance)[:k]]
        predictions.append(Counter(neighbors.tolist()).most_common(1)[0][0])
    return np.asarray(predictions, dtype=np.int64)


def accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.mean(y_true == y_pred))


def main() -> None:
    checkpoint = load_checkpoint(BEST_MODEL_PATH)
    _, val_loader, splits = build_dataloaders()
    model = build_model()
    model.load_state_dict(checkpoint["model_state"])
    model.eval()
    pred_parts = []
    with torch.no_grad():
        for features, _targets in val_loader:
            pred_parts.append(model(features).argmax(dim=1).numpy())
    mlp_val_pred = np.concatenate(pred_parts)
    mlp_val_acc = accuracy(splits["y_val"], mlp_val_pred)

    knn_val_pred = knn_predict(splits["x_train"], splits["y_train"], splits["x_val"], KNN_K)
    knn_val_acc = accuracy(splits["y_val"], knn_val_pred)

    mlp_sub = read_submission(SUBMISSION_CSV)
    knn_sub = read_submission(BASELINE_SUBMISSION_CSV)
    shared_ids = [item for item in mlp_sub if item in knn_sub]
    agree = sum(mlp_sub[item] == knn_sub[item] for item in shared_ids)
    agree_rate = agree / len(shared_ids)

    lines = [
        f"mlp_val_acc={mlp_val_acc:.4f}",
        f"knn_val_acc={knn_val_acc:.4f}",
        f"test_prediction_agreement={agree_rate:.4f}",
        f"test_rows={len(shared_ids)}",
        f"knn_k={KNN_K}",
        "note=验证集指标使用同一分层划分；均值和标准差只由训练子集估计。kNN 测试提交仍由 baseline.py 在全部训练集上拟合。",
    ]
    COMPARE_REPORT.parent.mkdir(parents=True, exist_ok=True)
    COMPARE_REPORT.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))
    print(f"saved {COMPARE_REPORT}")


if __name__ == "__main__":
    main()
