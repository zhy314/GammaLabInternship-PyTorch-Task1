"""加载最优模型，对测试集生成 submission.csv。"""
from __future__ import annotations

import csv

import torch

from config import BEST_MODEL_PATH, SUBMISSION_CSV
from src.dataset import load_test, make_loader
from src.model import build_model
from src.utils import load_checkpoint


def main() -> None:
    checkpoint = load_checkpoint(BEST_MODEL_PATH)
    model = build_model()
    model.load_state_dict(checkpoint["model_state"])
    model.eval()

    ids, features = load_test(checkpoint["means"], checkpoint["stds"])
    loader = make_loader(features, shuffle=False)
    predictions: list[int] = []
    with torch.no_grad():
        for batch in loader:
            predictions.extend(model(batch).argmax(dim=1).tolist())

    SUBMISSION_CSV.parent.mkdir(parents=True, exist_ok=True)
    with SUBMISSION_CSV.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["id", "prediction"])
        writer.writerows(zip(ids, predictions))
    print(f"saved {SUBMISSION_CSV} rows={len(predictions)}")


if __name__ == "__main__":
    main()
