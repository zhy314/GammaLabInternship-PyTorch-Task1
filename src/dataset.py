"""表格 Dataset、训练集统计量预处理与分层划分。"""
from __future__ import annotations

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset

from config import BATCH_SIZE, FEATURES, SEED, TEST_CSV, TRAIN_CSV, VAL_RATIO


class TabularDataset(Dataset):
    def __init__(self, features: np.ndarray, targets: np.ndarray | None = None):
        self.features = torch.as_tensor(np.asarray(features), dtype=torch.float32)
        self.targets = None if targets is None else torch.as_tensor(np.asarray(targets), dtype=torch.long)

    def __len__(self) -> int:
        return int(self.features.shape[0])

    def __getitem__(self, index: int):
        if self.targets is None:
            return self.features[index]
        return self.features[index], self.targets[index]


def load_frame(path) -> pd.DataFrame:
    return pd.read_csv(path, encoding="utf-8-sig")


def feature_matrix(frame: pd.DataFrame) -> np.ndarray:
    return frame[FEATURES].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=np.float64)


def stratified_split(y: np.ndarray, val_ratio: float, seed: int) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    train_parts: list[np.ndarray] = []
    val_parts: list[np.ndarray] = []
    for label in np.unique(y):
        idx = np.flatnonzero(y == label)
        rng.shuffle(idx)
        n_val = int(round(len(idx) * val_ratio))
        n_val = min(max(n_val, 1), len(idx) - 1)
        val_parts.append(idx[:n_val])
        train_parts.append(idx[n_val:])
    return np.concatenate(train_parts), np.concatenate(val_parts)


def fit_scaler(x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    means = np.nanmean(x, axis=0)
    filled = np.where(np.isnan(x), means, x)
    stds = filled.std(axis=0)
    stds = np.where(stds < 1e-8, 1.0, stds)
    return means.astype(np.float64), stds.astype(np.float64)


def apply_scaler(x: np.ndarray, means: np.ndarray, stds: np.ndarray) -> np.ndarray:
    filled = np.where(np.isnan(x), means, x)
    return ((filled - means) / stds).astype(np.float32)


def prepare_splits(val_ratio: float = VAL_RATIO, seed: int = SEED) -> dict:
    frame = load_frame(TRAIN_CSV)
    x = feature_matrix(frame)
    y = frame["target"].to_numpy(dtype=np.int64)
    ids = frame["id"].astype(str).to_numpy()
    train_idx, val_idx = stratified_split(y, val_ratio, seed)
    means, stds = fit_scaler(x[train_idx])
    return {
        "x_train": apply_scaler(x[train_idx], means, stds),
        "y_train": y[train_idx],
        "id_train": ids[train_idx],
        "x_val": apply_scaler(x[val_idx], means, stds),
        "y_val": y[val_idx],
        "id_val": ids[val_idx],
        "means": means,
        "stds": stds,
    }


def load_test(means: np.ndarray, stds: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    frame = load_frame(TEST_CSV)
    ids = frame["id"].astype(str).to_numpy()
    features = apply_scaler(feature_matrix(frame), means, stds)
    return ids, features


def make_loader(
    features: np.ndarray,
    targets: np.ndarray | None = None,
    shuffle: bool = False,
    batch_size: int = BATCH_SIZE,
    seed: int = SEED,
) -> DataLoader:
    generator = torch.Generator()
    generator.manual_seed(seed)
    return DataLoader(
        TabularDataset(features, targets),
        batch_size=batch_size,
        shuffle=shuffle,
        generator=generator,
    )


def build_dataloaders(
    val_ratio: float = VAL_RATIO,
    seed: int = SEED,
    batch_size: int = BATCH_SIZE,
) -> tuple[DataLoader, DataLoader, dict]:
    splits = prepare_splits(val_ratio, seed)
    train_loader = make_loader(splits["x_train"], splits["y_train"], shuffle=True, batch_size=batch_size, seed=seed)
    val_loader = make_loader(splits["x_val"], splits["y_val"], shuffle=False, batch_size=batch_size, seed=seed)
    return train_loader, val_loader, splits


if __name__ == "__main__":
    train_loader, val_loader, splits = build_dataloaders()
    xb, yb = next(iter(train_loader))
    print("train_batch", tuple(xb.shape), tuple(yb.shape))
    print("val_batch", tuple(next(iter(val_loader))[0].shape))
    print("train_size", len(splits["y_train"]), "val_size", len(splits["y_val"]))
    print("train_class_counts", np.bincount(splits["y_train"]).tolist())
    print("val_class_counts", np.bincount(splits["y_val"]).tolist())
    print("means", np.array2string(splits["means"], precision=6))
    print("stds", np.array2string(splits["stds"], precision=6))
