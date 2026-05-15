"""Load RadioML 2016.10A, preprocess, split (stratified by mod + SNR),
and expose PyTorch DataLoaders.

Usage:
    python data_loader.py          # one-time: builds processed/*.npy cache
    from data_loader import get_dataloaders
    train_dl, val_dl, test_dl, meta = get_dataloaders()
"""
import pickle
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split

import config as C


# ---------------------------------------------------------------------------
# Raw pickle -> flat numpy arrays
# ---------------------------------------------------------------------------
def load_raw():
    """Load RML2016.10a_dict.pkl and flatten into (X, y, snr) arrays.

    Returns:
        X   : float32, shape (N, 2, 128)
        y   : int64,   shape (N,)         -- modulation index 0..10
        snr : int64,   shape (N,)         -- SNR value in dB
    """
    if not C.RAW_PKL.exists():
        raise FileNotFoundError(
            f"Expected {C.RAW_PKL}. Download RML2016.10a_dict.pkl from "
            "https://www.deepsig.ai/datasets and put it in data/."
        )
    with open(C.RAW_PKL, "rb") as f:
        Xd = pickle.load(f, encoding="latin1")  # py2 pickle compat

    X_list, y_list, snr_list = [], [], []
    for (mod, snr), arr in Xd.items():
        n = arr.shape[0]
        X_list.append(arr.astype(np.float32))
        y_list.append(np.full(n, C.MOD_TO_IDX[mod], dtype=np.int64))
        snr_list.append(np.full(n, snr, dtype=np.int64))

    X = np.concatenate(X_list, axis=0)
    y = np.concatenate(y_list, axis=0)
    snr = np.concatenate(snr_list, axis=0)
    return X, y, snr


# ---------------------------------------------------------------------------
# Normalization
# ---------------------------------------------------------------------------
def normalize_per_sample(X: np.ndarray) -> np.ndarray:
    """Per-sample L2 normalization across the (2,128) tensor.
    Keeps relative I/Q magnitude meaningful but removes scale variance.
    """
    norm = np.sqrt(np.sum(X ** 2, axis=(1, 2), keepdims=True))
    norm = np.maximum(norm, 1e-8)
    return X / norm


# ---------------------------------------------------------------------------
# Stratified split (by joint mod x snr label)
# ---------------------------------------------------------------------------
def stratified_split(X, y, snr):
    """60/20/20 split, stratified on the joint (mod, snr) bucket so every
    subset has equal coverage of all 11 mods x 20 SNRs.
    """
    joint = y * 100 + (snr + 20)  # unique id per (mod, snr) cell

    X_trval, X_test, y_trval, y_test, s_trval, s_test = train_test_split(
        X, y, snr,
        test_size=C.TEST_RATIO,
        stratify=joint,
        random_state=C.SEED,
    )
    joint_trval = y_trval * 100 + (s_trval + 20)
    val_size = C.VAL_RATIO / (C.TRAIN_RATIO + C.VAL_RATIO)
    X_tr, X_val, y_tr, y_val, s_tr, s_val = train_test_split(
        X_trval, y_trval, s_trval,
        test_size=val_size,
        stratify=joint_trval,
        random_state=C.SEED,
    )
    return (X_tr, y_tr, s_tr), (X_val, y_val, s_val), (X_test, y_test, s_test)


# ---------------------------------------------------------------------------
# Build & cache
# ---------------------------------------------------------------------------
def build_processed():
    """Run once. Loads pickle, normalizes, splits, saves to processed/*.npy."""
    print("[data] loading raw pickle...")
    X, y, snr = load_raw()
    print(f"[data] raw shape: X={X.shape}, y={y.shape}, snr={snr.shape}")

    print("[data] normalizing (per-sample L2)...")
    X = normalize_per_sample(X)

    print("[data] stratified 60/20/20 split...")
    train, val, test = stratified_split(X, y, snr)

    for name, (Xs, ys, ss) in [("train", train), ("val", val), ("test", test)]:
        np.save(C.PROCESSED_DIR / f"X_{name}.npy", Xs)
        np.save(C.PROCESSED_DIR / f"y_{name}.npy", ys)
        np.save(C.PROCESSED_DIR / f"snr_{name}.npy", ss)
        print(f"[data] {name}: X={Xs.shape}, y={ys.shape}, snr={ss.shape}")

    print(f"[data] cached to {C.PROCESSED_DIR}")


# ---------------------------------------------------------------------------
# PyTorch Dataset
# ---------------------------------------------------------------------------
class RadioMLDataset(Dataset):
    def __init__(self, split: str):
        if split not in {"train", "val", "test"}:
            raise ValueError(split)
        self.X = np.load(C.PROCESSED_DIR / f"X_{split}.npy")
        self.y = np.load(C.PROCESSED_DIR / f"y_{split}.npy")
        self.snr = np.load(C.PROCESSED_DIR / f"snr_{split}.npy")

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, idx):
        x = torch.from_numpy(self.X[idx])           # (2, 128)
        y = int(self.y[idx])
        s = int(self.snr[idx])
        return x, y, s


def get_dataloaders(batch_size: int = None, num_workers: int = None):
    """Return train/val/test DataLoaders. Builds the cache if missing."""
    if not (C.PROCESSED_DIR / "X_train.npy").exists():
        build_processed()

    bs = batch_size or C.BATCH_SIZE
    nw = num_workers if num_workers is not None else C.NUM_WORKERS

    train_ds = RadioMLDataset("train")
    val_ds = RadioMLDataset("val")
    test_ds = RadioMLDataset("test")

    train_dl = DataLoader(train_ds, batch_size=bs, shuffle=True,
                          num_workers=nw, pin_memory=True, drop_last=True)
    val_dl = DataLoader(val_ds, batch_size=bs, shuffle=False,
                        num_workers=nw, pin_memory=True)
    test_dl = DataLoader(test_ds, batch_size=bs, shuffle=False,
                         num_workers=nw, pin_memory=True)
    meta = dict(num_classes=C.NUM_CLASSES,
                modulations=C.MODULATIONS,
                snr_range=C.SNR_RANGE)
    return train_dl, val_dl, test_dl, meta


if __name__ == "__main__":
    build_processed()
