"""EDA / visualization. Run after data_loader.py has built the cache.

Outputs go to figures/eda/.
"""
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

import config as C


OUT_DIR = C.FIG_DIR / "eda"
OUT_DIR.mkdir(parents=True, exist_ok=True)


def load_split(split: str = "train"):
    X = np.load(C.PROCESSED_DIR / f"X_{split}.npy")
    y = np.load(C.PROCESSED_DIR / f"y_{split}.npy")
    snr = np.load(C.PROCESSED_DIR / f"snr_{split}.npy")
    return X, y, snr


def plot_constellation_grid(X, y, snr, target_snr: int = 18,
                            samples_per_class: int = 200):
    """One subplot per modulation, scatter of I vs Q at the chosen SNR."""
    fig, axes = plt.subplots(3, 4, figsize=(14, 10))
    axes = axes.flatten()
    for i, mod in enumerate(C.MODULATIONS):
        ax = axes[i]
        mask = (y == i) & (snr == target_snr)
        idx = np.where(mask)[0][:samples_per_class]
        if len(idx) == 0:
            ax.set_title(f"{mod} (no samples)")
            continue
        I = X[idx, 0, :].flatten()
        Q = X[idx, 1, :].flatten()
        ax.scatter(I, Q, s=2, alpha=0.4)
        ax.set_title(f"{mod}  (SNR={target_snr} dB)")
        ax.set_xlabel("I")
        ax.set_ylabel("Q")
        ax.grid(True, alpha=0.3)
        ax.set_aspect("equal", adjustable="box")
    # hide unused subplot (we have 11 mods, 12 axes)
    for j in range(len(C.MODULATIONS), len(axes)):
        axes[j].set_visible(False)
    plt.tight_layout()
    out = OUT_DIR / f"constellations_snr{target_snr}.png"
    plt.savefig(out, dpi=150)
    plt.close()
    print(f"[viz] saved {out}")


def plot_snr_progression(X, y, snr, mod_name: str = "QAM16",
                         snr_levels=(-10, 0, 10, 18),
                         samples_per_panel: int = 200):
    """Same modulation, four SNR levels side by side. Shows noise impact."""
    mod_idx = C.MOD_TO_IDX[mod_name]
    fig, axes = plt.subplots(1, len(snr_levels), figsize=(4 * len(snr_levels), 4))
    for ax, s in zip(axes, snr_levels):
        mask = (y == mod_idx) & (snr == s)
        idx = np.where(mask)[0][:samples_per_panel]
        if len(idx) == 0:
            ax.set_title(f"{mod_name}  SNR={s} dB (n=0)")
            continue
        I = X[idx, 0, :].flatten()
        Q = X[idx, 1, :].flatten()
        ax.scatter(I, Q, s=2, alpha=0.4)
        ax.set_title(f"{mod_name}  SNR={s} dB")
        ax.set_xlabel("I"); ax.set_ylabel("Q")
        ax.grid(True, alpha=0.3)
        ax.set_aspect("equal", adjustable="box")
    plt.tight_layout()
    out = OUT_DIR / f"snr_progression_{mod_name}.png"
    plt.savefig(out, dpi=150)
    plt.close()
    print(f"[viz] saved {out}")


def plot_iq_timeseries(X, y, snr, target_snr: int = 18):
    """Raw I/Q vs time for one example of each modulation."""
    fig, axes = plt.subplots(len(C.MODULATIONS), 1,
                             figsize=(10, 1.6 * len(C.MODULATIONS)),
                             sharex=True)
    t = np.arange(128)
    for i, mod in enumerate(C.MODULATIONS):
        ax = axes[i]
        mask = (y == i) & (snr == target_snr)
        idx = np.where(mask)[0]
        if len(idx) == 0:
            ax.set_title(f"{mod} (no samples)")
            continue
        sample = X[idx[0]]
        ax.plot(t, sample[0], label="I", linewidth=0.9)
        ax.plot(t, sample[1], label="Q", linewidth=0.9, alpha=0.7)
        ax.set_ylabel(mod, rotation=0, labelpad=30, ha="right")
        ax.grid(True, alpha=0.3)
        if i == 0:
            ax.legend(loc="upper right", fontsize=8)
    axes[-1].set_xlabel("sample index")
    plt.suptitle(f"Raw I/Q time series at SNR={target_snr} dB")
    plt.tight_layout()
    out = OUT_DIR / f"iq_timeseries_snr{target_snr}.png"
    plt.savefig(out, dpi=150)
    plt.close()
    print(f"[viz] saved {out}")


def plot_class_balance(y, snr):
    """Bar chart: samples per (mod, snr) -- sanity check that splits are balanced."""
    counts = np.zeros((C.NUM_CLASSES, len(C.SNR_RANGE)), dtype=int)
    snr_to_col = {s: j for j, s in enumerate(C.SNR_RANGE)}
    for i, s in zip(y, snr):
        if s in snr_to_col:
            counts[i, snr_to_col[s]] += 1
    fig, ax = plt.subplots(figsize=(12, 6))
    im = ax.imshow(counts, aspect="auto", cmap="viridis")
    ax.set_yticks(range(C.NUM_CLASSES))
    ax.set_yticklabels(C.MODULATIONS)
    ax.set_xticks(range(len(C.SNR_RANGE)))
    ax.set_xticklabels(C.SNR_RANGE, rotation=45)
    ax.set_xlabel("SNR (dB)")
    ax.set_title("Sample count per (modulation, SNR) -- training split")
    plt.colorbar(im, ax=ax, label="#samples")
    plt.tight_layout()
    out = OUT_DIR / "class_balance.png"
    plt.savefig(out, dpi=150)
    plt.close()
    print(f"[viz] saved {out}")


def main():
    print("[viz] loading training split...")
    X, y, snr = load_split("train")
    print(f"[viz] X={X.shape}, y={y.shape}, snr={snr.shape}")

    plot_class_balance(y, snr)
    plot_constellation_grid(X, y, snr, target_snr=18)
    plot_constellation_grid(X, y, snr, target_snr=0)
    plot_snr_progression(X, y, snr, mod_name="QAM16")
    plot_snr_progression(X, y, snr, mod_name="QAM64")
    plot_snr_progression(X, y, snr, mod_name="QPSK")
    plot_iq_timeseries(X, y, snr, target_snr=18)


if __name__ == "__main__":
    main()
