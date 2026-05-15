"""Evaluate a trained checkpoint and produce the three required figures:
    1. Overall accuracy summary
    2. Accuracy vs SNR curve
    3. Confusion matrix (overall + per-SNR optional)

Usage:
    python evaluate.py --ckpt runs/cnn2_20260505-120000/best.pt

NOT: Bu surum checkpoint'teki 'best_params' veya 'args' dict'ini okuyup modeli
dogru hiperparametrelerle (dropout, reduction) insa eder. Tuned modellerde
default'tan farkli reduction kullanilabilir.
"""
import argparse
import json
from pathlib import Path

import numpy as np
import torch
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix

import config as C
from data_loader import get_dataloaders
from models import CNN2, CNN2_CBAM


@torch.no_grad()
def collect_predictions(model, loader, device):
    model.eval()
    ys, ps, snrs = [], [], []
    for x, y, snr in loader:
        x = x.to(device, non_blocking=True)
        logits = model(x)
        pred = logits.argmax(1).cpu().numpy()
        ys.append(y.numpy())
        ps.append(pred)
        snrs.append(snr.numpy())
    return (np.concatenate(ys), np.concatenate(ps), np.concatenate(snrs))


def plot_acc_vs_snr(snr_vals, acc_vals, out_path: Path, label: str = "model"):
    plt.figure(figsize=(8, 5))
    plt.plot(snr_vals, acc_vals, marker="o", label=label)
    plt.xlabel("SNR (dB)")
    plt.ylabel("Accuracy")
    plt.title("Classification Accuracy vs SNR")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


def plot_confusion(cm, classes, out_path: Path, title: str = "Confusion matrix"):
    cm_norm = cm.astype(np.float64)
    row_sum = cm_norm.sum(axis=1, keepdims=True)
    cm_norm = np.divide(cm_norm, row_sum, out=np.zeros_like(cm_norm), where=row_sum > 0)

    plt.figure(figsize=(9, 7))
    sns.heatmap(cm_norm, annot=True, fmt=".2f", cmap="Blues",
                xticklabels=classes, yticklabels=classes,
                cbar=True, square=True, annot_kws={"size": 7})
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title(title)
    plt.xticks(rotation=45, ha="right")
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


def build_model_from_ckpt(ckpt: dict):
    """Checkpoint'teki params'a gore modeli insa eder.

    Tuned modeller 'best_params' icerir (lr, dropout, weight_decay, reduction).
    Default egitilmis modeller 'args' icerir (model, epochs, dropout vs).
    """
    model_name = ckpt.get("model_name", "cnn2")

    # Hiperparametreleri ckpt'den oku, yoksa C'den default al
    if "best_params" in ckpt:
        bp = ckpt["best_params"]
        dropout = bp.get("dropout", C.DROPOUT)
        reduction = bp.get("reduction", 8)
        print(f"[eval] Tuned model detected. params={bp}")
    elif "args" in ckpt:
        args = ckpt["args"]
        if isinstance(args, dict):
            dropout = C.DROPOUT
        else:
            dropout = C.DROPOUT
        reduction = 8
    else:
        dropout = C.DROPOUT
        reduction = 8

    print(f"[eval] Model insa: dropout={dropout}, reduction={reduction}")

    if model_name == "cnn2":
        model = CNN2(num_classes=C.NUM_CLASSES, dropout=dropout)
    elif model_name in ("cnn2_cbam", "cbam"):
        model = CNN2_CBAM(num_classes=C.NUM_CLASSES, dropout=dropout, reduction=reduction)
    else:
        raise ValueError(f"Bilinmeyen model: {model_name}")

    return model, model_name


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", required=True, help="path to best.pt")
    parser.add_argument("--split", default="test", choices=["val", "test"])
    args = parser.parse_args()

    ckpt_path = Path(args.ckpt)
    ckpt = torch.load(ckpt_path, map_location=C.DEVICE, weights_only=False)
    model_name = ckpt.get("model_name", "cnn2")
    print(f"[eval] ckpt={ckpt_path}  model={model_name}  epoch={ckpt.get('epoch')}")

    model, model_name = build_model_from_ckpt(ckpt)
    model = model.to(C.DEVICE)
    model.load_state_dict(ckpt["state_dict"])

    train_dl, val_dl, test_dl, meta = get_dataloaders(num_workers=0)
    loader = test_dl if args.split == "test" else val_dl

    y_true, y_pred, snr = collect_predictions(model, loader, C.DEVICE)

    overall_acc = float((y_true == y_pred).mean())
    print(f"[eval] overall accuracy ({args.split}): {overall_acc:.4f}")

    snr_levels = sorted(np.unique(snr).tolist())
    per_snr = {}
    for s in snr_levels:
        mask = snr == s
        if mask.sum() == 0:
            continue
        acc = float((y_true[mask] == y_pred[mask]).mean())
        per_snr[int(s)] = acc
    print("[eval] accuracy vs SNR:")
    for s, a in per_snr.items():
        print(f"   SNR {s:+3d} dB  ->  {a:.4f}")

    cm = confusion_matrix(y_true, y_pred, labels=list(range(C.NUM_CLASSES)))

    out_dir = C.FIG_DIR / ckpt_path.parent.name
    out_dir.mkdir(parents=True, exist_ok=True)

    plot_acc_vs_snr(list(per_snr.keys()), list(per_snr.values()),
                    out_dir / "acc_vs_snr.png", label=model_name)
    plot_confusion(cm, C.MODULATIONS, out_dir / "confusion_overall.png",
                   title=f"{model_name} -- overall confusion ({args.split})")

    hi_mask = snr >= 0
    cm_hi = None
    if hi_mask.any():
        cm_hi = confusion_matrix(y_true[hi_mask], y_pred[hi_mask],
                                 labels=list(range(C.NUM_CLASSES)))
        plot_confusion(cm_hi, C.MODULATIONS,
                       out_dir / "confusion_highSNR.png",
                       title=f"{model_name} -- confusion (SNR >= 0 dB)")

    np.save(out_dir / "cm_overall.npy", cm)
    if cm_hi is not None:
        np.save(out_dir / "cm_highSNR.npy", cm_hi)

    summary = {
        "ckpt": str(ckpt_path),
        "model": model_name,
        "split": args.split,
        "overall_accuracy": overall_acc,
        "accuracy_vs_snr": per_snr,
    }
    with open(out_dir / "metrics.json", "w") as f:
        json.dump(summary, f, indent=2)

    print(f"[eval] figures + metrics saved to {out_dir}")


if __name__ == "__main__":
    main()
