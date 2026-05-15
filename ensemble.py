"""Top-3 model ensemble.

Optuna top-3 rerun'da elde edilen 3 modeli yukler ve logit-averaging ile
ensemble tahmini yapar. Egitim YOK, sadece inference.

Cikti:
  figures/ensemble_top3/
    acc_vs_snr.png
    confusion_overall.png
    confusion_highSNR.png
    cm_overall.npy
    cm_highSNR.npy
    metrics.json

Kullanim:
    python ensemble.py
        # optuna_studies/top3_results.json'dan 3 modeli otomatik bulur

    python ensemble.py --ckpts run1/best.pt run2/best.pt run3/best.pt
        # spesifik checkpoint'ler

    python ensemble.py --weights 0.5 0.3 0.2
        # agirlikli ortalama (default: esit agirlik)
"""
import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix

import config as C
from data_loader import get_dataloaders
from models import CNN2_CBAM


STUDY_DIR = C.PROJECT_ROOT / "optuna_studies"
ENSEMBLE_DIR = C.FIG_DIR / "ensemble_top3"


def build_model_from_ckpt(ckpt: dict):
    """Checkpoint'teki params'a gore CNN2_CBAM insa eder."""
    if "best_params" in ckpt:
        bp = ckpt["best_params"]
        dropout = bp.get("dropout", C.DROPOUT)
        reduction = bp.get("reduction", 8)
    else:
        dropout = C.DROPOUT
        reduction = 8
    model = CNN2_CBAM(num_classes=C.NUM_CLASSES, dropout=dropout, reduction=reduction)
    return model, dropout, reduction


def load_top3_ckpts():
    """top3_results.json'dan 3 ckpt yolunu cikar."""
    results_path = STUDY_DIR / "top3_results.json"
    if not results_path.exists():
        raise FileNotFoundError(
            f"{results_path} bulunamadi. Once: python rerun_top3.py"
        )
    results = json.loads(results_path.read_text())
    ckpts = [Path(r["ckpt_path"]) for r in results]
    info = [(r["rank"], r["trial_num"], r["test_acc"], r["params"]) for r in results]
    return ckpts, info


@torch.no_grad()
def collect_ensemble_predictions(models, loader, device, weights=None):
    """Tum modelleri loader uzerinde calistir, logit'leri ortala, tahmin et.

    Returns: y_true, y_pred_ensemble, snr, individual_correct (her model icin)
    """
    n_models = len(models)
    if weights is None:
        weights = [1.0 / n_models] * n_models
    weights = np.array(weights, dtype=np.float32)
    weights = weights / weights.sum()  # normalize

    for m in models:
        m.eval()

    ys, ps_ensemble, snrs = [], [], []
    individual_correct = [0] * n_models
    total = 0

    for x, y, snr in loader:
        x = x.to(device, non_blocking=True)
        y_np = y.numpy()

        # Her modelin softmax probabilitilerini topla (weighted)
        probs_sum = None
        for i, m in enumerate(models):
            logits = m(x)
            probs = F.softmax(logits, dim=1)
            if probs_sum is None:
                probs_sum = weights[i] * probs
            else:
                probs_sum = probs_sum + weights[i] * probs

            # Bireysel model accuracy izle
            ind_pred = logits.argmax(1).cpu().numpy()
            individual_correct[i] += (ind_pred == y_np).sum()

        pred_ensemble = probs_sum.argmax(1).cpu().numpy()
        ys.append(y_np)
        ps_ensemble.append(pred_ensemble)
        snrs.append(snr.numpy())
        total += x.size(0)

    individual_acc = [c / total for c in individual_correct]
    return (
        np.concatenate(ys),
        np.concatenate(ps_ensemble),
        np.concatenate(snrs),
        individual_acc,
    )


def plot_acc_vs_snr(snr_vals, acc_vals, out_path: Path, label: str = "ensemble"):
    plt.figure(figsize=(8, 5))
    plt.plot(snr_vals, acc_vals, marker="o", linewidth=2, label=label, color="#9467bd")
    plt.xlabel("SNR (dB)")
    plt.ylabel("Accuracy")
    plt.title("Ensemble Classification Accuracy vs SNR")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


def plot_confusion(cm, classes, out_path: Path, title: str):
    cm_norm = cm.astype(np.float64)
    rs = cm_norm.sum(axis=1, keepdims=True)
    cm_norm = np.divide(cm_norm, rs, out=np.zeros_like(cm_norm), where=rs > 0)

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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpts", nargs="+", default=None,
                        help="Spesifik ckpt yollari (default: top3_results.json'dan)")
    parser.add_argument("--weights", nargs="+", type=float, default=None,
                        help="Model agirliklari (default: esit)")
    parser.add_argument("--split", default="test", choices=["val", "test"])
    args = parser.parse_args()

    device = C.DEVICE
    print(f"[ensemble] device={device}  split={args.split}")

    # Ckpt yollarini topla
    if args.ckpts:
        ckpt_paths = [Path(p) for p in args.ckpts]
        info = None
    else:
        ckpt_paths, info = load_top3_ckpts()
        print("[ensemble] Top-3 trial'larin bireysel test acc'lari:")
        for rank, trial_num, test_acc, params in info:
            print(f"  top{rank}  trial#{trial_num}  acc={test_acc:.4f}  params={params}")

    # Modelleri yukle
    print(f"\n[ensemble] {len(ckpt_paths)} model yukleniyor...")
    models = []
    for cp in ckpt_paths:
        ckpt = torch.load(cp, map_location=device, weights_only=False)
        model, dropout, reduction = build_model_from_ckpt(ckpt)
        model.load_state_dict(ckpt["state_dict"])
        model = model.to(device)
        models.append(model)
        print(f"  loaded: {cp.parent.name}  (dropout={dropout}, reduction={reduction})")

    # Data
    train_dl, val_dl, test_dl, _ = get_dataloaders(num_workers=0)
    loader = test_dl if args.split == "test" else val_dl
    print(f"[ensemble] {args.split} samples: {len(loader.dataset)}")

    # Ensemble inference
    print(f"\n[ensemble] inference (weights={args.weights or 'esit'})...")
    y_true, y_pred, snr, individual_accs = collect_ensemble_predictions(
        models, loader, device, weights=args.weights
    )

    # Bireysel modellerin (sanity check) accuracy'leri
    print("\n[ensemble] BIREYSEL model accuracy'leri (sanity check):")
    for i, acc in enumerate(individual_accs, 1):
        print(f"  model #{i}  acc={acc:.4f}")

    # Ensemble overall
    overall_acc = float((y_true == y_pred).mean())
    print(f"\n[ensemble] ENSEMBLE overall accuracy ({args.split}): {overall_acc:.4f}")

    # En iyi bireysel ile karsilastir
    best_individual = max(individual_accs)
    delta = (overall_acc - best_individual) * 100
    print(f"[ensemble] En iyi bireysel: {best_individual:.4f}")
    print(f"[ensemble] Ensemble - en iyi bireysel: {delta:+.2f} pp")

    # Per-SNR
    snr_levels = sorted(np.unique(snr).tolist())
    per_snr = {}
    for s in snr_levels:
        mask = snr == s
        if mask.sum() == 0:
            continue
        acc = float((y_true[mask] == y_pred[mask]).mean())
        per_snr[int(s)] = acc
    print("\n[ensemble] accuracy vs SNR:")
    for s, a in per_snr.items():
        print(f"   SNR {s:+3d} dB  ->  {a:.4f}")

    # Confusion matrices
    cm = confusion_matrix(y_true, y_pred, labels=list(range(C.NUM_CLASSES)))
    hi_mask = snr >= 0
    cm_hi = None
    if hi_mask.any():
        cm_hi = confusion_matrix(y_true[hi_mask], y_pred[hi_mask],
                                 labels=list(range(C.NUM_CLASSES)))

    # Save
    ENSEMBLE_DIR.mkdir(parents=True, exist_ok=True)
    plot_acc_vs_snr(list(per_snr.keys()), list(per_snr.values()),
                    ENSEMBLE_DIR / "acc_vs_snr.png", label="Top-3 Ensemble")
    plot_confusion(cm, C.MODULATIONS,
                   ENSEMBLE_DIR / "confusion_overall.png",
                   title=f"Top-3 Ensemble -- overall confusion ({args.split})")
    if cm_hi is not None:
        plot_confusion(cm_hi, C.MODULATIONS,
                       ENSEMBLE_DIR / "confusion_highSNR.png",
                       title=f"Top-3 Ensemble -- confusion (SNR >= 0 dB)")
        np.save(ENSEMBLE_DIR / "cm_highSNR.npy", cm_hi)
    np.save(ENSEMBLE_DIR / "cm_overall.npy", cm)

    summary = {
        "ckpt_paths": [str(p) for p in ckpt_paths],
        "weights": args.weights or [1.0 / len(ckpt_paths)] * len(ckpt_paths),
        "split": args.split,
        "model": "top3_ensemble",
        "individual_accuracies": individual_accs,
        "best_individual_accuracy": best_individual,
        "overall_accuracy": overall_acc,
        "ensemble_gain_pp": delta,
        "accuracy_vs_snr": per_snr,
    }
    (ENSEMBLE_DIR / "metrics.json").write_text(json.dumps(summary, indent=2))

    print(f"\n[ensemble] cikti -> {ENSEMBLE_DIR}")
    print("=" * 60)
    print("[ensemble] OZET:")
    print(f"  Bireysel en iyi : {best_individual:.4f}")
    print(f"  ENSEMBLE        : {overall_acc:.4f}")
    print(f"  Kazanc          : {delta:+.2f} pp")
    print("=" * 60)


if __name__ == "__main__":
    main()
