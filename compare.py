"""Iki modelin yan yana karsilastirmasi.

Her iki modeli de iki yoldan kabul eder:
  1. Run dizini: runs/cnn2_xxx -> figures/cnn2_xxx altinda metrics.json, cm_*.npy arar
  2. Direkt figures dizini: figures/ensemble_top3 -> dogrudan metrics.json okur
     (ensemble veya benzeri non-run kayitlar icin)

Cikti: figures/comparison/ altina:
  - acc_vs_snr_overlay.png
  - confusion_side_by_side.png
  - qam_confusion_bar.png
  - summary.json
  - summary.md

Kullanim:
    python compare.py
        # otomatik en son cnn2 ve cnn2_cbam runlarini bulur

    python compare.py --baseline runs/cnn2_xxx --cbam runs/cnn2_cbam_yyy
        # spesifik run dizinleri

    python compare.py --baseline runs/cnn2_xxx --cbam_metrics figures/ensemble_top3
        # cbam tarafi direkt figures dizini (ensemble icin)

    python compare.py --baseline_metrics figures/X --cbam_metrics figures/Y
        # her ikisi de figures dizini
"""
import argparse
import json
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

import config as C


COMPARE_DIR = C.FIG_DIR / "comparison"


def find_latest_runs():
    """runs/ altindaki en son baseline ve cbam run'larini bul."""
    runs = sorted(Path(C.RUNS_DIR).glob("*"))
    baseline_runs = [r for r in runs if r.name.startswith("cnn2_") and "cbam" not in r.name]
    cbam_runs = [r for r in runs if "cbam" in r.name]
    baseline = baseline_runs[-1] if baseline_runs else None
    cbam = cbam_runs[-1] if cbam_runs else None
    return baseline, cbam


def load_metrics_from_run(run_dir: Path):
    """Run dizininden (figures/<run_name>) metrics + cm yukle."""
    fig_dir = C.FIG_DIR / run_dir.name
    metrics_path = fig_dir / "metrics.json"
    if not metrics_path.exists():
        raise FileNotFoundError(
            f"{metrics_path} bulunamadi. Once: python evaluate.py --ckpt {run_dir}/best.pt"
        )
    return _load_from_dir(fig_dir)


def load_metrics_from_dir(metrics_dir: Path):
    """Direkt bir figures dizininden metrics + cm yukle (ensemble icin)."""
    metrics_dir = Path(metrics_dir)
    metrics_path = metrics_dir / "metrics.json"
    if not metrics_path.exists():
        raise FileNotFoundError(f"{metrics_path} bulunamadi.")
    return _load_from_dir(metrics_dir)


def _load_from_dir(d: Path):
    metrics = json.loads((d / "metrics.json").read_text())
    cm_overall = np.load(d / "cm_overall.npy")
    cm_high_path = d / "cm_highSNR.npy"
    cm_high = np.load(cm_high_path) if cm_high_path.exists() else None
    return metrics, cm_overall, cm_high


def plot_acc_vs_snr_overlay(metrics_b, metrics_c, label_b: str, label_c: str, out_path: Path):
    snrs_b = sorted(int(k) for k in metrics_b["accuracy_vs_snr"].keys())
    accs_b = [metrics_b["accuracy_vs_snr"][str(s)] for s in snrs_b]
    snrs_c = sorted(int(k) for k in metrics_c["accuracy_vs_snr"].keys())
    accs_c = [metrics_c["accuracy_vs_snr"][str(s)] for s in snrs_c]

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 7), sharex=True,
                                    gridspec_kw={"height_ratios": [3, 1]})

    ax1.plot(snrs_b, accs_b, marker="o", linewidth=2, label=label_b, color="#1f77b4")
    ax1.plot(snrs_c, accs_c, marker="s", linewidth=2, label=label_c, color="#d62728")
    ax1.set_ylabel("Accuracy")
    ax1.set_title(f"Accuracy vs SNR -- {label_b} vs {label_c}")
    ax1.grid(True, alpha=0.3)
    ax1.legend(loc="lower right")
    ax1.set_ylim(0, 1.0)

    common_snrs = sorted(set(snrs_b) & set(snrs_c))
    deltas = [metrics_c["accuracy_vs_snr"][str(s)] - metrics_b["accuracy_vs_snr"][str(s)]
              for s in common_snrs]
    colors = ["#2ca02c" if d >= 0 else "#d62728" for d in deltas]
    ax2.bar(common_snrs, deltas, color=colors, alpha=0.7, width=1.5)
    ax2.axhline(0, color="black", linewidth=0.5)
    ax2.set_xlabel("SNR (dB)")
    ax2.set_ylabel(f"Δ accuracy")
    ax2.grid(True, alpha=0.3, axis="y")

    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"[compare] saved {out_path}")


def plot_confusion_side_by_side(cm_b, cm_c, label_b: str, label_c: str, out_path: Path,
                                title_suffix: str = "(SNR >= 0 dB)"):
    def normalize(cm):
        cm_n = cm.astype(np.float64)
        rs = cm_n.sum(axis=1, keepdims=True)
        return np.divide(cm_n, rs, out=np.zeros_like(cm_n), where=rs > 0)

    cm_b_n = normalize(cm_b)
    cm_c_n = normalize(cm_c)

    fig, axes = plt.subplots(1, 2, figsize=(18, 7))
    for ax, cm_n, title in [
        (axes[0], cm_b_n, f"{label_b} {title_suffix}"),
        (axes[1], cm_c_n, f"{label_c} {title_suffix}"),
    ]:
        sns.heatmap(cm_n, annot=True, fmt=".2f", cmap="Blues",
                    xticklabels=C.MODULATIONS, yticklabels=C.MODULATIONS,
                    cbar=True, square=True, annot_kws={"size": 7},
                    vmin=0, vmax=1, ax=ax)
        ax.set_xlabel("Predicted")
        ax.set_ylabel("True")
        ax.set_title(title)
        ax.tick_params(axis="x", rotation=45)
        ax.tick_params(axis="y", rotation=0)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"[compare] saved {out_path}")


def qam_confusion_stats(cm, classes=("QAM16", "QAM64")):
    i_q16 = C.MOD_TO_IDX[classes[0]]
    i_q64 = C.MOD_TO_IDX[classes[1]]
    cm_n = cm.astype(np.float64)
    rs = cm_n.sum(axis=1, keepdims=True)
    cm_n = np.divide(cm_n, rs, out=np.zeros_like(cm_n), where=rs > 0)
    return {
        "QAM16_as_QAM64": float(cm_n[i_q16, i_q64]),
        "QAM64_as_QAM16": float(cm_n[i_q64, i_q16]),
        "QAM16_correct":  float(cm_n[i_q16, i_q16]),
        "QAM64_correct":  float(cm_n[i_q64, i_q64]),
    }


def plot_qam_confusion_bar(qam_b, qam_c, label_b: str, label_c: str, out_path: Path):
    labels = ["QAM16→QAM64", "QAM64→QAM16", "QAM16 correct", "QAM64 correct"]
    keys = ["QAM16_as_QAM64", "QAM64_as_QAM16", "QAM16_correct", "QAM64_correct"]
    base_vals = [qam_b[k] for k in keys]
    cbam_vals = [qam_c[k] for k in keys]

    x = np.arange(len(labels))
    width = 0.35
    fig, ax = plt.subplots(figsize=(10, 5))
    bars_b = ax.bar(x - width/2, base_vals, width, label=label_b, color="#1f77b4")
    bars_c = ax.bar(x + width/2, cbam_vals, width, label=label_c, color="#d62728")

    for bars in (bars_b, bars_c):
        for bar in bars:
            h = bar.get_height()
            ax.annotate(f"{h:.2f}", xy=(bar.get_x() + bar.get_width() / 2, h),
                        xytext=(0, 3), textcoords="offset points",
                        ha="center", fontsize=8)

    ax.set_ylabel("Rate")
    ax.set_title("QAM16 ↔ QAM64 confusion (SNR >= 0 dB)")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.legend()
    ax.grid(True, alpha=0.3, axis="y")
    ax.set_ylim(0, 1.0)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"[compare] saved {out_path}")


def avg_acc_in_range(metrics, lo: int, hi: int) -> float:
    vals = [v for k, v in metrics["accuracy_vs_snr"].items() if lo <= int(k) <= hi]
    return float(np.mean(vals)) if vals else float("nan")


def write_summary(metrics_b, metrics_c, qam_b, qam_c, label_b, label_c,
                  source_b, source_c, out_dir):
    overall_b = metrics_b["overall_accuracy"]
    overall_c = metrics_c["overall_accuracy"]
    low_b = avg_acc_in_range(metrics_b, -10, 0)
    low_c = avg_acc_in_range(metrics_c, -10, 0)
    mid_b = avg_acc_in_range(metrics_b, 0, 18)
    mid_c = avg_acc_in_range(metrics_c, 0, 18)
    very_low_b = avg_acc_in_range(metrics_b, -20, -10)
    very_low_c = avg_acc_in_range(metrics_c, -20, -10)

    summary = {
        "baseline_label": label_b,
        "cbam_label": label_c,
        "baseline_source": str(source_b),
        "cbam_source": str(source_c),
        "overall_accuracy": {
            "baseline": overall_b,
            "cbam": overall_c,
            "delta_pp": (overall_c - overall_b) * 100,
        },
        "low_snr_avg_-10_to_0_dB": {
            "baseline": low_b,
            "cbam": low_c,
            "delta_pp": (low_c - low_b) * 100,
        },
        "high_snr_avg_0_to_18_dB": {
            "baseline": mid_b,
            "cbam": mid_c,
            "delta_pp": (mid_c - mid_b) * 100,
        },
        "very_low_snr_avg_-20_to_-10_dB": {
            "baseline": very_low_b,
            "cbam": very_low_c,
            "delta_pp": (very_low_c - very_low_b) * 100,
        },
        "qam_confusion_high_snr": {"baseline": qam_b, "cbam": qam_c},
        "per_snr_baseline": metrics_b["accuracy_vs_snr"],
        "per_snr_cbam": metrics_c["accuracy_vs_snr"],
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))

    md = []
    md.append("# Karsilastirma Ozeti\n")
    md.append(f"- {label_b} : `{source_b}`")
    md.append(f"- {label_c} : `{source_c}`\n")
    md.append("## Ana Metrikler\n")
    md.append(f"| Metrik | {label_b} | {label_c} | Δ (pp) |")
    md.append("|---|---:|---:|---:|")
    md.append(f"| Overall accuracy | {overall_b:.4f} | {overall_c:.4f} | {(overall_c-overall_b)*100:+.2f} |")
    md.append(f"| Avg acc (-20..-10 dB) | {very_low_b:.4f} | {very_low_c:.4f} | {(very_low_c-very_low_b)*100:+.2f} |")
    md.append(f"| Avg acc (-10..0 dB)   | {low_b:.4f} | {low_c:.4f} | {(low_c-low_b)*100:+.2f} |")
    md.append(f"| Avg acc (0..18 dB)    | {mid_b:.4f} | {mid_c:.4f} | {(mid_c-mid_b)*100:+.2f} |\n")
    md.append("## QAM16 ↔ QAM64 Confusion (SNR >= 0 dB)\n")
    md.append(f"| Olay | {label_b} | {label_c} |")
    md.append("|---|---:|---:|")
    md.append(f"| QAM16 dogru tahmin | {qam_b['QAM16_correct']:.4f} | {qam_c['QAM16_correct']:.4f} |")
    md.append(f"| QAM64 dogru tahmin | {qam_b['QAM64_correct']:.4f} | {qam_c['QAM64_correct']:.4f} |")
    md.append(f"| QAM16 → QAM64 hata | {qam_b['QAM16_as_QAM64']:.4f} | {qam_c['QAM16_as_QAM64']:.4f} |")
    md.append(f"| QAM64 → QAM16 hata | {qam_b['QAM64_as_QAM16']:.4f} | {qam_c['QAM64_as_QAM16']:.4f} |\n")
    (out_dir / "summary.md").write_text("\n".join(md))

    print("\n" + "=" * 60)
    print(f"[compare] OZET: {label_b} vs {label_c}")
    print(f"  Overall:                {label_b}={overall_b:.4f}  {label_c}={overall_c:.4f}  Δ={(overall_c-overall_b)*100:+.2f} pp")
    print(f"  Avg acc (-10..0 dB):    {label_b}={low_b:.4f}  {label_c}={low_c:.4f}  Δ={(low_c-low_b)*100:+.2f} pp")
    print(f"  Avg acc (0..18 dB):     {label_b}={mid_b:.4f}  {label_c}={mid_c:.4f}  Δ={(mid_c-mid_b)*100:+.2f} pp")
    print(f"  QAM16->QAM64 confusion: {label_b}={qam_b['QAM16_as_QAM64']:.4f}  {label_c}={qam_c['QAM16_as_QAM64']:.4f}")
    print(f"  QAM64->QAM16 confusion: {label_b}={qam_b['QAM64_as_QAM16']:.4f}  {label_c}={qam_c['QAM64_as_QAM16']:.4f}")
    print("=" * 60)


def resolve_source(run_arg, metrics_arg, default_search_for: str):
    """run_dir veya metrics_dir argumantini cozumle, metrics yukle, label belirle."""
    if metrics_arg is not None:
        d = Path(metrics_arg)
        metrics, cm_all, cm_hi = load_metrics_from_dir(d)
        # Metrics icindeki 'model' alanindan label cikarmaya calis
        label = metrics.get("model", d.name)
        return metrics, cm_all, cm_hi, label, d
    elif run_arg is not None:
        run_dir = Path(run_arg)
        metrics, cm_all, cm_hi = load_metrics_from_run(run_dir)
        # Run isminden label tahmin et
        if "ensemble" in run_dir.name.lower():
            label = "Ensemble"
        elif "tuned" in run_dir.name.lower() or "top1" in run_dir.name.lower() or "top2" in run_dir.name.lower() or "top3" in run_dir.name.lower():
            label = "CNN2+CBAM (tuned)"
        elif "cbam" in run_dir.name.lower():
            label = "CNN2+CBAM"
        else:
            label = "CNN2 (baseline)"
        return metrics, cm_all, cm_hi, label, run_dir
    else:
        return None, None, None, None, None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline", default=None, help="baseline run dir")
    parser.add_argument("--cbam", default=None, help="cbam run dir")
    parser.add_argument("--baseline_metrics", default=None,
                        help="baseline icin direkt figures dizini (run yerine)")
    parser.add_argument("--cbam_metrics", default=None,
                        help="cbam icin direkt figures dizini (ensemble icin)")
    parser.add_argument("--baseline_label", default=None, help="ozel label")
    parser.add_argument("--cbam_label", default=None, help="ozel label")
    parser.add_argument("--out_subdir", default=None,
                        help="Cikti alt-klasoru (default: comparison)")
    args = parser.parse_args()

    # Eger her ikisi de None ise otomatik bul
    if (args.baseline is None and args.baseline_metrics is None
            and args.cbam is None and args.cbam_metrics is None):
        bd, cd = find_latest_runs()
        if bd is None or cd is None:
            raise SystemExit("Otomatik run bulunamadi, --baseline ve --cbam belirt.")
        args.baseline = str(bd)
        args.cbam = str(cd)

    # Resolve
    metrics_b, cm_b_all, cm_b_hi, label_b, source_b = resolve_source(
        args.baseline, args.baseline_metrics, "baseline"
    )
    metrics_c, cm_c_all, cm_c_hi, label_c, source_c = resolve_source(
        args.cbam, args.cbam_metrics, "cbam"
    )

    if metrics_b is None or metrics_c is None:
        raise SystemExit("Karsilastirma icin iki kaynak da gerekli.")

    # Ozel label override
    if args.baseline_label:
        label_b = args.baseline_label
    if args.cbam_label:
        label_c = args.cbam_label

    print(f"[compare] BASELINE: {label_b}  <- {source_b}")
    print(f"[compare] CBAM    : {label_c}  <- {source_c}")

    # Cikti dizini
    out_dir = COMPARE_DIR if args.out_subdir is None else (C.FIG_DIR / args.out_subdir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Figures
    plot_acc_vs_snr_overlay(metrics_b, metrics_c, label_b, label_c,
                            out_dir / "acc_vs_snr_overlay.png")

    if cm_b_hi is not None and cm_c_hi is not None:
        plot_confusion_side_by_side(cm_b_hi, cm_c_hi, label_b, label_c,
                                    out_dir / "confusion_side_by_side.png",
                                    title_suffix="(SNR >= 0 dB)")
        qam_b = qam_confusion_stats(cm_b_hi)
        qam_c = qam_confusion_stats(cm_c_hi)
    else:
        plot_confusion_side_by_side(cm_b_all, cm_c_all, label_b, label_c,
                                    out_dir / "confusion_side_by_side.png",
                                    title_suffix="(overall)")
        qam_b = qam_confusion_stats(cm_b_all)
        qam_c = qam_confusion_stats(cm_c_all)

    plot_qam_confusion_bar(qam_b, qam_c, label_b, label_c,
                           out_dir / "qam_confusion_bar.png")

    write_summary(metrics_b, metrics_c, qam_b, qam_c, label_b, label_c,
                  source_b, source_c, out_dir)

    print(f"\n[compare] tum ciktilar -> {out_dir}")


if __name__ == "__main__":
    main()
