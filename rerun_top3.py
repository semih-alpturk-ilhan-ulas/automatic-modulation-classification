"""Optuna top-3 trial'i alip her birini 30 epoch tam egitim ile yeniden kosar.

Amac: Optuna'nin proxy bias'ini (8 epoch -> 30 epoch farki) bypass etmek.
Top-3 trial'in her biri AMP'li tam egitime sokulur, en iyisi secilir.

Cikti:
  runs/cnn2_cbam_<timestamp>_top1/best.pt   (her trial icin ayri klasor)
  runs/cnn2_cbam_<timestamp>_top2/best.pt
  runs/cnn2_cbam_<timestamp>_top3/best.pt
  optuna_studies/top3_results.json          (3 modelin test acc'lari)

Kullanim:
    python rerun_top3.py                  # default 30 epoch
    python rerun_top3.py --epochs 25      # daha az
"""
import argparse
import json
import random
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import optuna

import config as C
from data_loader import get_dataloaders
from models import CNN2_CBAM


STUDY_DIR = C.PROJECT_ROOT / "optuna_studies"
DB_PATH = f"sqlite:///{STUDY_DIR}/study.db"


def set_seed(seed: int = C.SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_top_n_configs(n: int = 3):
    """Tamamlanmis trial'lar arasindan en yuksek value'ya sahip n tanesini bul."""
    study = optuna.load_study(study_name="cnn2_cbam_search", storage=DB_PATH)
    completed = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
    if len(completed) < n:
        print(f"[WARN] Sadece {len(completed)} tamamlanmis trial var, n={n} istendi")
        n = len(completed)
    completed.sort(key=lambda t: t.value, reverse=True)
    top_n = completed[:n]
    return [(t.number, t.value, t.params) for t in top_n]


def train_one_config(rank: int, trial_num: int, params: dict, train_dl, val_dl, test_dl,
                     device, n_epochs: int = 30, patience: int = 10):
    """Tek bir hiperparametre kombinasyonuyla 30 epoch tam egitim."""
    set_seed()
    model = CNN2_CBAM(
        num_classes=C.NUM_CLASSES,
        dropout=params["dropout"],
        reduction=params["reduction"],
    ).to(device)
    n_params = sum(p.numel() for p in model.parameters())

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=params["lr"],
        weight_decay=params["weight_decay"],
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=3
    )
    criterion = nn.CrossEntropyLoss()
    scaler = torch.amp.GradScaler("cuda", enabled=True)

    stamp = time.strftime("%Y%m%d-%H%M%S")
    run_name = f"cnn2_cbam_{stamp}_top{rank}_trial{trial_num}"
    run_dir = C.RUNS_DIR / run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = run_dir / "best.pt"

    print(f"\n{'=' * 60}")
    print(f"[top{rank}] Trial #{trial_num} | params={params}")
    print(f"[top{rank}] params={n_params:,} | run_dir={run_name}")
    print(f"{'=' * 60}")

    best_val_loss = float("inf")
    best_epoch = -1
    epochs_without_improve = 0

    for epoch in range(1, n_epochs + 1):
        t0 = time.time()
        # Train
        model.train()
        tr_loss_sum, tr_correct, tr_total = 0.0, 0, 0
        for x, y, _ in train_dl:
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast(device_type="cuda", dtype=torch.float16, enabled=True):
                logits = model(x)
                loss = criterion(logits, y)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            tr_loss_sum += loss.item() * x.size(0)
            tr_correct += (logits.argmax(1) == y).sum().item()
            tr_total += x.size(0)
        tr_loss = tr_loss_sum / tr_total
        tr_acc = tr_correct / tr_total

        # Val
        model.eval()
        v_loss_sum, v_correct, v_total = 0.0, 0, 0
        with torch.no_grad():
            for x, y, _ in val_dl:
                x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
                with torch.amp.autocast(device_type="cuda", dtype=torch.float16, enabled=True):
                    logits = model(x)
                    loss = criterion(logits, y)
                v_loss_sum += loss.item() * x.size(0)
                v_correct += (logits.argmax(1) == y).sum().item()
                v_total += x.size(0)
        val_loss = v_loss_sum / v_total
        val_acc = v_correct / v_total
        scheduler.step(val_loss)
        dt = time.time() - t0

        improved = val_loss < best_val_loss
        marker = "  *" if improved else ""
        print(f"  top{rank} epoch {epoch:3d}/{n_epochs} | {dt:5.1f}s | tr_loss={tr_loss:.4f} tr_acc={tr_acc:.4f} | val_loss={val_loss:.4f} val_acc={val_acc:.4f}{marker}")

        if improved:
            best_val_loss = val_loss
            best_epoch = epoch
            epochs_without_improve = 0
            torch.save({
                "epoch": epoch,
                "model_name": "cnn2_cbam",
                "state_dict": model.state_dict(),
                "val_loss": val_loss,
                "val_acc": val_acc,
                "best_params": params,
                "trial_num": trial_num,
                "rank": rank,
            }, ckpt_path)
        else:
            epochs_without_improve += 1
            if epochs_without_improve >= patience:
                print(f"[top{rank}] early stop at epoch {epoch}")
                break

    # Test
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    t_correct, t_total = 0, 0
    with torch.no_grad():
        for x, y, _ in test_dl:
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            logits = model(x)
            t_correct += (logits.argmax(1) == y).sum().item()
            t_total += x.size(0)
    test_acc = t_correct / t_total
    print(f"[top{rank}] FINAL TEST acc = {test_acc:.4f}  (best epoch={best_epoch})")

    return {
        "rank": rank,
        "trial_num": trial_num,
        "params": params,
        "run_dir": str(run_dir),
        "ckpt_path": str(ckpt_path),
        "best_epoch": best_epoch,
        "best_val_loss": best_val_loss,
        "test_acc": test_acc,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--n", type=int, default=3, help="kac trial'i rerun et")
    args = parser.parse_args()

    device = C.DEVICE
    print(f"[rerun] device={device}")
    print(f"[rerun] epochs={args.epochs}  patience={args.patience}")

    # Top-N configs
    top = get_top_n_configs(args.n)
    print(f"\n[rerun] Optuna top-{len(top)} trials:")
    for rank, (num, val, params) in enumerate(top, 1):
        print(f"  top{rank}: trial#{num}  val_high_snr_acc={val:.4f}  params={params}")

    # Data
    set_seed()
    train_dl, val_dl, test_dl, _ = get_dataloaders(num_workers=0)
    print(f"[rerun] train={len(train_dl.dataset)}  val={len(val_dl.dataset)}  test={len(test_dl.dataset)}")

    # Run each
    results = []
    for rank, (trial_num, _, params) in enumerate(top, 1):
        res = train_one_config(rank, trial_num, params,
                               train_dl, val_dl, test_dl,
                               device, n_epochs=args.epochs, patience=args.patience)
        results.append(res)

    # Sort by test acc
    results.sort(key=lambda r: r["test_acc"], reverse=True)

    # Save summary
    summary_path = STUDY_DIR / "top3_results.json"
    summary_path.write_text(json.dumps(results, indent=2))

    # Print
    print("\n" + "=" * 60)
    print("[rerun] TOP-3 OZET (test acc sirasiyla):")
    print("=" * 60)
    for i, r in enumerate(results, 1):
        marker = " <-- BEST" if i == 1 else ""
        print(f"  #{i}  trial#{r['trial_num']}  test_acc={r['test_acc']:.4f}  best_epoch={r['best_epoch']}{marker}")
        print(f"      lr={r['params']['lr']:.6f}  dropout={r['params']['dropout']}  "
              f"wd={r['params']['weight_decay']}  reduction={r['params']['reduction']}")
    print("=" * 60)

    # Best vs default reference
    print(f"\n[rerun] En iyi konfigurasyon: {results[0]['ckpt_path']}")
    print(f"[rerun] Bunu evaluate etmek icin:")
    print(f"   python evaluate.py --ckpt {results[0]['ckpt_path']}")
    print(f"\n[rerun] Default CBAM ile karsilastirmak icin (CBAM degiskenini Drive'daki run'a ayarla):")
    print(f"   python compare.py --baseline runs/cnn2_cbam_<DEFAULT_TIMESTAMP>_cbam_v1 --cbam {results[0]['run_dir']}")


if __name__ == "__main__":
    main()
