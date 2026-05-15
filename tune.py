"""Hiperparametre optimizasyonu (Optuna, Bayesian / TPE).

Sadece cnn2_cbam modelini optimize eder.
Objective: SNR >= 0 dB validation accuracy (proposal'la uyumlu).

Akis:
  1. Optuna study basla (SQLite'e persist eder, kopusa dayanikli)
  2. Her trial: kisa egitim (8 epoch) + median pruning
  3. 15 trial sonra: en iyi config'le 30 epoch tam egitim

Kullanim:
    python tune.py                            # 15 trial default
    python tune.py --n_trials 10              # daha az trial
    python tune.py --resume                   # onceki study'ye devam et
    python tune.py --skip_final               # final egitimi atla, sadece arama yap
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
from optuna.pruners import MedianPruner
from optuna.samplers import TPESampler

import config as C
from data_loader import get_dataloaders
from models import CNN2_CBAM


STUDY_DIR = C.PROJECT_ROOT / "optuna_studies"
STUDY_DIR.mkdir(parents=True, exist_ok=True)
STUDY_NAME = "cnn2_cbam_search"
DB_PATH = f"sqlite:///{STUDY_DIR}/study.db"


def set_seed(seed: int = C.SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# ---------------------------------------------------------------------------
# Egitim & degerlendirme yardimcilari (AMP'li)
# ---------------------------------------------------------------------------
def train_one_epoch(model, loader, optimizer, criterion, device, scaler):
    model.train()
    for x, y, _snr in loader:
        x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        with torch.amp.autocast(device_type="cuda", dtype=torch.float16, enabled=True):
            logits = model(x)
            loss = criterion(logits, y)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()


@torch.no_grad()
def eval_high_snr_acc(model, loader, device):
    """Validation set'inde SNR >= 0 dB icin dogruluk hesapla."""
    model.eval()
    correct, total = 0, 0
    for x, y, snr in loader:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        snr = snr.to(device, non_blocking=True)
        mask = snr >= 0
        if mask.sum() == 0:
            continue
        with torch.amp.autocast(device_type="cuda", dtype=torch.float16, enabled=True):
            logits = model(x)
        pred = logits.argmax(1)
        correct += (pred[mask] == y[mask]).sum().item()
        total += mask.sum().item()
    return correct / total if total > 0 else 0.0


# ---------------------------------------------------------------------------
# Optuna objective
# ---------------------------------------------------------------------------
def objective(trial: optuna.Trial, train_dl, val_dl, device, n_epochs: int = 8):
    # Search space
    lr = trial.suggest_float("lr", 1e-4, 5e-3, log=True)
    dropout = trial.suggest_categorical("dropout", [0.3, 0.4, 0.5, 0.6])
    weight_decay = trial.suggest_categorical("weight_decay", [1e-5, 1e-4, 1e-3])
    reduction = trial.suggest_categorical("reduction", [4, 8, 16])

    set_seed()
    model = CNN2_CBAM(num_classes=C.NUM_CLASSES, dropout=dropout, reduction=reduction).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    criterion = nn.CrossEntropyLoss()
    scaler = torch.amp.GradScaler("cuda", enabled=True)

    best_high_snr_acc = 0.0
    for epoch in range(1, n_epochs + 1):
        t0 = time.time()
        train_one_epoch(model, train_dl, optimizer, criterion, device, scaler)
        high_acc = eval_high_snr_acc(model, val_dl, device)
        dt = time.time() - t0
        best_high_snr_acc = max(best_high_snr_acc, high_acc)

        print(f"  trial#{trial.number} epoch {epoch}/{n_epochs} | {dt:5.1f}s | val_high_snr_acc={high_acc:.4f}")

        # Pruning: epoch'a gore intermediate value rapor et
        trial.report(high_acc, epoch)
        if trial.should_prune():
            print(f"  trial#{trial.number} PRUNED at epoch {epoch}")
            raise optuna.TrialPruned()

    return best_high_snr_acc


# ---------------------------------------------------------------------------
# Final egitim (en iyi config'le, train.py mantigi)
# ---------------------------------------------------------------------------
def train_final(best_params: dict, n_epochs: int = 30):
    """En iyi hiperparametrelerle 30 epoch tam egitim yap, ckpt kaydet."""
    print("\n" + "=" * 60)
    print("[tune] FINAL TRAINING with best params:")
    for k, v in best_params.items():
        print(f"  {k} = {v}")
    print("=" * 60)

    set_seed()
    device = C.DEVICE
    train_dl, val_dl, test_dl, _ = get_dataloaders(num_workers=0)

    model = CNN2_CBAM(
        num_classes=C.NUM_CLASSES,
        dropout=best_params["dropout"],
        reduction=best_params["reduction"],
    ).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"[tune] model params = {n_params:,}")

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=best_params["lr"],
        weight_decay=best_params["weight_decay"],
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=3)
    criterion = nn.CrossEntropyLoss()
    scaler = torch.amp.GradScaler("cuda", enabled=True)

    stamp = time.strftime("%Y%m%d-%H%M%S")
    run_name = f"cnn2_cbam_{stamp}_tuned"
    run_dir = C.RUNS_DIR / run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = run_dir / "best.pt"

    best_val_loss = float("inf")
    best_epoch = -1
    epochs_without_improve = 0
    PATIENCE = 10

    for epoch in range(1, n_epochs + 1):
        t0 = time.time()
        # train
        model.train()
        tr_loss_sum, tr_correct, tr_total = 0.0, 0, 0
        for x, y, _snr in train_dl:
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
        tr_loss, tr_acc = tr_loss_sum / tr_total, tr_correct / tr_total

        # val
        model.eval()
        v_loss_sum, v_correct, v_total = 0.0, 0, 0
        with torch.no_grad():
            for x, y, _snr in val_dl:
                x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
                with torch.amp.autocast(device_type="cuda", dtype=torch.float16, enabled=True):
                    logits = model(x)
                    loss = criterion(logits, y)
                v_loss_sum += loss.item() * x.size(0)
                v_correct += (logits.argmax(1) == y).sum().item()
                v_total += x.size(0)
        val_loss, val_acc = v_loss_sum / v_total, v_correct / v_total
        scheduler.step(val_loss)
        dt = time.time() - t0

        improved = val_loss < best_val_loss
        marker = "  *" if improved else ""
        print(f"epoch {epoch:3d}/{n_epochs} | {dt:5.1f}s | tr_loss={tr_loss:.4f} tr_acc={tr_acc:.4f} | val_loss={val_loss:.4f} val_acc={val_acc:.4f}{marker}")

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
                "best_params": best_params,
            }, ckpt_path)
        else:
            epochs_without_improve += 1
            if epochs_without_improve >= PATIENCE:
                print(f"[tune] early stop at epoch {epoch}")
                break

    # Test
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    t_correct, t_total = 0, 0
    with torch.no_grad():
        for x, y, _snr in test_dl:
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            logits = model(x)
            t_correct += (logits.argmax(1) == y).sum().item()
            t_total += x.size(0)
    test_acc = t_correct / t_total
    print(f"[tune] FINAL TEST accuracy = {test_acc:.4f}")
    print(f"[tune] checkpoint: {ckpt_path}")
    return run_dir, test_acc


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_trials", type=int, default=15, help="# of Optuna trials")
    parser.add_argument("--epochs_per_trial", type=int, default=8)
    parser.add_argument("--final_epochs", type=int, default=30)
    parser.add_argument("--resume", action="store_true",
                        help="Onceki study'ye devam et (SQLite kayitli)")
    parser.add_argument("--skip_final", action="store_true",
                        help="Final 30-epoch egitimi atla")
    args = parser.parse_args()

    device = C.DEVICE
    print(f"[tune] device={device}")
    print(f"[tune] n_trials={args.n_trials}  epochs/trial={args.epochs_per_trial}")
    print(f"[tune] study DB: {DB_PATH}")

    # DataLoader'lari bir kere yukle (her trial yeniden yuklemesin)
    set_seed()
    train_dl, val_dl, _, _ = get_dataloaders(num_workers=0)
    print(f"[tune] train={len(train_dl.dataset)}  val={len(val_dl.dataset)}")

    # Study (SQLite'de persist - kopusa dayanikli)
    sampler = TPESampler(seed=C.SEED)
    pruner = MedianPruner(n_startup_trials=3, n_warmup_steps=2)
    study = optuna.create_study(
        study_name=STUDY_NAME,
        storage=DB_PATH,
        direction="maximize",
        sampler=sampler,
        pruner=pruner,
        load_if_exists=args.resume,
    )

    n_done = len([t for t in study.trials if t.state in (
        optuna.trial.TrialState.COMPLETE,
        optuna.trial.TrialState.PRUNED,
    )])
    n_remaining = max(0, args.n_trials - n_done)
    print(f"[tune] previously completed/pruned trials: {n_done}")
    print(f"[tune] running {n_remaining} more trials...")

    if n_remaining > 0:
        study.optimize(
            lambda t: objective(t, train_dl, val_dl, device, n_epochs=args.epochs_per_trial),
            n_trials=n_remaining,
            show_progress_bar=False,
        )

    print("\n" + "=" * 60)
    print("[tune] STUDY OZET:")
    print(f"  Toplam trial: {len(study.trials)}")
    print(f"  Tamamlanan  : {len([t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE])}")
    print(f"  Pruned      : {len([t for t in study.trials if t.state == optuna.trial.TrialState.PRUNED])}")
    print(f"  Failed      : {len([t for t in study.trials if t.state == optuna.trial.TrialState.FAIL])}")
    print(f"  En iyi value (val high-SNR acc): {study.best_value:.4f}")
    print(f"  En iyi params:")
    for k, v in study.best_params.items():
        print(f"    {k} = {v}")
    print("=" * 60)

    # Sonuclari JSON'a kaydet
    results = {
        "study_name": STUDY_NAME,
        "n_trials": len(study.trials),
        "n_complete": len([t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]),
        "n_pruned": len([t for t in study.trials if t.state == optuna.trial.TrialState.PRUNED]),
        "best_value": study.best_value,
        "best_params": study.best_params,
        "all_trials": [
            {
                "number": t.number,
                "state": t.state.name,
                "value": t.value,
                "params": t.params,
            }
            for t in study.trials
        ],
    }
    out_json = STUDY_DIR / "results.json"
    out_json.write_text(json.dumps(results, indent=2))
    print(f"[tune] sonuclar -> {out_json}")

    # Final egitim
    if not args.skip_final:
        run_dir, test_acc = train_final(study.best_params, n_epochs=args.final_epochs)
        results["final_run_dir"] = str(run_dir)
        results["final_test_acc"] = test_acc
        out_json.write_text(json.dumps(results, indent=2))
        print(f"\n[tune] Final test acc: {test_acc:.4f}")
        print(f"[tune] Final ckpt: {run_dir}/best.pt")
        print(f"[tune] Su komutla degerlendir:")
        print(f"   python evaluate.py --ckpt {run_dir}/best.pt")


if __name__ == "__main__":
    main()
