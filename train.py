"""Training loop with early stopping, TensorBoard logging, and AMP (mixed precision)."""
import argparse
import time
import random

import numpy as np
import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter

import config as C
from data_loader import get_dataloaders
from models import build_model


def set_seed(seed: int = C.SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def evaluate(model, loader, criterion, device, use_amp: bool = False):
    model.eval()
    total, correct, loss_sum = 0, 0, 0.0
    amp_enabled = use_amp and device.type == "cuda"
    with torch.no_grad():
        for x, y, _snr in loader:
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            with torch.amp.autocast(device_type="cuda", dtype=torch.float16, enabled=amp_enabled):
                logits = model(x)
                loss = criterion(logits, y)
            loss_sum += loss.item() * x.size(0)
            correct += (logits.argmax(1) == y).sum().item()
            total += x.size(0)
    return loss_sum / total, correct / total


def train_one_epoch(model, loader, optimizer, criterion, device, scaler, use_amp: bool = False):
    model.train()
    total, correct, loss_sum = 0, 0, 0.0
    amp_enabled = use_amp and device.type == "cuda"
    for x, y, _snr in loader:
        x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        with torch.amp.autocast(device_type="cuda", dtype=torch.float16, enabled=amp_enabled):
            logits = model(x)
            loss = criterion(logits, y)
        if amp_enabled:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()
        loss_sum += loss.item() * x.size(0)
        correct += (logits.argmax(1) == y).sum().item()
        total += x.size(0)
    return loss_sum / total, correct / total


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="cnn2", choices=["cnn2", "cnn2_cbam"])
    parser.add_argument("--epochs", type=int, default=C.NUM_EPOCHS)
    parser.add_argument("--batch_size", type=int, default=C.BATCH_SIZE)
    parser.add_argument("--lr", type=float, default=C.LEARNING_RATE)
    parser.add_argument("--weight_decay", type=float, default=C.WEIGHT_DECAY)
    parser.add_argument("--patience", type=int, default=C.EARLY_STOP_PATIENCE)
    parser.add_argument("--num_workers", type=int, default=C.NUM_WORKERS)
    parser.add_argument("--tag", default="")
    parser.add_argument("--amp", action="store_true", default=True,
                        help="Enable Automatic Mixed Precision (default: on)")
    parser.add_argument("--no_amp", dest="amp", action="store_false",
                        help="Disable AMP (use full FP32)")
    args = parser.parse_args()

    set_seed()
    device = C.DEVICE
    use_amp = args.amp and device.type == "cuda"
    print(f"[train] device={device}  model={args.model}  amp={use_amp}")

    stamp = time.strftime("%Y%m%d-%H%M%S")
    run_name = f"{args.model}_{stamp}{('_' + args.tag) if args.tag else ''}"
    run_dir = C.RUNS_DIR / run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    writer = SummaryWriter(run_dir / "tb")

    train_dl, val_dl, test_dl, meta = get_dataloaders(
        batch_size=args.batch_size, num_workers=args.num_workers)
    print(f"[train] train={len(train_dl.dataset)}  val={len(val_dl.dataset)}  test={len(test_dl.dataset)}")

    model = build_model(args.model).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"[train] params={n_params:,}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=3)
    criterion = nn.CrossEntropyLoss()
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    best_val_loss = float("inf")
    best_epoch = -1
    epochs_without_improve = 0
    ckpt_path = run_dir / "best.pt"

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        tr_loss, tr_acc = train_one_epoch(model, train_dl, optimizer, criterion, device, scaler, use_amp)
        val_loss, val_acc = evaluate(model, val_dl, criterion, device, use_amp)
        scheduler.step(val_loss)
        dt = time.time() - t0

        writer.add_scalar("loss/train", tr_loss, epoch)
        writer.add_scalar("loss/val", val_loss, epoch)
        writer.add_scalar("acc/train", tr_acc, epoch)
        writer.add_scalar("acc/val", val_acc, epoch)
        writer.add_scalar("lr", optimizer.param_groups[0]["lr"], epoch)

        improved = val_loss < best_val_loss
        marker = "  *" if improved else ""
        print(f"epoch {epoch:3d}/{args.epochs} | {dt:5.1f}s | tr_loss={tr_loss:.4f} tr_acc={tr_acc:.4f} | val_loss={val_loss:.4f} val_acc={val_acc:.4f}{marker}")

        if improved:
            best_val_loss = val_loss
            best_epoch = epoch
            epochs_without_improve = 0
            torch.save({
                "epoch": epoch,
                "model_name": args.model,
                "state_dict": model.state_dict(),
                "val_loss": val_loss,
                "val_acc": val_acc,
                "args": vars(args),
            }, ckpt_path)
        else:
            epochs_without_improve += 1
            if epochs_without_improve >= args.patience:
                print(f"[train] early stop at epoch {epoch} (best={best_epoch}, val_loss={best_val_loss:.4f})")
                break

    print(f"[train] loading best ckpt from epoch {best_epoch}...")
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["state_dict"])
    test_loss, test_acc = evaluate(model, test_dl, criterion, device, use_amp)
    print(f"[train] TEST  loss={test_loss:.4f}  acc={test_acc:.4f}")
    writer.add_scalar("loss/test", test_loss, best_epoch)
    writer.add_scalar("acc/test", test_acc, best_epoch)
    writer.close()

    print(f"[train] checkpoint: {ckpt_path}")


if __name__ == "__main__":
    main()
