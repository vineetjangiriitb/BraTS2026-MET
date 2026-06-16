#!/usr/bin/env python3
"""
Plot the nnU-Net learning curve from the training log, for the presentation.

Run LOCALLY after download_results.sh, pointing at the downloaded log:
    python scripts/plot_training_curve.py runpod_results/logs/02_train_fold0.log

nnU-Net logs one line per epoch. We parse train loss, validation (pseudo-Dice),
and epoch time, then plot loss + pseudo-Dice vs epoch on twin axes.
"""
import re
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# nnU-Net v2 prints lines like:
#   train_loss -0.1234
#   val_loss -0.2345
#   Pseudo dice [0.8123, 0.7654, 0.6543]
#   Epoch 12
EPOCH_RE = re.compile(r"^Epoch (\d+)")
TRAIN_RE = re.compile(r"train_loss\s+(-?\d+\.\d+)")
VAL_RE = re.compile(r"val_loss\s+(-?\d+\.\d+)")
DICE_RE = re.compile(r"Pseudo dice\s+\[([^\]]+)\]")


def parse(log_path):
    epochs, train_loss, val_loss, mean_dice = [], [], [], []
    cur = {}
    for line in Path(log_path).read_text().splitlines():
        if m := EPOCH_RE.search(line):
            if cur.get("epoch") is not None:
                epochs.append(cur["epoch"])
                train_loss.append(cur.get("train"))
                val_loss.append(cur.get("val"))
                mean_dice.append(cur.get("dice"))
            cur = {"epoch": int(m.group(1))}
        if m := TRAIN_RE.search(line):
            cur["train"] = float(m.group(1))
        if m := VAL_RE.search(line):
            cur["val"] = float(m.group(1))
        if m := DICE_RE.search(line):
            vals = [float(x) for x in m.group(1).split(",")]
            cur["dice"] = sum(vals) / len(vals)
    if cur.get("epoch") is not None:
        epochs.append(cur["epoch"])
        train_loss.append(cur.get("train"))
        val_loss.append(cur.get("val"))
        mean_dice.append(cur.get("dice"))
    return epochs, train_loss, val_loss, mean_dice


def main():
    log_path = sys.argv[1] if len(sys.argv) > 1 else "runpod_results/logs/02_train_fold0.log"
    epochs, train_loss, val_loss, mean_dice = parse(log_path)
    if not epochs:
        raise SystemExit(f"No epoch lines parsed from {log_path}")

    fig, ax1 = plt.subplots(figsize=(10, 6))
    ax1.plot(epochs, train_loss, label="train loss", color="tab:blue")
    ax1.plot(epochs, val_loss, label="val loss", color="tab:cyan", linestyle="--")
    ax1.set_xlabel("epoch")
    ax1.set_ylabel("loss (CE + Dice; negative is better)")
    ax1.legend(loc="upper left")

    ax2 = ax1.twinx()
    ax2.plot(epochs, mean_dice, label="mean pseudo-Dice", color="tab:red")
    ax2.set_ylabel("mean pseudo-Dice (val)")
    ax2.set_ylim(0, 1)
    ax2.legend(loc="lower right")

    plt.title("nnU-Net 3d_fullres — BraTS-MET 100-case run (fold 0)")
    out = Path(log_path).parent / "training_curve.png"
    plt.tight_layout()
    plt.savefig(out, dpi=130)
    print(f"Saved {out}")
    if mean_dice and mean_dice[-1] is not None:
        print(f"Final mean pseudo-Dice: {mean_dice[-1]:.4f}")


if __name__ == "__main__":
    main()
