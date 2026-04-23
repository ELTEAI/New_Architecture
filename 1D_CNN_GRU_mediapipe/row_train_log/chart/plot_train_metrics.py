import os
import re
from typing import Dict, List

import matplotlib

# Headless server-friendly backend
matplotlib.use("Agg")
import matplotlib.pyplot as plt


LOG_PATH = "/home/ubuntu/New_Architecture/1D_CNN_GRU/train_log/train_20260325_073638.log"
OUT_DIR = "/home/ubuntu/New_Architecture/1D_CNN_GRU/train_log/chart"


def parse_log_metrics(log_path: str) -> Dict[str, List[float]]:
    pattern = re.compile(
        r"Epoch\s*\[\s*(\d+)/\d+\].*?"
        r"Train Loss:\s*([0-9]*\.?[0-9]+),\s*Acc:\s*([0-9]*\.?[0-9]+)\s*\|\|\s*"
        r"Val Loss:\s*([0-9]*\.?[0-9]+),\s*Acc:\s*([0-9]*\.?[0-9]+),\s*"
        r"Precision:\s*([0-9]*\.?[0-9]+),\s*Recall:\s*([0-9]*\.?[0-9]+),\s*F1:\s*([0-9]*\.?[0-9]+)"
    )

    metrics = {
        "epoch": [],
        "train_loss": [],
        "train_acc": [],
        "val_loss": [],
        "val_acc": [],
        "val_precision": [],
        "val_recall": [],
        "val_f1": [],
    }

    with open(log_path, "r", encoding="utf-8") as f:
        for line in f:
            m = pattern.search(line)
            if not m:
                continue
            metrics["epoch"].append(int(m.group(1)))
            metrics["train_loss"].append(float(m.group(2)))
            metrics["train_acc"].append(float(m.group(3)))
            metrics["val_loss"].append(float(m.group(4)))
            metrics["val_acc"].append(float(m.group(5)))
            metrics["val_precision"].append(float(m.group(6)))
            metrics["val_recall"].append(float(m.group(7)))
            metrics["val_f1"].append(float(m.group(8)))

    if not metrics["epoch"]:
        raise ValueError(f"未在日志中解析到 epoch 指标: {log_path}")

    return metrics


def plot_single_metric(
    x: List[int], y: List[float], title: str, y_label: str, out_path: str, color: str
) -> None:
    plt.figure(figsize=(10, 5), dpi=130)
    plt.plot(x, y, color=color, linewidth=2.0)
    plt.scatter(x, y, color=color, s=10)
    plt.title(title)
    plt.xlabel("Epoch")
    plt.ylabel(y_label)
    plt.grid(True, linestyle="--", alpha=0.35)
    plt.tight_layout()
    plt.savefig(out_path, bbox_inches="tight")
    plt.close()


def plot_all_metrics(metrics: Dict[str, List[float]], out_path: str) -> None:
    epochs = metrics["epoch"]
    fig, ax1 = plt.subplots(figsize=(12, 6), dpi=130)

    # Left y-axis: losses
    l1 = ax1.plot(epochs, metrics["train_loss"], color="#e74c3c", linewidth=2, label="Train Loss")
    l2 = ax1.plot(epochs, metrics["val_loss"], color="#d35400", linewidth=2, label="Val Loss")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss")
    ax1.grid(True, linestyle="--", alpha=0.3)

    # Right y-axis: accuracy/precision/recall/f1
    ax2 = ax1.twinx()
    l3 = ax2.plot(epochs, metrics["train_acc"], color="#3498db", linewidth=1.8, label="Train Acc")
    l4 = ax2.plot(epochs, metrics["val_acc"], color="#2980b9", linewidth=1.8, label="Val Acc")
    l5 = ax2.plot(epochs, metrics["val_precision"], color="#2ecc71", linewidth=1.8, label="Val Precision")
    l6 = ax2.plot(epochs, metrics["val_recall"], color="#16a085", linewidth=1.8, label="Val Recall")
    l7 = ax2.plot(epochs, metrics["val_f1"], color="#8e44ad", linewidth=2.2, label="Val F1")
    ax2.set_ylabel("Score (0-1)")
    ax2.set_ylim(0, 1.05)

    lines = l1 + l2 + l3 + l4 + l5 + l6 + l7
    labels = [line.get_label() for line in lines]
    ax1.legend(lines, labels, loc="center right", fontsize=9)

    plt.title("Training Metrics Overview")
    plt.tight_layout()
    plt.savefig(out_path, bbox_inches="tight")
    plt.close()


def main() -> None:
    os.makedirs(OUT_DIR, exist_ok=True)
    metrics = parse_log_metrics(LOG_PATH)
    epochs = metrics["epoch"]

    plot_single_metric(
        epochs,
        metrics["train_loss"],
        "Train Loss vs Epoch",
        "Train Loss",
        os.path.join(OUT_DIR, "train_loss.png"),
        "#e74c3c",
    )
    plot_single_metric(
        epochs,
        metrics["train_acc"],
        "Train Accuracy vs Epoch",
        "Train Accuracy",
        os.path.join(OUT_DIR, "train_acc.png"),
        "#3498db",
    )
    plot_single_metric(
        epochs,
        metrics["val_loss"],
        "Validation Loss vs Epoch",
        "Val Loss",
        os.path.join(OUT_DIR, "val_loss.png"),
        "#d35400",
    )
    plot_single_metric(
        epochs,
        metrics["val_acc"],
        "Validation Accuracy vs Epoch",
        "Val Accuracy",
        os.path.join(OUT_DIR, "val_acc.png"),
        "#2980b9",
    )
    plot_single_metric(
        epochs,
        metrics["val_precision"],
        "Validation Precision (Macro) vs Epoch",
        "Val Precision",
        os.path.join(OUT_DIR, "val_precision.png"),
        "#2ecc71",
    )
    plot_single_metric(
        epochs,
        metrics["val_recall"],
        "Validation Recall (Macro) vs Epoch",
        "Val Recall",
        os.path.join(OUT_DIR, "val_recall.png"),
        "#16a085",
    )
    plot_single_metric(
        epochs,
        metrics["val_f1"],
        "Validation F1 (Macro) vs Epoch",
        "Val F1",
        os.path.join(OUT_DIR, "val_f1.png"),
        "#8e44ad",
    )

    plot_all_metrics(metrics, os.path.join(OUT_DIR, "all_metrics_overview.png"))
    print(f"图表已生成，共 8 张，输出目录: {OUT_DIR}")


if __name__ == "__main__":
    main()

