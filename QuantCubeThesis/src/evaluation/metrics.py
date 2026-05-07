"""
Evaluation Metrics
==================
Comprehensive evaluation for the FOMC sentiment classifier,
including per-class metrics, calibration analysis, and
distributional quality checks.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    classification_report,
    confusion_matrix,
    log_loss,
)
from scipy.stats import entropy as scipy_entropy


def full_classification_report(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_probs: np.ndarray,
    label_names: List[str],
    output_dir: Optional[str] = None,
) -> Dict:
    """
    Generate a comprehensive classification report.

    Args:
        y_true: Ground truth labels (integer IDs).
        y_pred: Predicted labels (integer IDs).
        y_probs: Predicted probabilities (n_samples x n_classes).
        label_names: List of label names (ordered by ID).
        output_dir: If provided, save plots here.

    Returns:
        Dict with all metrics.
    """
    results = {
        "accuracy": accuracy_score(y_true, y_pred),
        "f1_macro": f1_score(y_true, y_pred, average="macro", zero_division=0),
        "f1_weighted": f1_score(y_true, y_pred, average="weighted", zero_division=0),
        "precision_macro": precision_score(y_true, y_pred, average="macro", zero_division=0),
        "recall_macro": recall_score(y_true, y_pred, average="macro", zero_division=0),
        "log_loss": log_loss(y_true, y_probs, labels=list(range(len(label_names)))),
    }

    # Per-class report
    report = classification_report(
        y_true, y_pred,
        target_names=label_names,
        output_dict=True,
        zero_division=0,
    )
    results["per_class"] = report

    # Print summary
    print("\n=== Classification Report ===")
    print(classification_report(y_true, y_pred, target_names=label_names, zero_division=0))
    print(f"Log Loss: {results['log_loss']:.4f}")

    if output_dir:
        _plot_confusion_matrix(y_true, y_pred, label_names, output_dir)
        _plot_calibration(y_true, y_probs, label_names, output_dir)
        _plot_entropy_distribution(y_probs, y_true, y_pred, output_dir)

    return results


def _plot_confusion_matrix(
    y_true, y_pred, label_names, output_dir
):
    """Plot and save confusion matrix."""
    cm = confusion_matrix(y_true, y_pred)
    cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    sns.heatmap(cm, annot=True, fmt="d", xticklabels=label_names,
                yticklabels=label_names, cmap="Blues", ax=ax1)
    ax1.set_title("Confusion Matrix (Counts)")
    ax1.set_xlabel("Predicted")
    ax1.set_ylabel("True")

    sns.heatmap(cm_norm, annot=True, fmt=".2f", xticklabels=label_names,
                yticklabels=label_names, cmap="Blues", ax=ax2)
    ax2.set_title("Confusion Matrix (Normalized)")
    ax2.set_xlabel("Predicted")
    ax2.set_ylabel("True")

    plt.tight_layout()
    path = Path(output_dir) / "confusion_matrix.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved confusion matrix to {path}")


def _plot_calibration(
    y_true, y_probs, label_names, output_dir
):
    """Plot reliability diagram (calibration curve) per class."""
    n_classes = len(label_names)
    n_bins = 10

    fig, axes = plt.subplots(1, min(n_classes, 5), figsize=(4 * min(n_classes, 5), 4))
    if n_classes == 1:
        axes = [axes]

    for cls_idx in range(min(n_classes, 5)):
        ax = axes[cls_idx]
        binary_true = (y_true == cls_idx).astype(int)
        cls_probs = y_probs[:, cls_idx]

        bin_edges = np.linspace(0, 1, n_bins + 1)
        bin_means = []
        bin_trues = []

        for i in range(n_bins):
            mask = (cls_probs >= bin_edges[i]) & (cls_probs < bin_edges[i + 1])
            if mask.sum() > 0:
                bin_means.append(cls_probs[mask].mean())
                bin_trues.append(binary_true[mask].mean())

        ax.plot([0, 1], [0, 1], "k--", alpha=0.5, label="Perfect")
        ax.plot(bin_means, bin_trues, "o-", label="Model")
        ax.set_title(label_names[cls_idx], fontsize=10)
        ax.set_xlabel("Mean predicted prob")
        ax.set_ylabel("Fraction positive")
        ax.legend(fontsize=8)

    plt.suptitle("Calibration (Reliability Diagrams)", fontsize=12)
    plt.tight_layout()
    path = Path(output_dir) / "calibration.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved calibration plot to {path}")


def _plot_entropy_distribution(
    y_probs, y_true, y_pred, output_dir
):
    """Plot entropy distribution for correct vs incorrect predictions."""
    entropies = scipy_entropy(y_probs, axis=1)
    correct = y_true == y_pred

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.hist(entropies[correct], bins=30, alpha=0.6, label="Correct", density=True)
    ax.hist(entropies[~correct], bins=30, alpha=0.6, label="Incorrect", density=True)
    ax.set_xlabel("Prediction Entropy")
    ax.set_ylabel("Density")
    ax.set_title("Entropy Distribution: Correct vs Incorrect Predictions")
    ax.legend()

    plt.tight_layout()
    path = Path(output_dir) / "entropy_distribution.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved entropy distribution to {path}")


def compare_baseline_vs_finetuned(
    baseline_results: Dict,
    finetuned_results: Dict,
) -> pd.DataFrame:
    """Create a comparison table between baseline and fine-tuned models."""
    metrics = ["accuracy", "f1_macro", "f1_weighted", "log_loss"]

    rows = []
    for m in metrics:
        rows.append({
            "metric": m,
            "baseline": baseline_results.get(m, float("nan")),
            "finetuned": finetuned_results.get(m, float("nan")),
            "improvement": (
                finetuned_results.get(m, 0) - baseline_results.get(m, 0)
            ),
        })

    df = pd.DataFrame(rows)
    print("\n=== Baseline vs Fine-Tuned ===")
    print(df.to_string(index=False))
    return df
