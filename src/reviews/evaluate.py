"""Scores with bootstrap intervals, and the run metadata saved next to every result."""

import json
import subprocess
from pathlib import Path

import numpy as np
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score

N_BOOT = 1000


def scores(y_true, y_pred, y_score=None) -> dict:
    out = {
        "accuracy": accuracy_score(y_true, y_pred),
        # Negative reviews are the complaints a team acts on, so their F1 is reported on its own.
        "f1_negative": f1_score(y_true, y_pred, pos_label=0, zero_division=0),
    }
    if y_score is not None and len(np.unique(y_true)) == 2:
        out["auc"] = roc_auc_score(y_true, y_score)
    return out


def bootstrap(y_true, y_pred, n: int = N_BOOT, seed: int = 0) -> dict:
    """95% percentile intervals for accuracy and negative-class F1, resampling reviews."""
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    if len(y_true) == 0:
        raise ValueError("no rows to score")
    rng = np.random.default_rng(seed)
    acc, f1 = np.empty(n), np.empty(n)
    for i in range(n):
        idx = rng.integers(0, len(y_true), len(y_true))
        t, p = y_true[idx], y_pred[idx]
        acc[i] = (t == p).mean()
        tp = np.sum((t == 0) & (p == 0))
        denom = np.sum(t == 0) + np.sum(p == 0)
        f1[i] = 2 * tp / denom if denom else 0.0
    return {
        "accuracy_95": np.percentile(acc, [2.5, 97.5]).round(4).tolist(),
        "f1_negative_95": np.percentile(f1, [2.5, 97.5]).round(4).tolist(),
    }


def paired_difference(y_true, pred_a, pred_b, n: int = N_BOOT, seed: int = 0) -> list[float]:
    """95% interval for accuracy(a) - accuracy(b) on the same resampled reviews. Two models'
    separate intervals can overlap while one is still reliably better; this checks that directly."""
    y_true, a, b = np.asarray(y_true), np.asarray(pred_a), np.asarray(pred_b)
    gain = (a == y_true).astype(float) - (b == y_true).astype(float)
    rng = np.random.default_rng(seed)
    means = [gain[rng.integers(0, len(gain), len(gain))].mean() for _ in range(n)]
    return np.percentile(means, [2.5, 97.5]).round(4).tolist()


def run_metadata(data_report: Path) -> dict:
    try:
        commit = subprocess.run(
            ["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        commit = "unknown"
    report = json.loads(data_report.read_text())
    return {
        "git_commit": commit,
        "data_revision": report["revision"],
        "splits_sha256": {k: v["sha256"] for k, v in report["splits"].items()},
    }
