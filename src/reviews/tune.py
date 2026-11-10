"""Day 4: TF-IDF + logistic regression on validation. Tunes C, draws a learning curve over train
size, measures what the title adds, checks calibration, then saves the chosen model.

    uv run python -m reviews.tune      # writes reports/tfidf_lr.json and models/tfidf-lr/
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd

from reviews import models
from reviews.compare import load
from reviews.data import SEED, data_dir
from reviews.evaluate import bootstrap, calibration, paired_difference, run_metadata, scores
from reviews.model_store import save_model
from reviews.text import review_text

C_GRID = (0.25, 0.5, 1.0, 2.0, 4.0, 8.0)
TRAIN_SIZES = (10_000, 25_000, 50_000, 100_000, 200_000)


def fit_score(train_x, train_y, val_x, val_y, c: float):
    model = models.tfidf()
    model.set_params(clf__C=c)
    model.fit(train_x, train_y)
    prob = model.predict_proba(val_x)[:, 1]
    pred = (prob >= 0.5).astype(int)
    return model, pred, prob, scores(val_y, pred, prob)


def main(out: Path = Path("reports/tfidf_lr.json"), model_dir: Path = Path("models/tfidf-lr")):
    train_x, train_y = load("train")
    val_x, val_y = load("validation")

    by_c, fitted = {}, {}
    for c in C_GRID:
        model, pred, prob, row = fit_score(train_x, train_y, val_x, val_y, c)
        by_c[c] = row
        fitted[c] = (model, pred, prob)
    best_c = max(by_c, key=lambda c: by_c[c]["accuracy"])
    model, pred, prob = fitted[best_c]

    # Same C, smaller random subsets of train: does more data still help at 200k?
    order = np.random.default_rng(SEED).permutation(len(train_y))
    curve = {}
    for n in TRAIN_SIZES:
        idx = order[:n]
        sub_x = [train_x[i] for i in idx]
        curve[n] = fit_score(sub_x, train_y.iloc[idx], val_x, val_y, best_c)[3]["accuracy"]

    # Body only, to see what the title adds.
    raw_train = pd.read_parquet(data_dir() / "processed" / "train.parquet")
    raw_val = pd.read_parquet(data_dir() / "processed" / "validation.parquet")
    body_train = [review_text("", c) for c in raw_train["content"]]
    body_val = [review_text("", c) for c in raw_val["content"]]
    _, body_pred, _, body_row = fit_score(body_train, train_y, body_val, val_y, best_c)

    report = {
        "split": "validation",
        "accuracy_by_C": {str(c): round(r["accuracy"], 4) for c, r in by_c.items()},
        "best_C": best_c,
        "best": scores(val_y, pred, prob) | bootstrap(val_y, pred),
        "learning_curve_accuracy": {str(n): round(a, 4) for n, a in curve.items()},
        "title_ablation": {
            "title + body": round(by_c[best_c]["accuracy"], 4),
            "body only": round(body_row["accuracy"], 4),
            "with title minus body only, 95%": paired_difference(val_y, pred, body_pred),
        },
        "calibration": calibration(val_y, prob),
        "run": run_metadata(data_dir() / "processed" / "report.json"),
    }
    report["model_sha256"] = save_model(
        model,
        model_dir,
        {"name": "tf-idf + logistic regression", "C": best_c, "validation": report["best"]},
    )
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2, default=float))
    return report


if __name__ == "__main__":
    print(json.dumps(main(), indent=2, default=float))
