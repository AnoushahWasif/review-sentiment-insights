"""Day 3 comparison on the validation set: baselines, then bag-of-words vs TF-IDF with the same
classifier. Writes reports/compare_bow_tfidf.json.

    uv run python -m reviews.compare
"""

import json
import time
from pathlib import Path

import pandas as pd

from reviews import models
from reviews.data import data_dir
from reviews.evaluate import bootstrap, paired_difference, run_metadata, scores
from reviews.text import review_text


def load(split: str) -> tuple[list[str], pd.Series]:
    df = pd.read_parquet(data_dir() / "processed" / f"{split}.parquet")
    texts = [review_text(t, c) for t, c in zip(df["title"], df["content"], strict=True)]
    return texts, df["label"]


def main(out: Path = Path("reports/compare_bow_tfidf.json")) -> dict:
    train_x, train_y = load("train")
    val_x, val_y = load("validation")

    candidates = {
        "majority": models.majority(),
        "length only": models.length_only(),
        "bag of words (counts)": models.bag_of_words(),
        "bag of words (binary)": models.bag_of_words(binary=True),
        "tf-idf": models.tfidf(),
    }
    results, predictions = {}, {}
    for name, model in candidates.items():
        start = time.perf_counter()
        model.fit(train_x, train_y)
        fit_s = time.perf_counter() - start
        pred = model.predict(val_x)
        prob = model.predict_proba(val_x)[:, 1]
        row = scores(val_y, pred, prob) | bootstrap(val_y, pred) | {"fit_seconds": round(fit_s, 1)}
        if "vec" in getattr(model, "named_steps", {}):
            row["vocabulary"] = len(model.named_steps["vec"].vocabulary_)
        results[name], predictions[name] = row, pred

    vader = models.Vader()
    start = time.perf_counter()
    vader_score = vader.score(val_x)
    vader_pred = (vader_score > vader.threshold).astype(int)
    results["VADER (no training)"] = (
        scores(val_y, vader_pred, vader_score)
        | bootstrap(val_y, vader_pred)
        | {"score_seconds": round(time.perf_counter() - start, 1)}
    )
    predictions["VADER (no training)"] = vader_pred

    report = {
        "split": "validation",
        "rows": {"train": len(train_y), "validation": len(val_y)},
        "results": results,
        "paired_accuracy_difference_95": {
            "tf-idf minus counts": paired_difference(
                val_y, predictions["tf-idf"], predictions["bag of words (counts)"]
            ),
            "tf-idf minus binary": paired_difference(
                val_y, predictions["tf-idf"], predictions["bag of words (binary)"]
            ),
            "binary minus counts": paired_difference(
                val_y, predictions["bag of words (binary)"], predictions["bag of words (counts)"]
            ),
            "counts minus VADER": paired_difference(
                val_y, predictions["bag of words (counts)"], predictions["VADER (no training)"]
            ),
        },
        "run": run_metadata(data_dir() / "processed" / "report.json"),
    }
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2, default=float))
    return report


if __name__ == "__main__":
    report = main()
    table = pd.DataFrame(report["results"]).T
    print(table.to_string())
    print(json.dumps(report["paired_accuracy_difference_95"], indent=2))
