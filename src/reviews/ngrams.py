"""Day 5: Naive Bayes and a linear SVM next to logistic regression, then word bigrams and trigrams.
Everything on validation. Writes reports/ngrams.json.

    uv run python -m reviews.ngrams

Each vocabulary is fitted once and its matrix reused for every classifier setting, so the grid
costs one vectoriser fit per n-gram range instead of one per point.
"""

import json
import re
import time
from pathlib import Path

import numpy as np
from sklearn.base import clone
from sklearn.pipeline import Pipeline

from reviews import models
from reviews.compare import load
from reviews.data import data_dir
from reviews.evaluate import bootstrap, paired_difference, run_metadata, scores
from reviews.model_store import save_model

# (name, builder, n-gram range, min_df, parameter name, grid)
GRID = [
    ("naive bayes", models.naive_bayes, (1, 1), 2, "alpha", (0.1, 0.5, 1.0, 2.0)),
    ("linear svm", models.linear_svm, (1, 1), 2, "C", (0.1, 0.25, 0.5, 1.0)),
    ("logistic regression", models.tfidf, (1, 1), 2, "C", (2.0,)),
    ("naive bayes", models.naive_bayes, (1, 2), 2, "alpha", (0.1, 0.5, 1.0, 2.0)),
    ("linear svm", models.linear_svm, (1, 2), 2, "C", (0.25, 0.5, 1.0, 2.0, 4.0)),
    ("logistic regression", models.tfidf, (1, 2), 2, "C", (2.0, 8.0, 32.0)),
    ("logistic regression", models.tfidf, (1, 3), 3, "C", (4.0, 8.0, 32.0)),
]
BASELINE = "logistic regression 1-1"
# Negated sentiment words, the case bag-of-words can't represent and bigrams should.
NEGATED = re.compile(
    r"\b(?:not|never|isn't|wasn't|don't|doesn't|didn't)\s+(?:\w+\s+)?"
    r"(?:good|great|bad|worth|recommend|disappoint\w*|happy|work\w*|like|love)\b",
    re.IGNORECASE,
)


def main(out: Path = Path("reports/ngrams.json"), model_dir: Path = Path("models/ngrams-best")):
    train_x, train_y = load("train")
    val_x, val_y = load("validation")

    results, predictions, fitted = {}, {}, {}
    for name, build, ngrams, min_df, param, grid in GRID:
        key = f"{name} {ngrams[0]}-{ngrams[1]}"
        template = build(ngram_range=ngrams, min_df=min_df)
        start = time.perf_counter()
        vec = clone(template.named_steps["vec"]).fit(train_x)
        train_m, val_m = vec.transform(train_x), vec.transform(val_x)
        vec_s = time.perf_counter() - start

        best = None
        for value in grid:
            clf = clone(template.named_steps["clf"]).set_params(**{param: value})
            start = time.perf_counter()
            clf.fit(train_m, train_y)
            fit_s = time.perf_counter() - start
            pipe = Pipeline([("vec", vec), ("clf", clf)])
            pred = clf.predict(val_m)
            score = models.positive_score(clf, val_m)
            row = scores(val_y, pred, score)
            row |= {param: value, "fit_seconds": round(fit_s + vec_s, 1)}
            if best is None or row["accuracy"] > best[0]["accuracy"]:
                best = (row, pred, pipe)
        row, pred, pipe = best
        row |= bootstrap(val_y, pred) | {"features": len(vec.vocabulary_)}
        results[key], predictions[key], fitted[key] = row, pred, pipe

    winner = max(results, key=lambda k: results[k]["accuracy"])
    negated = np.array([bool(NEGATED.search(t)) for t in val_x])
    y = val_y.to_numpy()
    negation = {
        "reviews_with_negated_sentiment_word": int(negated.sum()),
        "accuracy_on_them": {
            k: round(float((predictions[k][negated] == y[negated]).mean()), 4)
            for k in (BASELINE, winner)
        },
        "accuracy_on_the_rest": {
            k: round(float((predictions[k][~negated] == y[~negated]).mean()), 4)
            for k in (BASELINE, winner)
        },
    }
    report = {
        "split": "validation",
        "results": results,
        "winner": winner,
        "paired_accuracy_difference_95": {
            f"{k} minus {BASELINE}": paired_difference(val_y, predictions[k], predictions[BASELINE])
            for k in results
            if k != BASELINE
        }
        | {
            f"{winner} minus {k}": paired_difference(val_y, predictions[winner], predictions[k])
            for k in results
            if k not in (winner, BASELINE)
        },
        "negation": negation,
        "run": run_metadata(data_dir() / "processed" / "report.json"),
    }
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2, default=float))
    report["winner_sha256"] = save_model(
        fitted[winner], model_dir, {"name": winner, "validation": results[winner]}
    )
    return report


if __name__ == "__main__":
    print(json.dumps(main(), indent=2, default=float))
