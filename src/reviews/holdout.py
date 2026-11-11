"""Day 10: the chosen model on the 400k test reviews, scored once.

    uv run reviews holdout

Everything before this was chosen on validation. The test set is only useful as an honest
estimate if nothing is tuned after looking at it, so a second run is refused while
reports/holdout.json exists.
"""

import json
from pathlib import Path

from reviews.compare import load
from reviews.data import data_dir
from reviews.evaluate import bootstrap, calibration, run_metadata, scores
from reviews.model_store import load_model


def main(
    out: Path = Path("reports/holdout.json"),
    model_dir: Path = Path("models/ngrams-best"),
    ngrams_report: Path = Path("reports/ngrams.json"),
) -> dict:
    if out.exists():
        raise RuntimeError(
            f"{out} exists, so the test set has been scored already. Delete it only to "
            "score a model that was chosen without looking at the test results."
        )
    grams = json.loads(ngrams_report.read_text())
    model = load_model(model_dir, grams["winner_sha256"]).model
    texts, labels = load("test")
    prob = model.predict_proba(texts)[:, 1]
    pred = (prob >= 0.5).astype(int)
    validation = grams["results"][grams["winner"]]
    cal = calibration(labels, prob)
    report = {
        "split": "test",
        "reviews": len(labels),
        "model": grams["winner"],
        "model_sha256": grams["winner_sha256"],
        "test": scores(labels, pred, prob) | bootstrap(labels, pred),
        "validation_accuracy": validation["accuracy"],
        "majority_accuracy": round(float(max(labels.mean(), 1 - labels.mean())), 4),
        "calibration": {"brier": cal["brier"], "ece": cal["ece"]},
        "run": run_metadata(data_dir() / "processed" / "report.json"),
    }
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_suffix(".json.part")
    tmp.write_text(json.dumps(report, indent=2, default=float))
    tmp.replace(out)
    return report
