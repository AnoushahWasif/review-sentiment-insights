import json

import numpy as np
import pandas as pd
import pytest
from sklearn.metrics import f1_score

from reviews import compare, models
from reviews.evaluate import bootstrap, paired_difference, run_metadata, scores

POS = ["great product, love it", "excellent and works well", "love it, great value"] * 10
NEG = ["terrible, broke in a day", "awful, waste of money", "broke, terrible and awful"] * 10
TEXTS, LABELS = POS + NEG, [1] * len(POS) + [0] * len(NEG)


@pytest.mark.parametrize("make", [models.tfidf, models.bag_of_words, models.length_only])
def test_models_fit_and_cope_with_unseen_and_empty_text(make):
    model = make().fit(TEXTS, LABELS)
    pred = model.predict(["completely unseen words here", "", "great"])
    assert set(pred) <= {0, 1}
    assert model.predict_proba([""]).shape == (1, 2)


def test_text_models_learn_the_obvious_words():
    model = models.tfidf().fit(TEXTS, LABELS)
    assert model.predict(["great, love it", "awful, terrible"]).tolist() == [1, 0]


def test_one_letter_tokens_are_kept_for_negation():
    model = models.tfidf().fit(["it isn't good", "it is good"] * 2, [0, 1] * 2)
    assert "t" in model.named_steps["vec"].vocabulary_


def test_majority_predicts_the_most_common_class():
    model = models.majority().fit(TEXTS[:40], LABELS[:40])
    assert set(model.predict(["anything"])) == {1}


def test_vader_reads_obvious_sentiment_without_training():
    vader = models.Vader()
    assert vader.predict(["I love it, wonderful!", "Terrible. Awful. I hate it."]).tolist() == [
        1,
        0,
    ]
    # No words, no sentiment: a compound score of 0 is not above the threshold.
    assert vader.predict([""]).tolist() == [0]


def test_scores_report_negative_class_f1_and_skip_auc_with_one_class():
    out = scores([0, 0, 1, 1], [0, 1, 1, 1], [0.1, 0.6, 0.7, 0.9])
    assert out["accuracy"] == 0.75
    assert out["f1_negative"] == pytest.approx(f1_score([0, 0, 1, 1], [0, 1, 1, 1], pos_label=0))
    assert out["auc"] == 1.0
    assert "auc" not in scores([1, 1], [1, 1], [0.2, 0.9])


def test_bootstrap_interval_contains_the_estimate_and_is_reproducible():
    rng = np.random.default_rng(1)
    y = rng.integers(0, 2, 2000)
    pred = np.where(rng.random(2000) < 0.85, y, 1 - y)
    ci = bootstrap(y, pred, n=300)
    low, high = ci["accuracy_95"]
    assert low < (y == pred).mean() < high
    low, high = ci["f1_negative_95"]
    assert low < f1_score(y, pred, pos_label=0) < high
    assert bootstrap(y, pred, n=300) == ci


def test_bootstrap_edge_cases():
    assert bootstrap([1, 0, 1], [1, 0, 1], n=50)["accuracy_95"] == [1.0, 1.0]
    # No negatives anywhere: F1 for the negative class is 0, not a division error.
    assert bootstrap([1, 1], [1, 1], n=10)["f1_negative_95"] == [0.0, 0.0]
    with pytest.raises(ValueError):
        bootstrap([], [])


def test_paired_difference():
    y = np.array([0, 1] * 500)
    assert paired_difference(y, y, y, n=100) == [0.0, 0.0]
    worse = y.copy()
    worse[:100] = 1 - worse[:100]
    low, high = paired_difference(y, y, worse, n=200)
    assert 0 < low <= 0.1 <= high


def test_run_metadata_without_git(tmp_path, monkeypatch):
    report = tmp_path / "report.json"
    report.write_text(json.dumps({"revision": "abc", "splits": {"train": {"sha256": "f00"}}}))
    monkeypatch.chdir(tmp_path)
    meta = run_metadata(report)
    assert meta["git_commit"] == "unknown"
    assert meta["splits_sha256"] == {"train": "f00"}


def test_compare_runs_end_to_end_on_a_tiny_dataset(tmp_path, monkeypatch):
    processed = tmp_path / "processed"
    processed.mkdir()
    titles = ["Great"] * len(POS) + ["Bad"] * len(NEG)
    df = pd.DataFrame({"label": LABELS, "title": titles, "content": TEXTS})
    df.to_parquet(processed / "train.parquet")
    df.to_parquet(processed / "validation.parquet")
    (processed / "report.json").write_text(json.dumps({"revision": "r", "splits": {}}))
    monkeypatch.setenv("REVIEWS_DATA_DIR", str(tmp_path))

    out = tmp_path / "out.json"
    report = compare.main(out)
    assert json.loads(out.read_text())["results"].keys() == report["results"].keys()
    assert report["results"]["tf-idf"]["accuracy"] == 1.0
    assert report["results"]["majority"]["accuracy"] == 0.5
