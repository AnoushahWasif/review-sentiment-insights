import json

import numpy as np
import pandas as pd
import pytest

from reviews import models, ngrams
from reviews.model_store import load_model

POS = ["not bad at all, works", "great value, love it", "works well, not bad"] * 10
NEG = ["not good, broke", "awful, not worth it", "not good at all"] * 10
TEXTS, LABELS = POS + NEG, [1] * len(POS) + [0] * len(NEG)


@pytest.mark.parametrize("make", [models.naive_bayes, models.linear_svm, models.tfidf])
def test_ngram_models_fit_score_and_cope_with_empty_text(make):
    model = make(ngram_range=(1, 2)).fit(TEXTS, LABELS)
    assert "not bad" in model.named_steps["vec"].vocabulary_
    score = models.positive_score(model, ["", "never seen before"])
    assert score.shape == (2,) and np.isfinite(score).all()


def test_bigrams_separate_not_bad_from_not_good():
    # Unigrams see the same "not" on both sides; the pair is what carries the meaning.
    model = models.tfidf(ngram_range=(1, 2)).fit(TEXTS, LABELS)
    assert model.predict(["not bad", "not good"]).tolist() == [1, 0]


def test_positive_score_uses_the_margin_when_there_is_no_probability():
    svm = models.linear_svm().fit(TEXTS, LABELS)
    assert not hasattr(svm, "predict_proba")
    score = models.positive_score(svm, ["love it", "awful"])
    assert score[0] > 0 > score[1]


@pytest.mark.parametrize(
    "text, hit",
    [
        ("It's not good.", True),
        ("I didn't really like it", True),
        ("Not worth the money", True),
        ("would never recommend", True),
        ("It is good", False),
        ("not the one I ordered", False),
        ("knot good", False),
    ],
)
def test_negated_sentiment_pattern(text, hit):
    assert bool(ngrams.NEGATED.search(text)) is hit


def test_ngrams_end_to_end_on_a_tiny_grid(tmp_path, monkeypatch):
    processed = tmp_path / "processed"
    processed.mkdir()
    df = pd.DataFrame({"label": LABELS, "title": "", "content": TEXTS})
    df.to_parquet(processed / "train.parquet")
    df.to_parquet(processed / "validation.parquet")
    (processed / "report.json").write_text(json.dumps({"revision": "r", "splits": {}}))
    monkeypatch.setenv("REVIEWS_DATA_DIR", str(tmp_path))
    monkeypatch.setattr(
        ngrams,
        "GRID",
        [
            ("logistic regression", models.tfidf, (1, 1), 1, "C", (1.0,)),
            ("linear svm", models.linear_svm, (1, 2), 1, "C", (0.5, 1.0)),
        ],
    )
    report = ngrams.main(tmp_path / "out.json", tmp_path / "model")
    assert set(report["results"]) == {"logistic regression 1-1", "linear svm 1-2"}
    assert report["results"]["linear svm 1-2"]["accuracy"] == 1.0
    assert report["negation"]["reviews_with_negated_sentiment_word"] > 0
    loaded = load_model(tmp_path / "model", report["winner_sha256"])
    assert loaded.manifest["name"] == report["winner"]
