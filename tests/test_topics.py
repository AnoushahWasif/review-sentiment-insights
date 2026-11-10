import numpy as np
import pytest
from scipy import sparse

from reviews import topics

BATTERY = ["battery died after two months, stopped working", "stopped working, battery dead"]
SHIPPING = ["wrong item shipped, seller refund slow", "seller shipped wrong item, no refund"]
DOCS = (BATTERY + SHIPPING) * 15


def test_two_obvious_topics_are_found():
    vec, matrix = topics.vectorise(DOCS, min_df=1, max_df=1.0)
    nmf, weights = topics.fit(matrix, 2)
    words = [set(ws) for ws in topics.top_words(nmf, vec.get_feature_names_out(), 6)]
    assert any("battery" in ws or "stopped" in ws for ws in words)
    assert any("seller" in ws or "refund" in ws for ws in words)
    dom = topics.dominant(weights)
    # Reviews about the same thing land in the same topic.
    assert len(set(dom[0::4]) | set(dom[1::4])) == 1
    assert dom[0] != dom[2]


def test_npmi_extremes():
    docs = sparse.csr_matrix(np.array([[1, 1, 0], [1, 1, 0], [0, 0, 1], [0, 0, 1]]))
    assert topics.npmi(docs, [0, 1]) == pytest.approx(1.0, abs=1e-6)
    assert topics.npmi(docs, [0, 2]) == -1.0
    independent = sparse.csr_matrix(np.array([[1, 1], [1, 0], [0, 1], [0, 0]]))
    assert topics.npmi(independent, [0, 1]) == pytest.approx(0.0, abs=1e-6)
    assert topics.npmi(docs, [0]) == 0.0


def test_choose_k_reports_coherence_and_diversity():
    vec, matrix = topics.vectorise(DOCS, min_df=1, max_df=1.0)
    table = topics.choose_k(matrix, vec.get_feature_names_out(), [2, 3], n_words=3)
    assert list(table.index) == [2, 3]
    assert ((table["coherence"] >= -1) & (table["coherence"] <= 1)).all()
    assert ((table["diversity"] > 0) & (table["diversity"] <= 1)).all()


def test_reviews_with_no_kept_words_get_no_topic():
    weights = np.array([[0.2, 0.1], [0.0, 0.0], [0.0, 0.3]])
    assert topics.dominant(weights).tolist() == [0, -1, 1]


def test_summary_and_examples():
    vec, matrix = topics.vectorise(DOCS, min_df=1, max_df=1.0)
    nmf, weights = topics.fit(matrix, 2)
    table = topics.summary(nmf, weights, vec.get_feature_names_out(), {0: "first"})
    assert table["topic"].tolist() == ["first", "topic 1"]
    assert table["share"].sum() == pytest.approx(1.0)
    top = topics.examples(weights, DOCS, 0, n=2)
    assert len(top) == 2 and all(t in DOCS for t in top)


def test_fit_is_reproducible():
    _, matrix = topics.vectorise(DOCS, min_df=1, max_df=1.0)
    first = topics.fit(matrix, 2)[1]
    second = topics.fit(matrix, 2)[1]
    assert np.allclose(first, second)


def test_run_splits_media_from_goods_and_scores_validation(monkeypatch):
    books = ["boring book, the author writes badly", "the book was dull, bad author"] * 10
    monkeypatch.setattr(topics, "MIN_DF", 1)
    monkeypatch.setattr(topics, "STAGE1_K", 3)
    monkeypatch.setattr(topics, "STAGE2_K", 2)
    monkeypatch.setattr(topics, "CHOICES", (2,))
    monkeypatch.setattr(topics, "vectorise", lambda t: _vectorise(t, min_df=1, max_df=1.0))
    out = topics.run(books + DOCS, books[:4] + DOCS[:8])
    assert out["stage1"]["media_topics"] == [
        i for i, ws in enumerate(out["stage1"]["top_words"]) if ws[0] == "book"
    ]
    assert out["stage2"]["reviews"] == {"train": len(DOCS), "validation": 8}
    assert sum(out["stage2"]["share_validation"]) == pytest.approx(1.0)


_vectorise = topics.vectorise


def test_committed_topic_labels_cover_every_topic():
    from pathlib import Path

    import pandas as pd

    labels = pd.read_csv(Path(__file__).parents[1] / "reports" / "topic_labels.csv")
    assert labels["topic"].tolist() == list(range(topics.STAGE2_K))
    assert set(labels["kind"]) <= {"complaint", "verdict", "leftover media"}
    assert labels["check_words"].str.count(r"\|").eq(2).all()
