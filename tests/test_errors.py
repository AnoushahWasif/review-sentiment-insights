from pathlib import Path

import numpy as np
import pytest

from reviews import errors

LABELS = Path(__file__).parents[1] / "reports" / "error_labels.csv"


def test_sample_is_reproducible_and_only_mistakes():
    y = np.array([0, 1] * 100)
    prob = np.where(np.arange(200) % 3 == 0, 1 - y, y).astype(float)
    rows = errors.sample_mistakes(y, prob, n=20, seed=1)
    assert len(set(rows)) == 20 and (np.diff(rows) > 0).all()
    assert ((prob[rows] >= 0.5).astype(int) != y[rows]).all()
    assert (errors.sample_mistakes(y, prob, n=20, seed=1) == rows).all()


def test_sample_refuses_more_than_there_are():
    with pytest.raises(ValueError, match="only 1"):
        errors.sample_mistakes([0, 1], [0.9, 0.8], n=2)


def test_committed_labels_are_complete_and_valid():
    labels = errors.load_labels(LABELS)
    assert len(labels) == errors.SAMPLE
    assert set(labels["cause"]) <= set(errors.CAUSES)
    # Every labelled row is a mistake according to its own recorded score.
    assert ((labels["prob_positive"] >= 0.5).astype(int) != labels["true"]).all()


def test_labels_with_an_unknown_cause_or_a_repeat_are_rejected(tmp_path):
    bad = tmp_path / "bad.csv"
    bad.write_text("row,true,prob_positive,cause,note\n1,0,0.9,typo,x\n")
    with pytest.raises(ValueError, match="unknown"):
        errors.load_labels(bad)
    bad.write_text("row,true,prob_positive,cause,note\n1,0,0.9,mixed,x\n1,0,0.9,mixed,y\n")
    with pytest.raises(ValueError, match="twice"):
        errors.load_labels(bad)
