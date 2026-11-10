import json

import numpy as np
import pandas as pd
import pytest

from reviews import models, tune
from reviews.evaluate import calibration
from reviews.model_store import (
    MANIFEST_FILE,
    MODEL_FILE,
    ModelIntegrityError,
    load_model,
    save_model,
)

TEXTS = ["great, love it", "awful, broke"] * 20
LABELS = [1, 0] * 20


@pytest.fixture
def saved(tmp_path):
    model = models.tfidf().fit(TEXTS, LABELS)
    digest = save_model(model, tmp_path, {"C": 1.0})
    return tmp_path, digest


def test_round_trip(saved):
    folder, digest = saved
    loaded = load_model(folder, digest)
    assert loaded.model.predict(["love it", "broke"]).tolist() == [1, 0]
    assert loaded.manifest["C"] == 1.0


def test_swapped_model_file_is_refused(saved):
    folder, digest = saved
    other = models.tfidf().fit(TEXTS, [1 - y for y in LABELS])
    save_model(other, folder / "other")
    (folder / MODEL_FILE).write_bytes((folder / "other" / MODEL_FILE).read_bytes())
    with pytest.raises(ModelIntegrityError, match="hash"):
        load_model(folder)


def test_pinned_hash_beats_an_edited_manifest(saved):
    folder, digest = saved
    manifest = json.loads((folder / MANIFEST_FILE).read_text())
    manifest["sha256"] = "0" * 64
    (folder / MANIFEST_FILE).write_text(json.dumps(manifest))
    with pytest.raises(ModelIntegrityError):
        load_model(folder)
    assert load_model(folder, digest.upper()).model is not None


@pytest.mark.parametrize("damage", ["no manifest", "bad json", "no model", "no hash"])
def test_damaged_folders_are_refused(saved, damage):
    folder, _ = saved
    if damage == "no manifest":
        (folder / MANIFEST_FILE).unlink()
    elif damage == "bad json":
        (folder / MANIFEST_FILE).write_text("{not json")
    elif damage == "no model":
        (folder / MODEL_FILE).unlink()
    else:
        (folder / MANIFEST_FILE).write_text(json.dumps({"sklearn_version": "x"}))
    with pytest.raises(ModelIntegrityError):
        load_model(folder)


def test_model_from_another_sklearn_version_is_refused(saved):
    folder, digest = saved
    manifest = json.loads((folder / MANIFEST_FILE).read_text())
    manifest["sklearn_version"] = "0.1"
    (folder / MANIFEST_FILE).write_text(json.dumps(manifest))
    with pytest.raises(ModelIntegrityError, match="retrain"):
        load_model(folder, digest)


def test_calibration_of_a_perfectly_calibrated_model():
    rng = np.random.default_rng(0)
    prob = rng.random(100_000)
    y = (rng.random(100_000) < prob).astype(int)
    out = calibration(y, prob)
    assert out["ece"] < 0.01
    assert len(out["bins"]) == 10 and sum(b["rows"] for b in out["bins"]) == 100_000
    assert out["brier"] == pytest.approx(1 / 6, abs=0.01)


def test_calibration_of_an_overconfident_model_and_edge_values():
    out = calibration([1, 0, 1, 0], [1.0, 1.0, 0.0, 0.0])
    assert out["ece"] == 0.5 and out["brier"] == 0.5
    assert [b["bin"] for b in out["bins"]] == ["0.0-0.1", "0.9-1.0"]
    with pytest.raises(ValueError):
        calibration([1], [1.2])
    with pytest.raises(ValueError):
        calibration([], [])


def test_tune_end_to_end_on_a_tiny_dataset(tmp_path, monkeypatch):
    processed = tmp_path / "processed"
    processed.mkdir()
    df = pd.DataFrame({"label": LABELS, "title": ["Yes", "No"] * 20, "content": TEXTS})
    df.to_parquet(processed / "train.parquet")
    df.to_parquet(processed / "validation.parquet")
    (processed / "report.json").write_text(json.dumps({"revision": "r", "splits": {}}))
    monkeypatch.setenv("REVIEWS_DATA_DIR", str(tmp_path))
    monkeypatch.setattr(tune, "TRAIN_SIZES", (10, 40))
    monkeypatch.setattr(tune, "C_GRID", (0.5, 1.0))

    report = tune.main(tmp_path / "out.json", tmp_path / "model")
    assert report["best"]["accuracy"] == 1.0
    assert set(report["learning_curve_accuracy"]) == {"10", "40"}
    assert load_model(tmp_path / "model", report["model_sha256"]).manifest["C"] in (0.5, 1.0)
