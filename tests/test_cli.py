import json

import pandas as pd
import pytest

from reviews import cli, holdout, models, topics
from reviews.model_store import MODEL_FILE, save_model

TEXTS = ["great, love it", "awful, broke"] * 20
LABELS = [1, 0] * 20


@pytest.fixture
def setup(tmp_path, monkeypatch):
    """A toy sentiment model and topic model saved where the CLI looks for them."""
    save_model(models.tfidf().fit(TEXTS, LABELS), tmp_path / "model")
    vec, matrix = topics.vectorise(["awful broke battery", "awful wrong item seller"] * 5, 1, 1.0)
    nmf, _ = topics.fit(matrix, 2)
    save_model(topics.TopicModel(vec, nmf, (), vec, nmf, ("a", "b")), tmp_path / "topics")
    monkeypatch.setattr(cli, "MODEL_DIR", tmp_path / "model")
    monkeypatch.setattr(cli, "TOPIC_DIR", tmp_path / "topics")
    monkeypatch.delenv("REVIEWS_MODEL_SHA256", raising=False)
    monkeypatch.delenv("REVIEWS_TOPICS_SHA256", raising=False)
    return tmp_path


def write(path, text, encoding="utf-8"):
    path.write_bytes(text.encode(encoding) if isinstance(text, str) else text)
    return path


def run(*args):
    return cli.main([str(a) for a in args])


def test_score_adds_two_columns_and_keeps_the_rest(setup):
    src = write(setup / "in.csv", 'id,text\n7,"great, love it"\n8,"awful, broke"\n')
    assert run("score", "--in", src, "--out", setup / "out" / "scored.csv") == 0
    out = pd.read_csv(setup / "out" / "scored.csv")
    assert out["id"].tolist() == [7, 8]
    assert out["sentiment"].tolist() == ["positive", "negative"]
    assert out["prob_positive"].between(0, 1).all()
    assert not list(setup.glob("out/*.part"))


def test_title_is_optional_and_used_when_present(setup):
    src = write(setup / "in.csv", "title,text\nlove it,great\n")
    assert run("score", "--in", src, "--out", setup / "out.csv") == 0


def test_formula_cells_and_headers_are_escaped(setup):
    src = write(
        setup / "in.csv",
        '@cmd,text\n=1+1,"=HYPERLINK(""http://x"")"\nok,"line one\nline two, ""quoted"""\n',
    )
    assert run("score", "--in", src, "--out", setup / "out.csv") == 0
    out = pd.read_csv(setup / "out.csv", dtype=str)
    assert out.columns[0] == "'@cmd"
    assert out["text"][0].startswith("'=HYPERLINK")
    assert out["'@cmd"][0] == "'=1+1"
    assert out["text"][1] == 'line one\nline two, "quoted"'


def test_header_only_file_gives_an_empty_result(setup):
    src = write(setup / "in.csv", "text\n")
    assert run("score", "--in", src, "--out", setup / "out.csv") == 0
    assert len(pd.read_csv(setup / "out.csv")) == 0


BAD_INPUTS = {
    "not csv": ("in.txt", "text\nfine\n", "must be a .csv"),
    "no text column": ("in.csv", "body\nfine\n", "no `text` column"),
    "duplicate columns": ("in.csv", "text,text\na,b\n", "duplicate column"),
    "not utf-8": ("in.csv", b"text\n\xff\xfe broken\n", "isn't UTF-8"),
    "ragged rows": ("in.csv", "text\na,b,c\n", "not a readable CSV"),
    "empty file": ("in.csv", "", "not a readable CSV"),
    "would overwrite added columns": ("in.csv", "text,sentiment\na,b\n", "already has"),
}


@pytest.mark.parametrize("name", BAD_INPUTS)
def test_bad_input_fails_with_a_message(setup, capsys, name):
    filename, content, message = BAD_INPUTS[name]
    src = write(setup / filename, content)
    assert run("score", "--in", src, "--out", setup / "out.csv") == 1
    assert message in capsys.readouterr().err
    assert not (setup / "out.csv").exists()


def test_missing_input_file(setup, capsys):
    assert run("score", "--in", setup / "nope.csv", "--out", setup / "out.csv") == 1
    assert "can't read" in capsys.readouterr().err


def test_size_and_row_caps(setup, capsys, monkeypatch):
    src = write(setup / "in.csv", "text\na\nb\nc\n")
    monkeypatch.setattr(cli, "MAX_ROWS", 2)
    assert run("score", "--in", src, "--out", setup / "out.csv") == 1
    assert "rows" in capsys.readouterr().err
    monkeypatch.setattr(cli, "MAX_BYTES", 5)
    assert run("score", "--in", src, "--out", setup / "out.csv") == 1
    assert "limit" in capsys.readouterr().err


def test_output_must_not_replace_the_input_or_be_another_format(setup, capsys):
    src = write(setup / "in.csv", "text\ngreat\n")
    assert run("score", "--in", src, "--out", src) == 1
    assert "overwrite the input" in capsys.readouterr().err
    assert src.read_text() == "text\ngreat\n"
    assert run("score", "--in", src, "--out", setup / "out.xlsx") == 1


def test_a_swapped_model_file_is_refused(setup, capsys):
    (setup / "model" / MODEL_FILE).write_bytes(b"not the model")
    src = write(setup / "in.csv", "text\ngreat\n")
    assert run("score", "--in", src, "--out", setup / "out.csv") == 1
    assert "hash does not match" in capsys.readouterr().err


def test_a_pinned_hash_beats_the_manifest(setup, capsys, monkeypatch):
    monkeypatch.setenv("REVIEWS_MODEL_SHA256", "0" * 64)
    src = write(setup / "in.csv", "text\ngreat\n")
    assert run("score", "--in", src, "--out", setup / "out.csv") == 1
    assert "hash does not match" in capsys.readouterr().err


def test_topics_counts_the_negative_reviews(setup, capsys):
    src = write(setup / "in.csv", "text\nawful broke battery\nawful wrong item seller\ngreat\n")
    assert run("topics", "--in", src) == 0
    out = capsys.readouterr().out
    assert "2 of 3 reviews scored negative" in out


def test_topics_with_no_negative_reviews(setup, capsys):
    src = write(setup / "in.csv", 'text\n"great, love it"\n')
    assert run("topics", "--in", src) == 0
    assert "0 of 1 reviews scored negative" in capsys.readouterr().out


def test_holdout_runs_once(tmp_path, monkeypatch):
    model_dir = tmp_path / "model"
    sha = save_model(models.tfidf().fit(TEXTS, LABELS), model_dir)
    grams = tmp_path / "ngrams.json"
    grams.write_text(
        json.dumps({"winner": "lr", "winner_sha256": sha, "results": {"lr": {"accuracy": 0.9}}})
    )
    monkeypatch.setattr(holdout, "load", lambda split: (TEXTS, pd.Series(LABELS)))
    monkeypatch.setattr(holdout, "run_metadata", lambda path: {})
    out = tmp_path / "holdout.json"
    report = holdout.main(out, model_dir, grams)
    assert report["reviews"] == 40 and report["test"]["accuracy"] == 1.0
    assert json.loads(out.read_text())["majority_accuracy"] == 0.5
    with pytest.raises(RuntimeError, match="scored already"):
        holdout.main(out, model_dir, grams)


def test_cli_holdout_refusal_is_an_error_not_a_crash(tmp_path, capsys, monkeypatch):
    report = write(tmp_path / "holdout.json", "{}")
    monkeypatch.setattr(cli, "HOLDOUT_REPORT", report)
    assert cli.main(["holdout"]) == 1
    assert "scored already" in capsys.readouterr().err


def test_help_exits_cleanly(capsys):
    with pytest.raises(SystemExit) as exc:
        cli.main(["--help"])
    assert exc.value.code == 0
    assert "score" in capsys.readouterr().out


def test_a_cell_over_the_csv_field_limit_is_named(setup, capsys):
    src = write(setup / "in.csv", "text\n" + "a" * 200_000 + "\n")
    assert run("score", "--in", src, "--out", setup / "out.csv") == 1
    assert "a cell is longer than" in capsys.readouterr().err


def test_a_model_that_cannot_be_unpickled_is_an_error_not_a_traceback(setup, capsys):
    import hashlib

    # A pickle naming a function that doesn't exist, as if the code had moved since saving.
    data = b"cbuiltins\nno_such_function\n."
    (setup / "model" / MODEL_FILE).write_bytes(data)
    manifest = setup / "model" / "model.json"
    info = json.loads(manifest.read_text())
    info["sha256"] = hashlib.sha256(data).hexdigest()
    manifest.write_text(json.dumps(info))
    src = write(setup / "in.csv", "text\ngreat\n")
    assert run("score", "--in", src, "--out", setup / "out.csv") == 1
    assert "can't be loaded" in capsys.readouterr().err
