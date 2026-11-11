"""reviews score | topics | holdout.

reviews score --in reviews.csv --out scored.csv     add sentiment and a probability
reviews topics --in reviews.csv                     complaint topics of the negative ones
reviews holdout                                     score the untouched test set, once
"""

import argparse
import csv
import os
import sys
from pathlib import Path

import pandas as pd

from reviews.model_store import ModelIntegrityError, load_model
from reviews.text import review_text

MAX_BYTES = 50 * 1024 * 1024
MAX_ROWS = 200_000
MODEL_DIR = Path("models/ngrams-best")
TOPIC_DIR = Path("models/topics")
HOLDOUT_REPORT = Path("reports/holdout.json")
ADDED = ("prob_positive", "sentiment")
# A cell starting with one of these is read as a formula by Excel and LibreOffice.
FORMULA_START = ("=", "+", "-", "@", "\t", "\r")


class InputError(ValueError):
    pass


def safe_cell(value) -> str:
    """Neutralise spreadsheet formula injection in text that came from a review (decision 8)."""
    value = str(value)
    return "'" + value if value.startswith(FORMULA_START) else value


def read_input(path: Path) -> pd.DataFrame:
    """A CSV with a `text` column and optionally `title`. Everything else is kept as it is.
    Rejects anything it would have to guess about (decision 9)."""
    if path.suffix.lower() != ".csv":
        raise InputError("input must be a .csv file")
    try:
        size = path.stat().st_size
    except OSError as exc:
        raise InputError(f"can't read {path}") from exc
    if size > MAX_BYTES:
        raise InputError(f"input is {size / 1e6:.0f} MB; the limit is {MAX_BYTES // 2**20} MB")
    # Read with the csv module rather than pandas: pandas renames a repeated column to "name.1"
    # and quietly turns extra fields in a row into an index, and both should be errors here.
    try:
        with path.open(encoding="utf-8", newline="") as f:
            reader = csv.reader(f)
            header = next(reader, None)
            if header is None:
                raise InputError("not a readable CSV: the file is empty")
            rows = []
            for row in reader:
                if not row:
                    continue
                if len(row) != len(header):
                    raise InputError(
                        f"not a readable CSV: record {len(rows) + 1} has {len(row)} fields, "
                        f"the header has {len(header)}"
                    )
                rows.append(row)
                if len(rows) > MAX_ROWS:
                    raise InputError(f"more than {MAX_ROWS:,} rows; split the file")
    except UnicodeDecodeError as exc:
        raise InputError("input isn't UTF-8") from exc
    except csv.Error as exc:
        if "field limit" in str(exc):
            limit = csv.field_size_limit()
            raise InputError(f"a cell is longer than {limit:,} characters") from exc
        raise InputError(f"not a readable CSV: {exc}") from exc
    if len(set(header)) != len(header):
        raise InputError("duplicate column names")
    if "text" not in header:
        raise InputError(f"no `text` column; found {header[:10]}")
    clash = [c for c in ADDED if c in header]
    if clash:
        raise InputError(f"input already has {clash}; rename them so they aren't overwritten")
    return pd.DataFrame(rows, columns=header, dtype=str)


def texts_of(df: pd.DataFrame) -> list[str]:
    titles = df["title"] if "title" in df.columns else [""] * len(df)
    return [review_text(t, c) for t, c in zip(titles, df["text"], strict=True)]


def pinned(env: str) -> str | None:
    """A hash pinned in the environment beats the manifest next to the model."""
    return os.environ.get(env) or None


def score(df: pd.DataFrame, model) -> pd.DataFrame:
    prob = model.predict_proba(texts_of(df))[:, 1] if len(df) else []
    out = df.copy()
    out["prob_positive"] = [round(float(p), 4) for p in prob]
    out["sentiment"] = ["positive" if p >= 0.5 else "negative" for p in prob]
    return out


def write_output(df: pd.DataFrame, out: Path, source: Path) -> None:
    if out.suffix.lower() != ".csv":
        raise InputError("--out must be a .csv file")
    if out.resolve() == source.resolve():
        raise InputError("--out would overwrite the input")
    safe = df.copy()
    # Not `dtype == object`: pandas 3 stores text as its own str dtype, which that misses.
    for col in safe.columns:
        if not pd.api.types.is_numeric_dtype(safe[col]):
            safe[col] = safe[col].map(safe_cell)
    safe.columns = [safe_cell(c) for c in safe.columns]
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_suffix(".csv.part")
    safe.to_csv(tmp, index=False)
    tmp.replace(out)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="reviews", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    sub = parser.add_subparsers(dest="command", required=True)
    sc = sub.add_parser("score", help="add sentiment and a probability to a CSV of reviews")
    sc.add_argument("--in", dest="source", type=Path, required=True)
    sc.add_argument("--out", type=Path, required=True)
    tp = sub.add_parser("topics", help="complaint topics of the reviews scored negative")
    tp.add_argument("--in", dest="source", type=Path, required=True)
    sub.add_parser("holdout", help="score the chosen model on the test set, once")
    args = parser.parse_args(argv)

    try:
        if args.command == "holdout":
            from reviews.holdout import main as holdout_main

            result = holdout_main(HOLDOUT_REPORT)
            test = result["test"]
            low, high = test["accuracy_95"]
            print(
                f"test accuracy {test['accuracy']:.2%} ({low:.2%} to {high:.2%}) on "
                f"{result['reviews']:,} reviews; wrote {HOLDOUT_REPORT}"
            )
            return 0
        df = read_input(args.source)
        model = load_model(MODEL_DIR, pinned("REVIEWS_MODEL_SHA256")).model
        scored = score(df, model)
        if args.command == "score":
            write_output(scored, args.out, args.source)
            neg = (scored["sentiment"] == "negative").sum()
            print(f"scored {len(scored):,} reviews: {neg:,} negative; wrote {args.out}")
            return 0
        from reviews.topics import assign

        topic_model = load_model(TOPIC_DIR, pinned("REVIEWS_TOPICS_SHA256")).model
        negative = scored[scored["sentiment"] == "negative"]
        counts = pd.Series(assign(topic_model, texts_of(negative))).value_counts()
        print(f"{len(negative):,} of {len(scored):,} reviews scored negative")
        for topic, n in counts.items():
            print(f"{n:7,}  {n / max(len(negative), 1):6.1%}  {topic}")
        return 0
    except (InputError, ModelIntegrityError, RuntimeError, FileNotFoundError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
