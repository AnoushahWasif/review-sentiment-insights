"""Download amazon_polarity at a pinned commit, sample it, and drop reviews repeated across splits.

Outputs in data/processed: train.parquet, validation.parquet, test.parquet (columns: title,
content, label with 1 = positive) and report.json with the hashes, seed, sizes and duplicate counts.
"""

import hashlib
import json
import os
import re
import unicodedata
import urllib.request
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

REPO = "fancyzhx/amazon_polarity"
REVISION = "9d9c45c18f8c3cf1b23a3c27917b60cbf28f3289"


@dataclass(frozen=True)
class RemoteFile:
    path: str
    size: int
    sha256: str


TRAIN = (
    RemoteFile(
        "amazon_polarity/train-00000-of-00004.parquet",
        259761770,
        "57c367f8c74210dde3742b17d103af33820df3af39d029f2a5051a6f87810661",
    ),
    RemoteFile(
        "amazon_polarity/train-00001-of-00004.parquet",
        258363554,
        "14c7fee8f066fffc29eb906f80581836cca10218509a8ccf9898ff2a0c48f310",
    ),
    RemoteFile(
        "amazon_polarity/train-00002-of-00004.parquet",
        255471883,
        "5af7b024d3c4b759c1538b1987ece5bac47ef49ee98f6731fd2336caac283dc9",
    ),
    RemoteFile(
        "amazon_polarity/train-00003-of-00004.parquet",
        254410930,
        "04dcdf40f04b70dd213b910468049196c73e405decced4d292e10412bf248875",
    ),
)
TEST = RemoteFile(
    "amazon_polarity/test-00000-of-00001.parquet",
    117422360,
    "65613cc6ec1ab30e19c4dcb8d8fa5612159d77bfa493704b4a9a9c167424992e",
)
COLUMNS = ["label", "title", "content"]
TRAIN_SIZE, VALIDATION_SIZE, SEED = 200_000, 20_000, 0


def data_dir() -> Path:
    return Path(os.environ.get("REVIEWS_DATA_DIR", "data"))


def sha256_of(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def fetch(remote: RemoteFile, folder: Path, opener=urllib.request.urlopen) -> Path:
    """Download once, refusing anything bigger than the pinned size, then check the hash.
    A file that fails the check is deleted so the next run downloads it again."""
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / Path(remote.path).name
    if not path.exists():
        url = f"https://huggingface.co/datasets/{REPO}/resolve/{REVISION}/{remote.path}"
        tmp = path.with_suffix(".part")
        written = 0
        with opener(url) as response, tmp.open("wb") as out:
            for chunk in iter(lambda: response.read(1 << 20), b""):
                written += len(chunk)
                if written > remote.size:
                    out.close()
                    tmp.unlink()
                    raise ValueError(f"{path.name}: larger than the pinned {remote.size} bytes")
                out.write(chunk)
        tmp.replace(path)
    actual = sha256_of(path)
    if actual != remote.sha256:
        path.unlink()
        raise ValueError(f"{path.name}: sha256 {actual} does not match the pinned hash")
    return path


def read(path: Path) -> pd.DataFrame:
    df = pq.read_table(path, columns=COLUMNS).to_pandas()
    if not df["label"].isin([0, 1]).all():
        raise ValueError(f"{path.name}: labels other than 0 and 1")
    df["title"] = df["title"].fillna("")
    df["content"] = df["content"].fillna("")
    return df


_NOT_WORD = re.compile(r"[\W_]+")


def dedupe_key(title: str, content: str) -> str:
    """Hash of the text with case, accents, punctuation and spacing removed, so reviews that differ
    only in those count as the same review."""
    text = unicodedata.normalize("NFKD", f"{title} {content}").casefold()
    text = "".join(c for c in text if not unicodedata.combining(c))
    text = _NOT_WORD.sub(" ", text).strip()
    return hashlib.blake2b(text.encode(), digest_size=16).hexdigest()


def add_keys(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["key"] = [dedupe_key(t, c) for t, c in zip(df["title"], df["content"], strict=True)]
    return df


def stratified_sample(frames: list[pd.DataFrame], n: int, seed: int) -> pd.DataFrame:
    """The same share of each class from every shard, so no shard or class is over-represented.
    Shards are sampled one at a time to keep memory to one shard's worth."""
    total = sum(len(f) for f in frames)
    rng = np.random.default_rng(seed)
    parts = []
    for frame in frames:
        for _, group in frame.groupby("label"):
            take = round(n * len(group) / total)
            parts.append(group.iloc[rng.choice(len(group), size=take, replace=False)])
    return pd.concat(parts, ignore_index=True)


def split_and_dedupe(pool: pd.DataFrame, test: pd.DataFrame, validation_size: int, seed: int):
    """Split the pool into train and validation, then drop from both every review whose text also
    appears in test, and from train every review that appears in validation or earlier in train.
    Test itself is never changed: it's the dataset's own test set."""
    pool = pool.sample(frac=1, random_state=seed).reset_index(drop=True)
    test_keys = set(test["key"])
    in_test = pool["key"].isin(test_keys)
    report = {"pool": len(pool), "pool_in_test": int(in_test.sum())}
    pool = pool[~in_test]
    report["pool_repeats"] = int(pool["key"].duplicated().sum())
    pool = pool.drop_duplicates("key")
    validation = pool.iloc[:validation_size]
    train = pool.iloc[validation_size:]
    report["conflicting_labels_in_test"] = int(test.groupby("key")["label"].nunique().gt(1).sum())
    report["test_repeats"] = int(test["key"].duplicated().sum())
    return train.reset_index(drop=True), validation.reset_index(drop=True), report


def build(dest: Path | None = None) -> dict:
    root = dest or data_dir()
    raw, processed = root / "raw", root / "processed"
    test = add_keys(read(fetch(TEST, raw)))
    shards = [read(fetch(remote, raw)) for remote in TRAIN]
    # Oversample so the train set is still full size after duplicates are dropped.
    pool = add_keys(stratified_sample(shards, int((TRAIN_SIZE + VALIDATION_SIZE) * 1.02), SEED))
    del shards
    train, validation, report = split_and_dedupe(pool, test, VALIDATION_SIZE, SEED)
    train = train.iloc[:TRAIN_SIZE]
    if len(train) < TRAIN_SIZE:
        raise RuntimeError(f"only {len(train)} train rows left after removing duplicates")

    processed.mkdir(parents=True, exist_ok=True)
    out = {}
    for name, df in {"train": train, "validation": validation, "test": test}.items():
        path = processed / f"{name}.parquet"
        df[COLUMNS].to_parquet(path, index=False)
        out[name] = {
            "rows": len(df),
            "positive_share": round(float(df["label"].mean()), 4),
            "sha256": sha256_of(path),
        }
    report = {
        "repo": REPO,
        "revision": REVISION,
        "seed": SEED,
        "files": {Path(r.path).name: r.sha256 for r in (*TRAIN, TEST)},
        "duplicates": report,
        "splits": out,
    }
    (processed / "report.json").write_text(json.dumps(report, indent=2))
    return report


if __name__ == "__main__":
    print(json.dumps(build(), indent=2))
