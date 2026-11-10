"""Day 9: sentence embeddings (all-MiniLM-L6-v2, pinned commit, CPU) + logistic regression,
compared with the TF-IDF n-gram model on validation. Writes reports/embeddings.json.

    uv run python -m reviews.embeddings

Embeddings are cached in data/embeddings/ (keyed by model commit and split hash), because
encoding 220k reviews on a laptop CPU takes a while and the comparison may be rerun.
"""

import json
import time
from pathlib import Path

import numpy as np

MODEL = "sentence-transformers/all-MiniLM-L6-v2"
REVISION = "1110a243fdf4706b3f48f1d95db1a4f5529b4d41"
MAX_TOKENS = 256  # the length the model was trained on; longer reviews are cut
BATCH = 64


def mean_pool(hidden, mask):
    """Average of the token vectors, ignoring padding, then unit length: how this model's
    sentence embeddings are defined."""
    import torch

    mask = mask.unsqueeze(-1).to(hidden.dtype)
    summed = (hidden * mask).sum(dim=1)
    counts = mask.sum(dim=1).clamp(min=1e-9)
    return torch.nn.functional.normalize(summed / counts, p=2, dim=1)


def load_encoder():
    from transformers import AutoModel, AutoTokenizer

    # use_safetensors: never fall back to a pickled .bin checkpoint.
    tokenizer = AutoTokenizer.from_pretrained(MODEL, revision=REVISION)
    model = AutoModel.from_pretrained(MODEL, revision=REVISION, use_safetensors=True)
    model.eval()
    return tokenizer, model


def encode(texts, tokenizer, model, batch: int = BATCH) -> np.ndarray:
    import torch

    # Batches of similar length need little padding, which is most of the CPU time saved.
    order = np.argsort([len(t) for t in texts], kind="stable")
    out = np.zeros((len(texts), 384), np.float32)
    with torch.inference_mode():
        for start in range(0, len(texts), batch):
            idx = order[start : start + batch]
            chunk = [texts[i] if texts[i].strip() else " " for i in idx]
            tokens = tokenizer(
                chunk, padding=True, truncation=True, max_length=MAX_TOKENS, return_tensors="pt"
            )
            hidden = model(**tokens).last_hidden_state
            out[idx] = mean_pool(hidden, tokens["attention_mask"]).numpy()
    return out


def cached(split: str, texts, cache_dir: Path, split_sha: str, encoder) -> np.ndarray:
    path = cache_dir / f"{split}-{REVISION[:12]}-{split_sha[:12]}.npy"
    if path.exists():
        vectors = np.load(path, allow_pickle=False)
        if vectors.shape[0] == len(texts):
            return vectors
    vectors = encode(texts, *encoder)
    cache_dir.mkdir(parents=True, exist_ok=True)
    np.save(path, vectors, allow_pickle=False)
    return vectors


def main(out: Path = Path("reports/embeddings.json")) -> dict:
    from sklearn.linear_model import LogisticRegression

    from reviews.compare import load
    from reviews.data import data_dir
    from reviews.evaluate import bootstrap, paired_difference, run_metadata, scores
    from reviews.model_store import load_model

    processed = data_dir() / "processed"
    split_sha = {
        k: v["sha256"]
        for k, v in json.loads((processed / "report.json").read_text())["splits"].items()
    }
    train_x, train_y = load("train")
    val_x, val_y = load("validation")
    encoder = load_encoder()

    start = time.perf_counter()
    val_e = cached("validation", val_x, data_dir() / "embeddings", split_sha["validation"], encoder)
    val_s = time.perf_counter() - start
    start = time.perf_counter()
    train_e = cached("train", train_x, data_dir() / "embeddings", split_sha["train"], encoder)
    train_s = time.perf_counter() - start

    by_c, preds = {}, {}
    for c in (0.1, 1.0, 10.0):
        clf = LogisticRegression(C=c, max_iter=3000).fit(train_e, train_y)
        prob = clf.predict_proba(val_e)[:, 1]
        preds[c] = (prob >= 0.5).astype(int)
        by_c[c] = scores(val_y, preds[c], prob)
    best_c = max(by_c, key=lambda c: by_c[c]["accuracy"])
    pred = preds[best_c]

    grams = json.loads(Path("reports/ngrams.json").read_text())
    tfidf = load_model(Path("models/ngrams-best"), grams["winner_sha256"]).model
    tfidf_pred = tfidf.predict(val_x)

    report = {
        "split": "validation",
        "model": {"name": MODEL, "revision": REVISION, "max_tokens": MAX_TOKENS},
        "encode_seconds": {"validation": round(val_s, 1), "train": round(train_s, 1)},
        "accuracy_by_C": {str(c): round(r["accuracy"], 4) for c, r in by_c.items()},
        "best_C": best_c,
        "embeddings": scores(val_y, pred, None) | bootstrap(val_y, pred) | by_c[best_c],
        "tfidf_ngrams_accuracy": round(float((tfidf_pred == val_y.to_numpy()).mean()), 4),
        "embeddings_minus_tfidf_95": paired_difference(val_y, pred, tfidf_pred),
        "both_wrong": int(((pred != val_y) & (tfidf_pred != val_y)).sum()),
        "only_embeddings_wrong": int(((pred != val_y) & (tfidf_pred == val_y)).sum()),
        "only_tfidf_wrong": int(((pred == val_y) & (tfidf_pred != val_y)).sum()),
        "run": run_metadata(processed / "report.json"),
    }
    np.save(data_dir() / "embeddings" / "validation-pred.npy", pred, allow_pickle=False)
    out.write_text(json.dumps(report, indent=2, default=float))
    return report


if __name__ == "__main__":
    print(json.dumps(main(), indent=2, default=float))
