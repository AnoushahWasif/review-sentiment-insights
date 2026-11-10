"""Complaint topics in negative reviews with NMF on TF-IDF.

On all negative reviews, NMF's topics are mostly product types (books, films, music, games),
because what people review differs more than what they complain about. So it runs in two stages:
stage 1 marks reviews whose main topic is a media type, stage 2 finds complaint topics among the
rest (physical products and orders), where a product team could act on them.
"""

import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.decomposition import NMF
from sklearn.feature_extraction.text import TfidfVectorizer

SEED = 0
MIN_DF, MAX_DF = 20, 0.3


def vectorise(texts, min_df: int = MIN_DF, max_df: float = MAX_DF):
    vec = TfidfVectorizer(
        stop_words="english",
        min_df=min_df,
        max_df=max_df,
        ngram_range=(1, 2),
        sublinear_tf=True,
    )
    return vec, vec.fit_transform(texts)


def fit(matrix, k: int, seed: int = SEED):
    nmf = NMF(n_components=k, init="nndsvda", random_state=seed, max_iter=500)
    weights = nmf.fit_transform(matrix)
    return nmf, weights


def top_words(nmf: NMF, names, n: int = 10) -> list[list[str]]:
    return [list(names[np.argsort(row)[::-1][:n]]) for row in nmf.components_]


def npmi(binary_docs, word_ids, eps: float = 1e-12) -> float:
    """Mean normalised PMI over pairs of a topic's top words, from document co-occurrence.
    1 means the words always appear together, 0 independent, -1 never together."""
    cols = sparse.csc_matrix(binary_docs[:, word_ids], dtype=np.float64)
    n_docs = binary_docs.shape[0]
    p = np.asarray(cols.sum(axis=0)).ravel() / n_docs
    joint = (cols.T @ cols).toarray() / n_docs
    scores = []
    for i in range(len(word_ids)):
        for j in range(i + 1, len(word_ids)):
            if joint[i, j] == 0:
                scores.append(-1.0)
                continue
            pmi = np.log(joint[i, j] / (p[i] * p[j] + eps))
            scores.append(pmi / -np.log(joint[i, j] + eps))
    return float(np.mean(scores)) if scores else 0.0


def choose_k(matrix, names, ks, n_words: int = 10, seed: int = SEED) -> pd.DataFrame:
    """Coherence (mean NPMI of each topic's top words) and diversity (share of unique words
    across all topics' top words) for each k. A guide for reading, not an answer."""
    binary = (matrix > 0).astype(np.int8).tocsr()
    index = {w: i for i, w in enumerate(names)}
    rows = []
    for k in ks:
        nmf, _ = fit(matrix, k, seed)
        words = top_words(nmf, names, n_words)
        coherence = [npmi(binary, [index[w] for w in ws]) for ws in words]
        unique = len({w for ws in words for w in ws}) / (k * n_words)
        rows.append(
            {
                "k": k,
                "coherence": round(float(np.mean(coherence)), 4),
                "diversity": round(unique, 3),
            }
        )
    return pd.DataFrame(rows).set_index("k")


def dominant(weights: np.ndarray, min_weight: float = 1e-9) -> np.ndarray:
    """Each review's strongest topic, or -1 for reviews with no weight on any topic (all their
    words were filtered out as too rare or too common)."""
    best = weights.argmax(axis=1)
    best[weights.max(axis=1) <= min_weight] = -1
    return best


def summary(nmf: NMF, weights, names, labels: dict[int, str] | None = None) -> pd.DataFrame:
    dom = dominant(weights)
    words = top_words(nmf, names, 8)
    return pd.DataFrame(
        {
            "topic": [(labels or {}).get(i, f"topic {i}") for i in range(nmf.n_components)],
            "share": [round(float((dom == i).mean()), 3) for i in range(nmf.n_components)],
            "top words": [", ".join(ws) for ws in words],
        }
    )


def examples(weights, texts, topic: int, n: int = 3) -> list[str]:
    """The reviews that load most on a topic. Callers pass them through `scrub.excerpt`."""
    order = np.argsort(weights[:, topic])[::-1][:n]
    return [texts[i] for i in order]


# Stage 1 topics whose first word is one of these are media types (books, films, music, games).
MEDIA_WORDS = {"book", "movie", "film", "cd", "album", "music", "game"}
STAGE1_K, STAGE2_K, CHOICES = 8, 15, (6, 8, 10, 12, 15, 20)


def run(neg_train: list[str], neg_val: list[str]) -> dict:
    """Both stages on the training negatives, then the validation negatives pushed through the
    same fitted models, to check the topic shares hold on reviews NMF never saw."""
    vec1, m1 = vectorise(neg_train)
    nmf1, w1 = fit(m1, STAGE1_K)
    names1 = vec1.get_feature_names_out()
    media = [i for i, ws in enumerate(top_words(nmf1, names1, 1)) if ws[0] in MEDIA_WORDS]

    def goods(texts, weights):
        dom = dominant(weights)
        return [t for t, d in zip(texts, dom, strict=True) if d != -1 and d not in media]

    goods_train = goods(neg_train, w1)
    goods_val = goods(neg_val, nmf1.transform(vec1.transform(neg_val)))

    vec2, m2 = vectorise(goods_train)
    names2 = vec2.get_feature_names_out()
    nmf2, w2 = fit(m2, STAGE2_K)
    w2_val = nmf2.transform(vec2.transform(goods_val))
    dom_train, dom_val = dominant(w2), dominant(w2_val)
    return {
        "stage1": {
            "k": STAGE1_K,
            "top_words": top_words(nmf1, names1, 6),
            "media_topics": media,
            "goods_share_train": round(len(goods_train) / len(neg_train), 3),
            "goods_share_validation": round(len(goods_val) / len(neg_val), 3),
        },
        "stage2": {
            "k": STAGE2_K,
            "choose_k": choose_k(m2, names2, CHOICES).reset_index().to_dict("records"),
            "top_words": top_words(nmf2, names2, 10),
            "share_train": [round(float((dom_train == i).mean()), 3) for i in range(STAGE2_K)],
            "share_validation": [round(float((dom_val == i).mean()), 3) for i in range(STAGE2_K)],
            "reviews": {"train": len(goods_train), "validation": len(goods_val)},
        },
        "_fitted": (nmf2, w2, goods_train),
    }


if __name__ == "__main__":
    import json
    from pathlib import Path

    from reviews.compare import load
    from reviews.data import data_dir
    from reviews.evaluate import run_metadata

    train_x, train_y = load("train")
    val_x, val_y = load("validation")
    out = run(
        [t for t, y in zip(train_x, train_y, strict=True) if y == 0],
        [t for t, y in zip(val_x, val_y, strict=True) if y == 0],
    )
    out.pop("_fitted")
    out["run"] = run_metadata(data_dir() / "processed" / "report.json")
    Path("reports/topics.json").write_text(json.dumps(out, indent=2))
    print(json.dumps({k: v for k, v in out["stage2"].items() if k != "top_words"}, indent=2))
