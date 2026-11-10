"""Day 7: a reproducible random sample of the chosen model's mistakes, and the causes labelled for
them by reading each one (reports/error_labels.csv).

The CSV holds row numbers, scores and short notes, never review text.
"""

import numpy as np
import pandas as pd

SEED, SAMPLE = 7, 50
CAUSES = {
    "mixed": "Mixed or lukewarm: praise and complaint in the same review",
    "elsewhere": "Praise or blame aimed at something else: another version, a rival product, "
    "the seller, shipping",
    "wording": "Opinion without sentiment words, or wording that misleads: negation, "
    "a topic that sounds negative, poetic or broken English",
    "no opinion": "No opinion about the product: news, politics, warranty tips",
    "sarcasm": "Sarcasm or irony",
    "label": "The label contradicts the text",
    "language": "Not English",
}


def sample_mistakes(y_true, prob, n: int = SAMPLE, seed: int = SEED) -> np.ndarray:
    """Row numbers of n mistakes drawn at random without replacement, sorted."""
    y_true, prob = np.asarray(y_true), np.asarray(prob)
    wrong = np.flatnonzero((prob >= 0.5).astype(int) != y_true)
    if len(wrong) < n:
        raise ValueError(f"only {len(wrong)} mistakes, asked for {n}")
    return np.sort(np.random.default_rng(seed).choice(wrong, n, replace=False))


def load_labels(path) -> pd.DataFrame:
    labels = pd.read_csv(path)
    unknown = set(labels["cause"]) - set(CAUSES)
    if unknown:
        raise ValueError(f"unknown causes: {sorted(unknown)}")
    if labels["row"].duplicated().any():
        raise ValueError("a row is labelled twice")
    return labels
