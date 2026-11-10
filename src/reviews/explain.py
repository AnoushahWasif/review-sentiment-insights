"""Which words push a review towards each class, for the model as a whole and for one review.

For a linear model the explanation is exact, not an approximation: the log-odds of "positive"
are the intercept plus, for each term in the review, its TF-IDF weight times its coefficient.
"""

import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline


def _parts(model: Pipeline):
    vec, clf = model.named_steps["vec"], model.named_steps["clf"]
    if clf.coef_.shape[0] != 1:
        raise ValueError("expected a binary linear model")
    return vec, clf, vec.get_feature_names_out(), clf.coef_[0]


def top_terms(model: Pipeline, n: int = 20) -> pd.DataFrame:
    """The n terms with the largest coefficients towards each class."""
    _, _, names, coef = _parts(model)
    order = np.argsort(coef)
    neg, pos = order[:n], order[::-1][:n]
    return pd.DataFrame(
        {
            "negative term": names[neg],
            "negative weight": coef[neg].round(2),
            "positive term": names[pos],
            "positive weight": coef[pos].round(2),
        }
    )


def contributions(model: Pipeline, text: str, n: int | None = 10) -> pd.DataFrame:
    """Terms in `text` and what each adds to the log-odds of positive, largest effect first."""
    vec, _, names, coef = _parts(model)
    row = vec.transform([text]).tocsr()
    idx, weights = row.indices, row.data
    out = pd.DataFrame({"term": names[idx], "contribution": weights * coef[idx]})
    out = out.reindex(out["contribution"].abs().sort_values(ascending=False).index)
    return out.head(n).reset_index(drop=True) if n else out.reset_index(drop=True)


def log_odds(model: Pipeline, text: str) -> float:
    """Intercept plus every contribution; equals the model's own decision function."""
    clf = model.named_steps["clf"]
    return float(clf.intercept_[0] + contributions(model, text, n=None)["contribution"].sum())


def terms_matching(model: Pipeline, pattern: str, n: int = 15) -> pd.DataFrame:
    """Coefficients of the terms matching a regex, e.g. to see how much star counts matter."""
    _, _, names, coef = _parts(model)
    hit = pd.Series(names).str.contains(pattern, regex=True).to_numpy()
    out = pd.DataFrame({"term": names[hit], "weight": coef[hit]})
    return out.reindex(out["weight"].abs().sort_values(ascending=False).index).head(n)
