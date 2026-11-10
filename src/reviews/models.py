"""Baselines and the linear text models. Inputs are texts already passed through `text.clean`."""

import numpy as np
from sklearn.dummy import DummyClassifier
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import FunctionTransformer
from sklearn.svm import LinearSVC

SEED = 0
# Shared by every bag-of-words variant so only the weighting differs between them. One-letter
# tokens are kept (sklearn drops them by default) so the "t" of "don't" / "isn't" survives as a
# negation marker.
VOCAB = {"lowercase": True, "min_df": 2, "token_pattern": r"(?u)\b\w+\b"}


def _log_words(texts) -> np.ndarray:
    return np.log1p([[len(t.split())] for t in texts])


def majority() -> Pipeline:
    return make_pipeline(DummyClassifier(strategy="most_frequent"))


def length_only() -> Pipeline:
    return make_pipeline(FunctionTransformer(_log_words), LogisticRegression())


def _classifier() -> LogisticRegression:
    return LogisticRegression(C=1.0, max_iter=2000, random_state=SEED)


def bag_of_words(binary: bool = False) -> Pipeline:
    return Pipeline([("vec", CountVectorizer(binary=binary, **VOCAB)), ("clf", _classifier())])


def tfidf(ngram_range: tuple[int, int] = (1, 1), min_df: int = 2) -> Pipeline:
    vec = TfidfVectorizer(
        sublinear_tf=True, **(VOCAB | {"min_df": min_df}), ngram_range=ngram_range
    )
    return Pipeline([("vec", vec), ("clf", _classifier())])


def naive_bayes(ngram_range: tuple[int, int] = (1, 1), min_df: int = 2) -> Pipeline:
    """Multinomial NB on raw counts, the model it assumes."""
    vec = CountVectorizer(**(VOCAB | {"min_df": min_df}), ngram_range=ngram_range)
    return Pipeline([("vec", vec), ("clf", MultinomialNB())])


def linear_svm(ngram_range: tuple[int, int] = (1, 1), min_df: int = 2) -> Pipeline:
    vec = TfidfVectorizer(
        sublinear_tf=True, **(VOCAB | {"min_df": min_df}), ngram_range=ngram_range
    )
    return Pipeline([("vec", vec), ("clf", LinearSVC(C=0.5, random_state=SEED))])


def positive_score(model, texts) -> np.ndarray:
    """Something to rank by for AUC: a probability where the model has one, else the SVM margin."""
    if hasattr(model, "predict_proba"):
        return model.predict_proba(texts)[:, 1]
    return model.decision_function(texts)


class Vader:
    """VADER's compound score, positive when above a threshold. No training: it's the word list
    every model has to beat. The threshold is 0, the lexicon's own neutral point."""

    def __init__(self, threshold: float = 0.0):
        from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

        self.threshold = threshold
        self._analyzer = SentimentIntensityAnalyzer()

    def score(self, texts) -> np.ndarray:
        return np.array([self._analyzer.polarity_scores(t)["compound"] for t in texts])

    def predict(self, texts) -> np.ndarray:
        return (self.score(texts) > self.threshold).astype(int)
