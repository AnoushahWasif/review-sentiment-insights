import pytest

from reviews import explain, models

POS = ["great value, love it", "love it, works great", "excellent, not bad"] * 10
NEG = ["awful, broke fast", "terrible, not good", "broke, awful waste"] * 10
TEXTS, LABELS = POS + NEG, [1] * len(POS) + [0] * len(NEG)


@pytest.fixture(scope="module")
def model():
    return models.tfidf(ngram_range=(1, 2), min_df=1).fit(TEXTS, LABELS)


def test_top_terms_point_the_right_way(model):
    top = explain.top_terms(model, n=3)
    assert (top["positive weight"] > 0).all() and (top["negative weight"] < 0).all()
    assert {"awful", "broke"} & set(top["negative term"])
    assert {"love", "great"} & set(top["positive term"])


@pytest.mark.parametrize("text", ["love it, works great", "awful, not good", "", "unseen words"])
def test_contributions_add_up_to_the_models_own_score(model, text):
    assert explain.log_odds(model, text) == pytest.approx(model.decision_function([text])[0])


def test_contributions_are_sorted_by_size_and_cut(model):
    out = explain.contributions(model, "great value, love it but broke", n=3)
    assert len(out) == 3
    sizes = out["contribution"].abs().tolist()
    assert sizes == sorted(sizes, reverse=True)


def test_empty_text_has_no_contributions(model):
    assert explain.contributions(model, "").empty


def test_terms_matching(model):
    out = explain.terms_matching(model, r"^not ")
    assert set(out["term"]) == {"not bad", "not good"}
    assert out.set_index("term").loc["not good", "weight"] < 0


def test_rejects_a_model_with_more_than_two_classes():
    multi = models.tfidf(min_df=1).fit(["a", "b", "c"] * 3, [0, 1, 2] * 3)
    with pytest.raises(ValueError, match="binary"):
        explain.top_terms(multi)
