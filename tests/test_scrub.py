import pytest

from reviews.scrub import excerpt, scrub


@pytest.mark.parametrize(
    "text, expected",
    [
        ("mail me at jo.smith+x@example.co.uk now", "mail me at [email] now"),
        ("see https://example.com/a?b=1 for more", "see [url] for more"),
        ("go to www.example.org", "go to [url]"),
        ("bought it on amazon.com last week", "bought it on [url] last week"),
        ("call 555-123-4567 today", "call [phone] today"),
        ("call (555) 123 4567", "call [phone]"),
        ("ring +44 20 7946 0958", "ring [phone]"),
    ],
)
def test_masks_contact_details(text, expected):
    assert scrub(text) == expected


@pytest.mark.parametrize(
    "text",
    [
        "I bought this in 1999 and it still works",
        "It cost $1,299 which is too much",
        "Rated 4.5 out of 5, model 3000",
        "Version 2.0 is better than 1.0",
        "",
    ],
)
def test_leaves_ordinary_numbers_alone(text):
    assert scrub(text) == text


def test_excerpt_cuts_on_a_word_boundary_and_marks_the_cut():
    text = "word " * 100
    out = excerpt(text, limit=50)
    assert len(out) <= 51 and out.endswith("…") and not out.endswith(" …")
    assert out[:-1].split() == ["word"] * len(out[:-1].split())


def test_excerpt_keeps_short_text_and_collapses_whitespace():
    assert excerpt("  short\n\treview  ") == "short review"


def test_excerpt_cuts_a_single_long_token():
    assert excerpt("x" * 500, limit=10) == "x" * 10 + "…"


def test_excerpt_scrubs_before_cutting():
    assert "@" not in excerpt("a" * 280 + " me@example.com", limit=300)
