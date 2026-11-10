import pytest

from reviews.text import MAX_CHARS, clean, review_text


@pytest.mark.parametrize(
    "raw, expected",
    [
        ("I &quot;love&quot; it &amp; more", 'I "love" it & more'),
        ("line one<br/>line two", "line one line two"),
        ("<p>Good</p>", "Good"),
        ("  lots   of\n\nspace\t ", "lots of space"),
        ("ﬁne ｆｕｌｌｗｉｄｔｈ", "fine fullwidth"),
        ("Café niño", "Café niño"),
        ("NOT GOOD!!!", "NOT GOOD!!!"),
        ("", ""),
    ],
)
def test_clean(raw, expected):
    assert clean(raw) == expected


def test_clean_keeps_comparisons_that_look_a_bit_like_tags():
    assert clean("size 3 < 5 and 7 > 2") == "size 3 < 5 and 7 > 2"


def test_clean_doesnt_turn_escaped_markup_into_tags_that_then_get_stripped():
    # Unescaping happens after tags are removed, so text that was escaped stays as text.
    assert clean("use &lt;b&gt; for bold") == "use <b> for bold"


def test_clean_caps_very_long_input():
    assert len(clean("a" * (MAX_CHARS * 3))) == MAX_CHARS


@pytest.mark.parametrize(
    "title, content, expected",
    [
        ("Great", "Works well.", "Great. Works well."),
        ("Great!", "Works well.", "Great! Works well."),
        ("Why?", "It broke.", "Why? It broke."),
        ("", "Only a body", "Only a body"),
        ("Only a title", "", "Only a title"),
        (None, None, ""),
        ("  &amp; spaced  ", " body ", "& spaced. body"),
    ],
)
def test_review_text(title, content, expected):
    assert review_text(title, content) == expected
