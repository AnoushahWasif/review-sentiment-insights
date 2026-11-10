"""The one cleaning step used for training and for scoring, so they can't drift apart.

Kept small on purpose: the EDA found HTML entities, tags and URLs in under 0.2% of reviews.
Case is left alone here; the vectorisers lowercase, and VADER uses capitals as a signal.
"""

import html
import re
import unicodedata

MAX_CHARS = 5_000

TAG = re.compile(r"<[a-zA-Z/][^<>]{0,200}>")
SPACE = re.compile(r"\s+")


def clean(text: str) -> str:
    text = unicodedata.normalize("NFKC", text[:MAX_CHARS])
    text = html.unescape(TAG.sub(" ", text))
    return SPACE.sub(" ", text).strip()


def review_text(title: str, content: str) -> str:
    """Title and body as one text. A title that already ends in punctuation gets no extra full
    stop, so 'Great!' doesn't become 'Great!.'."""
    title, content = clean(title or ""), clean(content or "")
    if not title:
        return content
    if not content:
        return title
    if title[-1] in ".!?":
        return f"{title} {content}"
    return f"{title}. {content}"
