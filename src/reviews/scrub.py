"""Mask emails, URLs and phone numbers in review text that goes into reports and notebooks.

The data is public, but reviews sometimes include contact details, and there's no reason to copy
them into a portfolio. This is a best-effort filter for excerpts, not anonymisation.
"""

import re

EXCERPT_CHARS = 300

EMAIL = re.compile(r"[\w.+-]+@[\w-]+(?:\.[\w-]+)+")
URL = re.compile(
    r"\b(?:https?://|www\.)\S+|\b[\w-]+(?:\.[\w-]+)*\.(?:com|net|org)(?:/\S*)?\b", re.I
)
# 7+ digits with optional separators, e.g. 555-123-4567, (555) 123 4567, +44 20 7946 0958.
# Plain years and prices ("1999", "$1,299") are too short to match.
PHONE = re.compile(r"(?<!\w)\+?\(?\d[\d\s().-]{5,}\d(?!\w)")


def _phone(match: re.Match) -> str:
    digits = sum(c.isdigit() for c in match.group())
    return "[phone]" if digits >= 7 else match.group()


def scrub(text: str) -> str:
    text = EMAIL.sub("[email]", text)
    text = URL.sub("[url]", text)
    return PHONE.sub(_phone, text)


def excerpt(text: str, limit: int = EXCERPT_CHARS) -> str:
    """Scrubbed and cut to `limit` characters, on a word boundary where there is one."""
    text = " ".join(scrub(text).split())
    if len(text) <= limit:
        return text
    cut = text[:limit]
    space = cut.rfind(" ")
    return (cut[:space] if space > limit // 2 else cut).rstrip() + "…"
