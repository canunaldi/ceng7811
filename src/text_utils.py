"""
Text cleaning and segmentation utilities.

Implements the four preprocessing steps from paper §4.1, in order:
  1. Remove HTML tags
  2. Remove URLs
  3. Remove English stop words
  4. Lowercase all text

Then segments the cleaned text into overlapping word chunks (§4.1).
"""

import re
from typing import List

from bs4 import BeautifulSoup

import nltk
# Auto-download stopwords corpus on first use
try:
    from nltk.corpus import stopwords as _sw
    _STOPWORDS = set(_sw.words("english"))
except LookupError:
    nltk.download("stopwords", quiet=True)
    from nltk.corpus import stopwords as _sw
    _STOPWORDS = set(_sw.words("english"))


def strip_html(text: str) -> str:
    return BeautifulSoup(text, "html.parser").get_text(separator=" ")


def strip_urls(text: str) -> str:
    return re.sub(r"http\S+|www\S+", "", text)


def remove_stopwords(text: str) -> str:
    return " ".join(w for w in text.split() if w.lower() not in _STOPWORDS)


def clean_text(text: str) -> str:
    text = strip_html(text)
    text = strip_urls(text)
    text = text.lower()
    text = remove_stopwords(text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def segment_text(text: str, size: int = 200, overlap: int = 50) -> List[str]:
    """Split text into overlapping word chunks.

    Each chunk is `size` words with `overlap` words shared with the next chunk.
    Always returns at least one segment, even for very short texts.
    """
    words = text.split()
    if not words:
        return [""]
    step = size - overlap  # 150 words between segment starts
    segments = []
    for i in range(0, len(words), step):
        chunk = words[i : i + size]
        if chunk:
            segments.append(" ".join(chunk))
    return segments
