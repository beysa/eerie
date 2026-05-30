"""Split story text into scenes (one sentence per scene) using NLTK.

The notebook's ``split_into_sentences`` helper used ``nltk.sent_tokenize`` after
``nltk.download('punkt')``. Newer NLTK releases serve the tokenizer tables under
the ``punkt_tab`` resource, so this module downloads ``punkt_tab`` instead of the
deprecated ``punkt``.
"""

from functools import lru_cache
from typing import List

import nltk


@lru_cache(maxsize=1)
def _ensure_punkt() -> None:
    """Download the NLTK sentence-tokenizer tables once per process.

    Uses ``punkt_tab`` (the current resource) rather than the deprecated
    ``punkt``. ``quiet=True`` keeps it silent when the data is already present.
    """
    nltk.download("punkt_tab", quiet=True)


def split_into_scenes(text: str) -> List[str]:
    """Split ``text`` into a list of scenes, one per sentence.

    Args:
        text: The story (or summary) prose to break into scenes.

    Returns:
        A list of sentence strings. Surrounding whitespace is stripped and empty
        fragments are dropped, so the result is safe to use as one prompt per
        panel.
    """
    _ensure_punkt()
    sentences = nltk.sent_tokenize(text)
    return [s.strip() for s in sentences if s.strip()]
