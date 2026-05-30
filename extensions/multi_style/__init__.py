"""multi_style — community SD1.5 art-style explorer (EXTENSION, not the thesis).

A post-thesis style exploration that swaps **community Stable Diffusion 1.5**
art-style checkpoints (Ghibli, modern animation, watercolor) to render a story.
This is deliberately separate from the thesis core, which uses the candidate's
fine-tuned models ``ektvho/bart-cnn`` (summarization) and ``ektvho/sd-vist``
(Stable Diffusion 2.1, VIST). None of the SD1.5 checkpoints here are the thesis
model.

For the thesis pipeline use :func:`eerie.run_pipeline`. Use
:func:`run_multi_style` here only to explore community SD1.5 styles.
"""

from .pipeline import run_multi_style
from .styles import STYLES, StylePreset, get_style, list_styles

__all__ = ["run_multi_style", "list_styles", "get_style", "STYLES", "StylePreset"]
