"""eerie — turn a written children's story into illustrated, restyled panels.

A small, importable pipeline distilled from the 2023 YTU senior thesis *"Text
Summarization and Image Generation"* (illustrating children's stories). The core
wires the candidate's **own fine-tuned models** into a single end-to-end flow:

1. summarize the story with the fine-tuned BART ``ektvho/bart-cnn`` (fine-tuned
   from the LSG long-context base ``ccdv/lsg-bart-base-4096-booksum`` on
   CNN/DailyMail) — on by default, the way the thesis summarizes first,
2. split the summary into one scene per sentence with NLTK,
3. illustrate each scene with the fine-tuned ``ektvho/sd-vist`` — Stable
   Diffusion 2.1 fine-tuned on the VIST (Visual Storytelling) dataset,
4. (optional) restyle every panel with InstructPix2Pix
   (``timbrooks/instruct-pix2pix``).

All models are loaded lazily (see :mod:`eerie.models`) so importing this package
does not allocate any GPU memory; the first call that needs a model loads it.

The fine-tuned models are review-verified — a RunPod GPU run confirms both
``ektvho/bart-cnn`` and ``ektvho/sd-vist`` load and run end-to-end on an NVIDIA
A40. Model cards: https://huggingface.co/ektvho/bart-cnn and
https://huggingface.co/ektvho/sd-vist .

Extensions (clearly **not** the thesis models) live under ``extensions/``: a
multi-style explorer that swaps community Stable Diffusion 1.5 art-style
checkpoints (``extensions/multi_style/``).

Honesty note: panels are generated independently. A fixed seed makes runs
reproducible, but there is **no** cross-panel character consistency mechanism
(no shared character embedding, IP-Adapter, or ControlNet). The same character
will look different from panel to panel.
"""

from eerie.pipeline import run_pipeline

__all__ = ["run_pipeline"]
__version__ = "0.2.0"
