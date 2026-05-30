"""eerie extensions — labeled post-thesis explorations, NOT the thesis models.

The thesis core lives in the :mod:`eerie` package and uses the candidate's
fine-tuned models (``ektvho/bart-cnn`` for summarization and ``ektvho/sd-vist``,
Stable Diffusion 2.1 on VIST, for image generation). Everything under
``extensions/`` is a separate exploration and is never the thesis pipeline:

* :mod:`extensions.multi_style` — render a story by swapping community Stable
  Diffusion 1.5 art-style checkpoints (Ghibli, modern animation, watercolor).
"""
