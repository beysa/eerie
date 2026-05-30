"""Lazy, cached loaders for the thesis's fine-tuned models.

This is the core of eerie: the two models the candidates fine-tuned for the
2023 YTU senior thesis *"Text Summarization and Image Generation"*, plus the
off-the-shelf editor used for optional style transfer:

* **Summarizer** — ``ektvho/bart-cnn``: a BART summarizer fine-tuned by the
  candidate *from* ``ccdv/lsg-bart-base-4096-booksum`` (an LSG-BART with a
  4096-token long context) on the CNN/DailyMail dataset. Its tokenizer is the
  LSG base ``ccdv/lsg-bart-base-4096-booksum``, which ships custom LSG attention
  code and therefore needs ``trust_remote_code=True``.
  https://huggingface.co/ektvho/bart-cnn
* **Image generator** — ``ektvho/sd-vist``: Stable Diffusion 2.1 fine-tuned by
  the candidate on the VIST (Visual Storytelling) dataset. SD2.1's native
  resolution is 768x768, which is the package-wide default.
  https://huggingface.co/ektvho/sd-vist
* **Style editor** — ``timbrooks/instruct-pix2pix``: off-the-shelf, image-
  conditioned editing for the optional post-hoc restyle stage (unchanged).

The original notebook loaded all models at the top of a single setup cell, which
allocated the full GPU footprint at import time. Here each model is loaded on
first use and cached for the lifetime of the process via
:func:`functools.lru_cache`. Importing :mod:`eerie` therefore loads nothing onto
the GPU; the summarizer is only built if you summarize, and the style pipeline
only if you restyle.

Both Stable Diffusion pipelines load in ``torch.float16`` and move to ``"cuda"``.
This package targets a single CUDA GPU; the loaders raise a clear error if none
is present.

The candidate's fine-tuned models are review-verified: a RunPod GPU run confirms
``ektvho/bart-cnn`` and ``ektvho/sd-vist`` both load and run end-to-end on an
NVIDIA A40.
"""

from functools import lru_cache

import torch
from diffusers import (
    EulerAncestralDiscreteScheduler,
    StableDiffusionInstructPix2PixPipeline,
    StableDiffusionPipeline,
)
from transformers import (
    AutoTokenizer,
    BartForConditionalGeneration,
    PreTrainedModel,
    PreTrainedTokenizerBase,
)

# Thesis fine-tuned models (the core), plus the off-the-shelf style editor.
# Verbatim Hugging Face ids — do not substitute off-the-shelf checkpoints here.
SUMMARIZER_MODEL_ID = "ektvho/bart-cnn"  # fine-tuned FROM the LSG base below
SUMMARIZER_TOKENIZER_ID = "ccdv/lsg-bart-base-4096-booksum"  # LSG long-context base
IMAGE_MODEL_ID = "ektvho/sd-vist"  # Stable Diffusion 2.1 fine-tuned on VIST
STYLE_MODEL_ID = "timbrooks/instruct-pix2pix"

_DEVICE = "cuda"


@lru_cache(maxsize=1)
def get_image_pipe() -> StableDiffusionPipeline:
    """Load and cache the thesis text-to-image pipeline (``ektvho/sd-vist``).

    ``ektvho/sd-vist`` is Stable Diffusion 2.1 fine-tuned by the candidate on the
    VIST (Visual Storytelling) dataset — the thesis's own image generator, not an
    off-the-shelf checkpoint. Loaded once in ``float16`` on CUDA and cached for
    the lifetime of the process. SD2.1's native resolution is 768x768 (see
    :func:`eerie.generate.generate_panels`).

    Returns:
        A cached :class:`~diffusers.StableDiffusionPipeline` on the GPU.

    Raises:
        RuntimeError: If no CUDA GPU is available (the pipe loads in float16 on cuda).
    """
    if not torch.cuda.is_available():
        raise RuntimeError(
            f"eerie requires a CUDA GPU ({IMAGE_MODEL_ID} loads in float16 on cuda)."
        )
    pipe = StableDiffusionPipeline.from_pretrained(
        IMAGE_MODEL_ID, torch_dtype=torch.float16
    )
    pipe = pipe.to(_DEVICE)
    pipe.enable_attention_slicing()
    pipe.enable_vae_slicing()
    return pipe


@lru_cache(maxsize=1)
def get_summarizer() -> "tuple[PreTrainedModel, PreTrainedTokenizerBase]":
    """Load and cache the thesis summarizer (``ektvho/bart-cnn``) + LSG tokenizer.

    ``ektvho/bart-cnn`` is the candidate's BART summarizer, fine-tuned *from*
    ``ccdv/lsg-bart-base-4096-booksum`` (an LSG-BART with a 4096-token long
    context) on CNN/DailyMail. Because the weights inherit the LSG attention, the
    matching tokenizer is the LSG base ``ccdv/lsg-bart-base-4096-booksum``, loaded
    with ``trust_remote_code=True`` so its custom LSG code is available.

    Returned as a ``(model, tokenizer)`` pair rather than a
    :class:`~transformers.Pipeline` so the caller controls the long-context
    encoding directly (see :func:`eerie.summarize.summarize`). The model is moved
    to CUDA when a GPU is available and otherwise stays on CPU — summarization is
    light enough to run on CPU, unlike the diffusion pipelines.

    Returns:
        A cached ``(BartForConditionalGeneration, tokenizer)`` tuple.
    """
    tokenizer = AutoTokenizer.from_pretrained(
        SUMMARIZER_TOKENIZER_ID, trust_remote_code=True
    )
    model = BartForConditionalGeneration.from_pretrained(SUMMARIZER_MODEL_ID)
    if torch.cuda.is_available():
        model = model.to(_DEVICE)
    model.eval()
    return model, tokenizer


@lru_cache(maxsize=1)
def get_style_pipe() -> StableDiffusionInstructPix2PixPipeline:
    """Load and cache the InstructPix2Pix image-editing pipeline.

    ``timbrooks/instruct-pix2pix`` edits an existing panel from a text
    instruction while preserving composition, which is what makes the optional
    post-hoc restyle stage possible. This is an off-the-shelf editor, not a
    thesis model; it is unchanged from the original pipeline. Loaded in
    ``float16`` on CUDA with the safety checker disabled and an
    :class:`~diffusers.EulerAncestralDiscreteScheduler`, matching the notebook.

    Returns:
        A cached :class:`~diffusers.StableDiffusionInstructPix2PixPipeline` on the GPU.

    Raises:
        RuntimeError: If no CUDA GPU is available (the pipe loads in float16 on cuda).
    """
    if not torch.cuda.is_available():
        raise RuntimeError(
            "eerie requires a CUDA GPU (InstructPix2Pix loads in float16 on cuda)."
        )
    pipe = StableDiffusionInstructPix2PixPipeline.from_pretrained(
        STYLE_MODEL_ID, torch_dtype=torch.float16, safety_checker=None
    )
    pipe = pipe.to(_DEVICE)
    pipe.scheduler = EulerAncestralDiscreteScheduler.from_config(pipe.scheduler.config)
    pipe.enable_attention_slicing()
    pipe.enable_vae_slicing()
    return pipe
