"""Multi-style SD1.5 explorer (EXTENSION — not the thesis pipeline).

This reproduces eerie's *original* art-style behavior, preserved as a labeled
post-thesis extension. Instead of the thesis image model (``ektvho/sd-vist``,
Stable Diffusion 2.1 fine-tuned on VIST), it renders each scene by **swapping a
community Stable Diffusion 1.5 art-style checkpoint** — Ghibli, modern animation
(Disney-like), or watercolor (see :mod:`.styles`).

It is **not** the thesis: the candidate's fine-tuned models are
``ektvho/bart-cnn`` and ``ektvho/sd-vist``. Use :func:`eerie.run_pipeline` for
the thesis flow; use :func:`run_multi_style` here only to explore community SD1.5
styles.

Self-contained image loader: the core :mod:`eerie.models` only loads the single
thesis image model, so this extension has its own ``_get_sd15_pipe`` keyed on the
checkpoint id (each art-style fine-tune cached separately, ``float16`` on CUDA).
Scene splitting, optional summarization, and the optional InstructPix2Pix restyle
are reused from the core package — those stages are model-agnostic / off-the-shelf.

Honesty note: panels are generated independently with a pinned seed for
reproducibility. There is no cross-panel character consistency mechanism.
"""

import os
import re
from functools import lru_cache
from typing import Dict, List, Optional, Sequence

import torch
from diffusers import StableDiffusionPipeline
from PIL.Image import Image

from eerie.scenes import split_into_scenes
from eerie.summarize import summarize as summarize_text
from eerie.transfer import apply_style

from .styles import StylePreset, get_style

_DEVICE = "cuda"


@lru_cache(maxsize=None)
def _get_sd15_pipe(model_id: str) -> StableDiffusionPipeline:
    """Load and cache a community Stable Diffusion 1.5 pipeline by checkpoint id.

    Keyed on ``model_id`` so each art-style fine-tune (Ghibli, modern animation,
    watercolor) is loaded once and cached separately. Loaded in ``float16`` on
    CUDA with attention/vae slicing, matching the core image loader's settings.

    Args:
        model_id: Hugging Face SD1.5 checkpoint to load (a ``StylePreset.model_id``).

    Returns:
        A cached :class:`~diffusers.StableDiffusionPipeline` on the GPU.

    Raises:
        RuntimeError: If no CUDA GPU is available (the pipe loads in float16 on cuda).
    """
    if not torch.cuda.is_available():
        raise RuntimeError(
            f"This extension requires a CUDA GPU ({model_id} loads in float16 on cuda)."
        )
    pipe = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.float16)
    pipe = pipe.to(_DEVICE)
    pipe.enable_attention_slicing()
    pipe.enable_vae_slicing()
    return pipe


def generate_panels(
    scenes: Sequence[str],
    preset: StylePreset,
    seed: int = 42,
) -> List[Image]:
    """Generate one storybook panel per scene in the preset's SD1.5 art style.

    The preset's checkpoint (``preset.model_id``) is loaded for generation, and
    for each scene the prompt is ``"{scene}, {preset.prompt_suffix}"`` with
    ``preset.negative_prompt`` applied. A fresh CUDA generator seeded with
    ``seed + index`` is passed so the run is reproducible while keeping per-panel
    variety.

    Args:
        scenes: Scene strings (typically from :func:`eerie.scenes.split_into_scenes`).
        preset: Art-style preset selecting the SD1.5 checkpoint, prompt suffix,
            and negative prompt.
        seed: Base seed; panel ``i`` uses ``seed + i`` — reproducible yet distinct.

    Returns:
        A list of PIL images, one per scene, in order.
    """
    pipe = _get_sd15_pipe(preset.model_id)
    panels: List[Image] = []
    for i, scene in enumerate(scenes):
        prompt = f"{scene}, {preset.prompt_suffix}"
        generator = torch.Generator("cuda").manual_seed(seed + i)
        image = pipe(
            prompt,
            negative_prompt=preset.negative_prompt,
            generator=generator,
        ).images[0]
        panels.append(image)
    return panels


def run_multi_style(
    story: str,
    art_style: str = "ghibli",
    style: Optional[str] = None,
    summarize: bool = False,
    seed: int = 42,
    out_dir: str = "output",
) -> Dict[str, object]:
    """Render a story with a community SD1.5 art-style checkpoint (EXTENSION).

    This is eerie's original behavior, preserved as an extension. It does **not**
    use the thesis models — the image stage swaps a community SD1.5 checkpoint
    selected by ``art_style``. For the thesis pipeline (``ektvho/bart-cnn`` +
    ``ektvho/sd-vist``) use :func:`eerie.run_pipeline` instead.

    Args:
        story: The children's story prose.
        art_style: Which community SD1.5 checkpoint to render the panels with — a
            :class:`~extensions.multi_style.styles.StylePreset` id (``"ghibli"``,
            ``"modern_animation"``, or ``"watercolor"``). Defaults to ``"ghibli"``.
            Raises ``ValueError`` on an unknown id.
        style: Optional InstructPix2Pix edit instruction applied after generation
            (e.g. ``"picasso"``). Independent of ``art_style``; if ``None`` the
            post-hoc restyle stage is skipped and ``styled_paths`` is empty.
        summarize: If ``True``, condense the story with the fine-tuned BART before
            splitting into scenes. Defaults to ``False`` (illustrate the full
            story), matching eerie's original art-style behavior.
        seed: Base seed pinned on every generated panel for reproducibility.
        out_dir: Directory to write panels into; created if it does not exist.

    Returns:
        A dict with:

        * ``"scenes"`` (``List[str]``): the scene texts that were illustrated.
        * ``"panel_paths"`` (``List[str]``): saved generated panels,
          ``{out_dir}/panel_{i}_{art_style}.png`` (tagged by ``art_style`` so runs
          in different styles do not overwrite each other).
        * ``"styled_paths"`` (``List[str]``): saved restyled panels,
          ``{out_dir}/panel_{i}_styled_{style}.png`` (empty if ``style`` is ``None``).
    """
    os.makedirs(out_dir, exist_ok=True)

    # Resolve the art-style preset up front so an unknown id fails fast.
    preset = get_style(art_style)

    # 1. Optionally condense, then 2. split into one scene per sentence.
    text = summarize_text(story) if summarize else story
    scenes = split_into_scenes(text)

    # 3. Illustrate each scene with the selected SD1.5 checkpoint (seed pinned).
    panels = generate_panels(scenes, preset=preset, seed=seed)
    panel_paths: List[str] = []
    for i, panel in enumerate(panels):
        path = os.path.join(out_dir, f"panel_{i}_{art_style}.png")
        panel.save(path)
        panel_paths.append(path)

    # 4. Optionally restyle every panel (separate InstructPix2Pix instruction).
    styled_paths: List[str] = []
    if style is not None:
        styled = apply_style(panels, instruction=style)
        slug = re.sub(r"[^A-Za-z0-9._-]+", "_", style).strip("_")[:64] or "styled"
        for i, image in enumerate(styled):
            path = os.path.join(out_dir, f"panel_{i}_styled_{slug}.png")
            image.save(path)
            styled_paths.append(path)

    return {
        "scenes": scenes,
        "panel_paths": panel_paths,
        "styled_paths": styled_paths,
    }
