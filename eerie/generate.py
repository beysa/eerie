"""Illustrate scenes as storybook panels with the thesis image model.

This is the thesis's "Image Generation" stage. Each scene becomes one panel,
generated independently by ``ektvho/sd-vist`` — the candidate's Stable Diffusion
2.1 fine-tune on the VIST (Visual Storytelling) dataset (see
:func:`eerie.models.get_image_pipe`). There is a single image model in the core
pipeline; art-style *selection* (swapping SD1.5 checkpoints) is a post-thesis
extension and lives in ``extensions/multi_style/``, not here.

Two improvements over the notebook are preserved:

* **Seed pinning.** The notebook used no seed, so panels were non-reproducible.
  Here each panel uses a distinct but deterministic seed (``seed + index``) via
  ``torch.Generator("cuda").manual_seed(seed + i)``, so a run is reproducible
  without every panel sharing identical initial noise.
* **Honest storybook prompt suffix.** Each scene is suffixed with a neutral
  ``"children's storybook illustration"`` hint to steer the model toward the
  picture-book look the thesis targets.

Because ``ektvho/sd-vist`` is Stable Diffusion 2.1, panels default to SD2.1's
native 768x768.

Honesty note: panels are still generated independently. The shared seed makes
runs reproducible but does **not** give cross-panel character consistency — the
same character will look different from panel to panel.
"""

from typing import List, Sequence

import torch
from PIL.Image import Image

from eerie.models import get_image_pipe

# Neutral storybook hint appended to every scene prompt (honest, not style-faking).
STORYBOOK_SUFFIX = "children's storybook illustration"


def generate_panels(
    scenes: Sequence[str],
    seed: int = 42,
    width: int = 768,
    height: int = 768,
) -> List[Image]:
    """Generate one storybook panel per scene with ``ektvho/sd-vist``.

    For each scene the prompt is ``"{scene}, {STORYBOOK_SUFFIX}"`` and a fresh
    CUDA generator seeded with ``seed + index`` is passed, so the run is
    reproducible while keeping per-panel variety.

    Args:
        scenes: Scene strings (typically from :func:`eerie.scenes.split_into_scenes`).
        seed: Base seed; panel ``i`` uses ``seed + i`` — reproducible yet distinct.
        width: Panel width in pixels. Defaults to ``768`` (SD2.1 native resolution).
        height: Panel height in pixels. Defaults to ``768`` (SD2.1 native resolution).

    Returns:
        A list of PIL images, one per scene, in order.
    """
    pipe = get_image_pipe()
    panels: List[Image] = []
    for i, scene in enumerate(scenes):
        prompt = f"{scene}, {STORYBOOK_SUFFIX}"
        generator = torch.Generator("cuda").manual_seed(seed + i)
        image = pipe(
            prompt,
            width=width,
            height=height,
            generator=generator,
        ).images[0]
        panels.append(image)
    return panels
