"""End-to-end orchestration: story in, illustrated panels out (thesis flow).

This module runs the real thesis pipeline from ``eerie.ipynb`` end to end, on the
candidate's own fine-tuned models::

    story -> (summarize, default on) ektvho/bart-cnn  -> split_into_scenes
          -> generate_panels (ektvho/sd-vist)         -> panel_paths
          -> (optional) instruct-pix2pix restyle      -> styled_paths
          -> save to out_dir -> return paths + scenes

It also fixes the central bug in the original demo notebook, where the BART
summary was computed but never consumed and the image stage re-read its own
hardcoded story string. Here the stages are a single data flow.

The core uses exactly two fine-tuned thesis models: ``ektvho/bart-cnn`` for
summarization and ``ektvho/sd-vist`` (Stable Diffusion 2.1, VIST) for image
generation. ``timbrooks/instruct-pix2pix`` provides the optional post-hoc
restyle. There is **no** ``art_style`` argument in the core: selecting among
community SD1.5 art-style checkpoints is a post-thesis exploration that lives in
``extensions/multi_style/`` (see ``run_multi_style``), kept clearly separate from
the thesis models.

Honesty note: panels are generated independently with a pinned seed for
reproducibility. There is no cross-panel character consistency mechanism.
"""

import os
import re
from typing import Dict, List, Optional

from eerie.generate import generate_panels
from eerie.scenes import split_into_scenes
from eerie.summarize import summarize as summarize_text
from eerie.transfer import apply_style


def run_pipeline(
    story: str,
    summarize: bool = True,
    style: Optional[str] = None,
    seed: int = 42,
    out_dir: str = "output",
) -> Dict[str, object]:
    """Run the full thesis eerie pipeline on a story and save the panels to disk.

    Args:
        story: The children's story prose.
        summarize: If ``True`` (the default — the thesis summarizes first),
            condense the story with the fine-tuned ``ektvho/bart-cnn`` before
            splitting it into scenes. Set ``False`` to illustrate the full story
            sentence-by-sentence without summarizing.
        style: Optional InstructPix2Pix edit instruction applied *after*
            generation (e.g. ``"picasso"``). Independent of the core models; if
            ``None`` the post-hoc restyle stage is skipped and ``styled_paths`` is
            empty.
        seed: Base seed pinned on every generated panel for reproducibility.
        out_dir: Directory to write panels into; created if it does not exist.

    Returns:
        A dict with:

        * ``"scenes"`` (``List[str]``): the scene texts that were illustrated.
        * ``"panel_paths"`` (``List[str]``): saved generated panels,
          ``{out_dir}/panel_{i}.png``.
        * ``"styled_paths"`` (``List[str]``): saved restyled panels,
          ``{out_dir}/panel_{i}_styled_{style}.png`` — a separate namespace from the
          generated panels, so the two never collide (empty if ``style`` is ``None``).
    """
    os.makedirs(out_dir, exist_ok=True)

    # 1. Optionally condense the story with the fine-tuned BART (thesis default),
    #    then 2. split into one scene per sentence.
    text = summarize_text(story) if summarize else story
    scenes = split_into_scenes(text)

    # 3. Illustrate each scene as a panel with ektvho/sd-vist (seed pinned).
    panels = generate_panels(scenes, seed=seed)
    panel_paths: List[str] = []
    for i, panel in enumerate(panels):
        path = os.path.join(out_dir, f"panel_{i}.png")
        panel.save(path)
        panel_paths.append(path)

    # 4. Optionally restyle every panel (separate InstructPix2Pix instruction)
    #    and save under a style-tagged name.
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
