"""Art-style presets for native SD1.5 checkpoint swapping (EXTENSION).

This is a **post-thesis extension**, not part of the thesis. The thesis core
generates images with the candidate's fine-tuned ``ektvho/sd-vist`` (Stable
Diffusion 2.1, VIST). This module instead illustrates scenes by swapping a
*community Stable Diffusion 1.5* checkpoint — a style exploration that is never
implied to be the thesis model.

Each preset names a community SD1.5 fine-tune on Hugging Face plus the prompt
tokens that activate its look, so :func:`run_multi_style` can render the same
story in a different art style by loading a different model. Every preset shares
one quality negative prompt (:data:`SHARED_NEGATIVE_PROMPT`).

Honesty note: these are community fine-tunes, not official studio models.
``modern_animation`` (``nitrosocke/mo-di-diffusion``) produces a Disney-*like*
look and is **not** affiliated with or endorsed by any studio; it is labelled
"Modern animation (Disney-like)" deliberately and is never called "Pixar".
"""

from dataclasses import dataclass
from typing import Dict, List

# Shared across every preset — keeps quality consistent regardless of checkpoint.
SHARED_NEGATIVE_PROMPT = "deformed, blurry, low quality, text, watermark"


@dataclass(frozen=True)
class StylePreset:
    """One selectable art style, backed by a Stable Diffusion 1.5 fine-tune.

    Attributes:
        style_id: Stable identifier used to select the preset (e.g. ``"ghibli"``).
        display_name: Human-readable label for UIs / discovery.
        model_id: Hugging Face SD1.5 checkpoint loaded for generation.
        prompt_suffix: Style tokens appended to every scene prompt; includes the
            checkpoint's trigger phrase where one exists.
        negative_prompt: Negative prompt passed to every generation.
    """

    style_id: str
    display_name: str
    model_id: str
    prompt_suffix: str
    negative_prompt: str = SHARED_NEGATIVE_PROMPT


# Registry of available art styles. All three are verified SD1.5 fine-tunes.
STYLES: Dict[str, StylePreset] = {
    "ghibli": StylePreset(
        style_id="ghibli",
        display_name="Ghibli",
        model_id="nitrosocke/Ghibli-Diffusion",
        prompt_suffix="ghibli style, storybook illustration",
    ),
    "modern_animation": StylePreset(
        style_id="modern_animation",
        display_name="Modern animation (Disney-like)",
        model_id="nitrosocke/mo-di-diffusion",
        prompt_suffix="modern disney style, storybook illustration",
    ),
    "watercolor": StylePreset(
        style_id="watercolor",
        display_name="Watercolor",
        model_id="ilee0022/watercolor_stable_diffusion",
        prompt_suffix="watercolor painting, storybook illustration",
    ),
}


def get_style(style_id: str) -> StylePreset:
    """Resolve a ``style_id`` to its :class:`StylePreset`.

    Args:
        style_id: One of the keys in :data:`STYLES` (e.g. ``"ghibli"``).

    Returns:
        The matching :class:`StylePreset`.

    Raises:
        ValueError: If ``style_id`` is unknown; the message lists every valid id.
    """
    try:
        return STYLES[style_id]
    except KeyError:
        valid = ", ".join(sorted(STYLES))
        raise ValueError(
            f"Unknown style_id {style_id!r}. Valid style ids are: {valid}."
        ) from None


def list_styles() -> List[StylePreset]:
    """Return all available art-style presets, for discovery in UIs.

    Returns:
        A list of every :class:`StylePreset` in :data:`STYLES`.
    """
    return list(STYLES.values())
