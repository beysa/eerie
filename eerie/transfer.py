"""Optional per-panel style transfer with InstructPix2Pix.

Distilled from the notebook's "Style Transfer" cell. ``timbrooks/instruct-pix2pix``
is image-conditioned: it edits an existing panel from a text instruction while
preserving composition, so the same panels can be re-rendered into any art style
by changing one instruction.

The original style-transfer parameters are preserved exactly:
``num_inference_steps=150`` and ``image_guidance_scale=1``.

Like generation, each panel is restyled independently with no cross-panel
coupling.
"""

from typing import List, Sequence

from PIL.Image import Image

from eerie.models import get_style_pipe


def apply_style(
    images: Sequence[Image],
    instruction: str,
    num_inference_steps: int = 150,
    image_guidance_scale: float = 1,
) -> List[Image]:
    """Restyle each panel according to a text ``instruction``.

    Args:
        images: Panels to restyle (e.g. the output of
            :func:`eerie.generate.generate_panels`).
        instruction: The edit instruction / style, passed positionally to
            InstructPix2Pix (e.g. ``"picasso"`` or ``"make it a watercolor painting"``).
        num_inference_steps: Diffusion steps per panel. Defaults to ``150`` (notebook value).
        image_guidance_scale: How strongly to preserve the source image. Defaults
            to ``1`` (notebook value).

    Returns:
        A list of restyled PIL images, one per input panel, in order.
    """
    pipe = get_style_pipe()
    styled: List[Image] = []
    for image in images:
        result = pipe(
            instruction,
            image=image,
            num_inference_steps=num_inference_steps,
            image_guidance_scale=image_guidance_scale,
        ).images[0]
        styled.append(result)
    return styled
