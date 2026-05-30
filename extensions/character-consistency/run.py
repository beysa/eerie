#!/usr/bin/env python3
"""
Biscuit the Bear Cub — story-to-comic, character-consistency demo (v1).

What this proves
----------------
Character consistency on Stable Diffusion 1.5 using IPAdapter
(cubiq/ComfyUI_IPAdapter_plus). The headline is an honest A/B:

  * We generate a single character reference (character_ref.png) once.
  * For each of 4 fixed scene prompts we render the SAME panel graph
    twice, changing exactly ONE variable: the IPAdapter `weight`.
      - weight = 0.0  -> "before"  (per IPAdapter math, identical to a
                          vanilla SD1.5 render of the same prompt/seed)
      - weight = 0.8  -> "after"   (the reference character is injected)
  * Seed, prompt, checkpoint, sampler, scheduler, steps, cfg and
    resolution are IDENTICAL across before/after. The only delta is the
    weight. That is the entire point: the difference you see is caused by
    IPAdapter and nothing else.

This is NOT a cherry-picked gallery. weight=0.0 is a real control.

How it runs
-----------
Talks to a ComfyUI server over its HTTP API at HOST (default
127.0.0.1:8188, reached via an SSH tunnel to a RunPod GPU pod). It:
  1. POSTs the reference graph to /prompt, polls /history/{id},
     downloads the image via /view, and re-uploads it as the IPAdapter
     reference via /upload/image.
  2. For each scene x weight, POSTs the panel graph and saves outputs to
     outputs/before/ or outputs/after/.

The graph dicts are emitted in ComfyUI's "API format" (the format you get
from the web UI via "Save (API Format)"). The node class names and input
fields below were taken from the live ComfyUI_IPAdapter_plus source
(IPAdapterPlus.py, NODE_CLASS_MAPPINGS) on 2026-05-30 — see manifest.yaml
and README for the pinned commit captured on the pod.

`python run.py --export` writes workflows/panel_api.json and
workflows/reference_api.json (this script is the source of truth for them).
"""

from __future__ import annotations

import argparse
import io
import json
import os
import sys
import time
import urllib.parse
import urllib.request
import uuid

# --------------------------------------------------------------------------
# Constants
# --------------------------------------------------------------------------

HOST = os.environ.get("COMFY_HOST", "127.0.0.1:8188")  # SSH tunnel -> pod
CLIENT_ID = str(uuid.uuid4())

# Verified model filenames (repo + filename recorded in manifest.yaml).
# These are the names ComfyUI sees in its models/ subfolders, NOT URLs.
CKPT_NAME = "dreamshaper_8.safetensors"               # models/checkpoints/
IPADAPTER_FILE = "ip-adapter-plus_sd15.safetensors"   # models/ipadapter/
CLIP_VISION_NAME = "CLIP-ViT-H-14-laion2B-s32B-b79K.safetensors"  # models/clip_vision/
# ^ This is the renamed h94/IP-Adapter image_encoder/model.safetensors.
#   manifest.yaml documents the source file and the rename. We use the
#   explicit IPAdapterModelLoader + CLIPVisionLoader pair (not the Unified
#   Loader) so the exact files are pinned and visible.

# IPAdapter weights for the A/B. 0.0 is the control (== vanilla SD1.5).
WEIGHT_BEFORE = 0.0
WEIGHT_AFTER = 0.8

# Reference image filename (produced once, then uploaded to the pod).
REF_FILENAME = "character_ref.png"

NEG_PROMPT = (
    "photo, realistic, photograph, 3d render, cgi, deformed, extra limbs, "
    "text, watermark, signature, blurry, low quality, lowres"
)

# Character sheet used only to MAKE the reference (text2img, no IPAdapter).
REFERENCE_PROMPT = (
    "character reference sheet of Biscuit, a small round brown bear cub with "
    "a cream-colored belly, big dark round eyes, a tiny black nose, and a red "
    "knitted scarf, storybook illustration, soft painterly children's book "
    "art, gentle watercolor and gouache, neutral grey background, full body, "
    "centered"
)

# 4 fixed scenes. The story bridges 'eerie': cozy turning quietly uncanny.
SCENES = [
    {
        "name": "01_porch",
        "prompt": (
            "Biscuit the bear cub sitting on a wooden cabin porch at dusk, "
            "holding a small lantern, warm cozy storybook illustration, soft "
            "painterly children's book art, gentle watercolor and gouache"
        ),
        "seed": 111111,
    },
    {
        "name": "02_forest_path",
        "prompt": (
            "Biscuit the bear cub walking down a misty forest path between "
            "tall dark pines, lantern glowing, slightly eerie quiet mood, "
            "storybook illustration, soft painterly children's book art"
        ),
        "seed": 222222,
    },
    {
        "name": "03_old_door",
        "prompt": (
            "Biscuit the bear cub standing before a crooked old wooden door in "
            "a hollow tree, faint glow leaking through the cracks, uneasy "
            "curious mood, storybook illustration, soft painterly art"
        ),
        "seed": 333333,
    },
    {
        "name": "04_clearing",
        "prompt": (
            "Biscuit the bear cub alone in a moonlit clearing surrounded by "
            "pale floating fireflies, wide quiet uncanny atmosphere, storybook "
            "illustration, soft painterly children's book art, muted palette"
        ),
        "seed": 444444,
    },
]

# Shared sampler settings — identical for reference and every panel.
STEPS = 30
CFG = 7.0
SAMPLER = "dpmpp_2m"
SCHEDULER = "karras"
WIDTH = 512
HEIGHT = 512
REFERENCE_SEED = 770077


# --------------------------------------------------------------------------
# ComfyUI HTTP API helpers
# --------------------------------------------------------------------------

def _post(path: str, payload: dict) -> dict:
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        f"http://{HOST}{path}", data=data,
        headers={"Content-Type": "application/json"},
    )
    try:
        with urllib.request.urlopen(req) as resp:
            return json.loads(resp.read())
    except urllib.error.HTTPError as e:
        body = e.read().decode("utf-8", "replace")
        raise RuntimeError(f"POST {path} -> HTTP {e.code}: {body}") from None


def _get(path: str) -> bytes:
    with urllib.request.urlopen(f"http://{HOST}{path}") as resp:
        return resp.read()


def queue_prompt(graph: dict) -> str:
    """POST a graph to /prompt; return the prompt_id."""
    res = _post("/prompt", {"prompt": graph, "client_id": CLIENT_ID})
    if "prompt_id" not in res:
        raise RuntimeError(f"/prompt rejected the graph: {res}")
    return res["prompt_id"]


def wait_done(prompt_id: str, timeout: float = 600.0, interval: float = 1.5) -> dict:
    """Poll /history/{id} until the prompt finishes; return its history entry."""
    deadline = time.time() + timeout
    while time.time() < deadline:
        hist = json.loads(_get(f"/history/{prompt_id}"))
        if prompt_id in hist:
            entry = hist[prompt_id]
            status = entry.get("status", {})
            if status.get("completed") or status.get("status_str") == "success":
                return entry
            if status.get("status_str") == "error":
                raise RuntimeError(f"Execution error for {prompt_id}: {status}")
        time.sleep(interval)
    raise TimeoutError(f"Timed out waiting for prompt {prompt_id}")


def pull_outputs(history_entry: dict, out_dir: str) -> list[str]:
    """Download every image in a history entry's outputs via /view."""
    os.makedirs(out_dir, exist_ok=True)
    saved: list[str] = []
    for node_out in history_entry.get("outputs", {}).values():
        for img in node_out.get("images", []):
            q = urllib.parse.urlencode({
                "filename": img["filename"],
                "subfolder": img.get("subfolder", ""),
                "type": img.get("type", "output"),
            })
            blob = _get(f"/view?{q}")
            dst = os.path.join(out_dir, img["filename"])
            with open(dst, "wb") as fh:
                fh.write(blob)
            saved.append(dst)
    return saved


def upload_image(image_bytes: bytes, name: str, overwrite: bool = True) -> str:
    """Upload bytes to ComfyUI's input dir via /upload/image (multipart).

    Returns the server-side filename to reference in a LoadImage node.
    """
    boundary = uuid.uuid4().hex
    buf = io.BytesIO()

    def field(name_, value):
        buf.write(f"--{boundary}\r\n".encode())
        buf.write(f'Content-Disposition: form-data; name="{name_}"\r\n\r\n'.encode())
        buf.write(f"{value}\r\n".encode())

    buf.write(f"--{boundary}\r\n".encode())
    buf.write(
        f'Content-Disposition: form-data; name="image"; filename="{name}"\r\n'.encode()
    )
    buf.write(b"Content-Type: image/png\r\n\r\n")
    buf.write(image_bytes)
    buf.write(b"\r\n")
    field("overwrite", "true" if overwrite else "false")
    buf.write(f"--{boundary}--\r\n".encode())

    req = urllib.request.Request(
        f"http://{HOST}/upload/image", data=buf.getvalue(),
        headers={"Content-Type": f"multipart/form-data; boundary={boundary}"},
    )
    with urllib.request.urlopen(req) as resp:
        res = json.loads(resp.read())
    return res["name"]


# --------------------------------------------------------------------------
# Graph builders (ComfyUI API format)
# --------------------------------------------------------------------------

def build_reference_graph() -> dict:
    """text2img graph that makes character_ref.png once. No IPAdapter."""
    return {
        "4": {
            "class_type": "CheckpointLoaderSimple",
            "inputs": {"ckpt_name": CKPT_NAME},
        },
        "5": {
            "class_type": "EmptyLatentImage",
            "inputs": {"width": WIDTH, "height": HEIGHT, "batch_size": 1},
        },
        "6": {
            "class_type": "CLIPTextEncode",
            "inputs": {"text": REFERENCE_PROMPT, "clip": ["4", 1]},
        },
        "7": {
            "class_type": "CLIPTextEncode",
            "inputs": {"text": NEG_PROMPT, "clip": ["4", 1]},
        },
        "3": {
            "class_type": "KSampler",
            "inputs": {
                "seed": REFERENCE_SEED,
                "steps": STEPS,
                "cfg": CFG,
                "sampler_name": SAMPLER,
                "scheduler": SCHEDULER,
                "denoise": 1.0,
                "model": ["4", 0],
                "positive": ["6", 0],
                "negative": ["7", 0],
                "latent_image": ["5", 0],
            },
        },
        "8": {
            "class_type": "VAEDecode",
            "inputs": {"samples": ["3", 0], "vae": ["4", 2]},
        },
        "9": {
            "class_type": "SaveImage",
            "inputs": {"filename_prefix": "character_ref", "images": ["8", 0]},
        },
    }


def build_panel_graph(scene_prompt: str, seed: int, ipadapter_weight: float,
                      ref_image_name: str = REF_FILENAME) -> dict:
    """One panel. The ONLY variable that should change between the A/B
    renders is `ipadapter_weight` (0.0 = before, 0.8 = after).

    Uses IPAdapterModelLoader + CLIPVisionLoader + IPAdapterAdvanced so the
    exact pinned files are explicit (vs. the Unified Loader's presets).
    """
    return {
        "4": {
            "class_type": "CheckpointLoaderSimple",
            "inputs": {"ckpt_name": CKPT_NAME},
        },
        "10": {
            "class_type": "IPAdapterModelLoader",
            "inputs": {"ipadapter_file": IPADAPTER_FILE},
        },
        "11": {
            "class_type": "CLIPVisionLoader",
            "inputs": {"clip_name": CLIP_VISION_NAME},
        },
        "12": {
            "class_type": "LoadImage",
            "inputs": {"image": ref_image_name},
        },
        "13": {
            "class_type": "IPAdapterAdvanced",
            "inputs": {
                "weight": ipadapter_weight,
                "weight_type": "linear",
                "combine_embeds": "concat",
                "start_at": 0.0,
                "end_at": 1.0,
                "embeds_scaling": "V only",
                "model": ["4", 0],
                "ipadapter": ["10", 0],
                "image": ["12", 0],
                "clip_vision": ["11", 0],
            },
        },
        "5": {
            "class_type": "EmptyLatentImage",
            "inputs": {"width": WIDTH, "height": HEIGHT, "batch_size": 1},
        },
        "6": {
            "class_type": "CLIPTextEncode",
            "inputs": {"text": scene_prompt, "clip": ["4", 1]},
        },
        "7": {
            "class_type": "CLIPTextEncode",
            "inputs": {"text": NEG_PROMPT, "clip": ["4", 1]},
        },
        "3": {
            "class_type": "KSampler",
            "inputs": {
                "seed": seed,
                "steps": STEPS,
                "cfg": CFG,
                "sampler_name": SAMPLER,
                "scheduler": SCHEDULER,
                "denoise": 1.0,
                "model": ["13", 0],          # model AFTER IPAdapter patch
                "positive": ["6", 0],
                "negative": ["7", 0],
                "latent_image": ["5", 0],
            },
        },
        "8": {
            "class_type": "VAEDecode",
            "inputs": {"samples": ["3", 0], "vae": ["4", 2]},
        },
        "9": {
            "class_type": "SaveImage",
            "inputs": {"filename_prefix": "panel", "images": ["8", 0]},
        },
    }


# --------------------------------------------------------------------------
# Orchestration
# --------------------------------------------------------------------------

def generate_reference() -> str:
    """Render the reference, download it, re-upload as the IPAdapter input.
    Returns the server-side filename to feed LoadImage in the panel graph."""
    print("[ref] generating character reference ...")
    entry = wait_done(queue_prompt(build_reference_graph()))
    saved = pull_outputs(entry, os.path.join("outputs", "reference"))
    if not saved:
        raise RuntimeError("reference produced no image")
    local = saved[0]
    print(f"[ref] saved {local}; uploading as {REF_FILENAME}")
    with open(local, "rb") as fh:
        return upload_image(fh.read(), REF_FILENAME)


def run_scene(scene: dict, weight: float, out_dir: str, ref_name: str) -> None:
    graph = build_panel_graph(scene["prompt"], scene["seed"], weight, ref_name)
    entry = wait_done(queue_prompt(graph))
    saved = pull_outputs(entry, out_dir)
    tag = f"{scene['name']} w={weight}"
    print(f"[panel] {tag} -> {', '.join(saved) if saved else 'NO OUTPUT'}")


def main() -> None:
    ref_name = generate_reference()
    for scene in SCENES:
        run_scene(scene, WEIGHT_BEFORE, os.path.join("outputs", "before"), ref_name)
        run_scene(scene, WEIGHT_AFTER, os.path.join("outputs", "after"), ref_name)
    print("\nDone. Compare outputs/before/ (weight=0.0, vanilla) vs "
          "outputs/after/ (weight=0.8, IPAdapter).")


def export_workflows() -> None:
    """Write the API-format graphs to workflows/. run.py is the source of truth."""
    os.makedirs("workflows", exist_ok=True)
    ref = build_reference_graph()
    panel = build_panel_graph(SCENES[0]["prompt"], SCENES[0]["seed"], WEIGHT_AFTER)
    with open(os.path.join("workflows", "reference_api.json"), "w") as fh:
        json.dump(ref, fh, indent=2)
    with open(os.path.join("workflows", "panel_api.json"), "w") as fh:
        json.dump(panel, fh, indent=2)
    print("wrote workflows/reference_api.json and workflows/panel_api.json")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Biscuit story-to-comic A/B driver")
    ap.add_argument("--export", action="store_true",
                    help="write workflows/*.json from the builders and exit")
    args = ap.parse_args()
    if args.export:
        export_workflows()
        sys.exit(0)
    main()
