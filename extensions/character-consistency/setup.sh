#!/usr/bin/env bash
# setup.sh -- provision a RunPod ComfyUI pod for the Biscuit story-to-comic demo.
#
# What it does:
#   1. git clone the required custom node(s) into ComfyUI/custom_nodes
#   2. download the verified model files into the right ComfyUI/models/* dirs
#      (filenames taken from manifest.yaml -- the verified repo+file pairs)
#
# The optional ControlNet (Canny) pieces -- a second custom node and the Canny
# model -- are fetched ONLY when WITH_CONTROLNET=1, because they are a secondary
# experiment and not part of the headline IPAdapter A/B.
#
# Run this ON THE POD, after ComfyUI is installed. Restart ComfyUI afterwards so
# it loads the new custom nodes.
#
# Usage:
#   COMFYUI_DIR=/workspace/ComfyUI bash setup.sh
#   COMFYUI_DIR=/workspace/ComfyUI WITH_CONTROLNET=1 bash setup.sh
#
# Requires: git, and the Hugging Face CLI (`huggingface-cli`, from the
# huggingface_hub package). If huggingface-cli is missing the script installs it
# with pip. All four model files are public (no HF token / auth needed).

set -euo pipefail

# --------------------------------------------------------------------------
# Config
# --------------------------------------------------------------------------
# ComfyUI install dir. Default assumes this repo sits NEXT TO ComfyUI/ in the
# pod workspace (e.g. /workspace/ComfyUI and /workspace/story-to-comic).
COMFYUI_DIR="${COMFYUI_DIR:-../ComfyUI}"
WITH_CONTROLNET="${WITH_CONTROLNET:-0}"

if [ ! -d "${COMFYUI_DIR}" ]; then
  echo "ERROR: COMFYUI_DIR='${COMFYUI_DIR}' does not exist." >&2
  echo "       Set COMFYUI_DIR to your ComfyUI install, e.g." >&2
  echo "       COMFYUI_DIR=/workspace/ComfyUI bash setup.sh" >&2
  exit 1
fi

# Resolve to an absolute path so the rest of the script is location-independent.
COMFYUI_DIR="$(cd "${COMFYUI_DIR}" && pwd)"
NODES_DIR="${COMFYUI_DIR}/custom_nodes"
CKPT_DIR="${COMFYUI_DIR}/models/checkpoints"
IPADAPTER_DIR="${COMFYUI_DIR}/models/ipadapter"
CLIPVISION_DIR="${COMFYUI_DIR}/models/clip_vision"
CONTROLNET_DIR="${COMFYUI_DIR}/models/controlnet"

echo "==> ComfyUI dir: ${COMFYUI_DIR}"
mkdir -p "${NODES_DIR}" "${CKPT_DIR}" "${IPADAPTER_DIR}" "${CLIPVISION_DIR}"

# --------------------------------------------------------------------------
# Tooling: ensure huggingface-cli is available
# --------------------------------------------------------------------------
if ! command -v hf >/dev/null 2>&1; then
  echo "==> hf CLI not found; installing huggingface_hub"
  pip install -q -U "huggingface_hub>=0.26"
fi

# clone <repo-url> <dest> -- clone if absent, else report it's already there.
clone() {
  url="$1"; dest="$2"
  if [ -d "${dest}/.git" ]; then
    echo "    already present: ${dest}"
  else
    echo "==> git clone ${url}"
    git clone --depth 1 "${url}" "${dest}"
  fi
}

# hf_get <repo> <path-in-repo> <dest-dir> <dest-filename>
# Downloads one file from a HF repo and places it at <dest-dir>/<dest-filename>,
# renaming if the source basename differs (needed for the CLIP-Vision encoder).
hf_get() {
  repo="$1"; src_path="$2"; dest_dir="$3"; dest_name="$4"
  mkdir -p "${dest_dir}"
  if [ -f "${dest_dir}/${dest_name}" ]; then
    echo "    already present: ${dest_dir}/${dest_name}"
    return
  fi
  echo "==> download ${repo} :: ${src_path}"
  hf download "${repo}" "${src_path}" --local-dir "${dest_dir}"
  downloaded="${dest_dir}/${src_path}"
  if [ "${downloaded}" != "${dest_dir}/${dest_name}" ]; then
    # HF preserves the repo subpath (e.g. models/...); flatten + rename it.
    mv "${downloaded}" "${dest_dir}/${dest_name}"
    # prune any now-empty subdirs the download created
    find "${dest_dir}" -type d -empty -delete 2>/dev/null || true
  fi
  echo "    -> ${dest_dir}/${dest_name}"
}

# --------------------------------------------------------------------------
# 1. Custom nodes
# --------------------------------------------------------------------------
echo ""
echo "### Custom nodes"
# Required: provides IPAdapterModelLoader / IPAdapterAdvanced (CLIPVisionLoader
# is a ComfyUI core node). Pin the commit in manifest.yaml after cloning.
clone "https://github.com/cubiq/ComfyUI_IPAdapter_plus" \
      "${NODES_DIR}/ComfyUI_IPAdapter_plus"

if [ "${WITH_CONTROLNET}" = "1" ]; then
  # Optional: Canny/preprocessors for the secondary ControlNet experiment only.
  clone "https://github.com/Fannovel16/comfyui_controlnet_aux" \
        "${NODES_DIR}/comfyui_controlnet_aux"
fi

# --------------------------------------------------------------------------
# 2. Models (verified repo + filename pairs from manifest.yaml)
# --------------------------------------------------------------------------
echo ""
echo "### Models"

# Base checkpoint -- DreamShaper 8 (SD1.5). Single-file checkpoint mirrored at
# fofr/comfyui (upstream author Lykon, Civitai 4384). Public, no auth.
hf_get "fofr/comfyui" "checkpoints/dreamshaper_8.safetensors" \
       "${CKPT_DIR}" "dreamshaper_8.safetensors"

# IP-Adapter Plus (SD1.5) -- the 'plus' model, NOT plus-face.
hf_get "h94/IP-Adapter" "models/ip-adapter-plus_sd15.safetensors" \
       "${IPADAPTER_DIR}" "ip-adapter-plus_sd15.safetensors"

# CLIP-ViT-H image encoder for IP-Adapter. Source file is
# models/image_encoder/model.safetensors; RENAMED to a self-describing name so
# CLIPVisionLoader lists it clearly.
hf_get "h94/IP-Adapter" "models/image_encoder/model.safetensors" \
       "${CLIPVISION_DIR}" "CLIP-ViT-H-14-laion2B-s32B-b79K.safetensors"

if [ "${WITH_CONTROLNET}" = "1" ]; then
  echo ""
  echo "### Optional: ControlNet v1.1 Canny (SD1.5, fp16) -- secondary experiment"
  mkdir -p "${CONTROLNET_DIR}"
  hf_get "comfyanonymous/ControlNet-v1-1_fp16_safetensors" \
         "control_v11p_sd15_canny_fp16.safetensors" \
         "${CONTROLNET_DIR}" "control_v11p_sd15_canny_fp16.safetensors"
fi

# --------------------------------------------------------------------------
# Done
# --------------------------------------------------------------------------
echo ""
echo "==> setup complete."
echo "    Restart ComfyUI so it loads the new custom node(s)."
echo ""
echo "    Then record the EXACT pinned versions into manifest.yaml"
echo "    (replace the TO_FILL_ON_POD placeholders -- do not invent hashes):"
echo "      git -C '${NODES_DIR}/ComfyUI_IPAdapter_plus' rev-parse HEAD"
if [ "${WITH_CONTROLNET}" = "1" ]; then
  echo "      git -C '${NODES_DIR}/comfyui_controlnet_aux' rev-parse HEAD"
fi
echo "      sha256sum '${CKPT_DIR}/dreamshaper_8.safetensors'"
echo "      sha256sum '${IPADAPTER_DIR}/ip-adapter-plus_sd15.safetensors'"
echo "      sha256sum '${CLIPVISION_DIR}/CLIP-ViT-H-14-laion2B-s32B-b79K.safetensors'"
if [ "${WITH_CONTROLNET}" = "1" ]; then
  echo "      sha256sum '${CONTROLNET_DIR}/control_v11p_sd15_canny_fp16.safetensors'"
fi
