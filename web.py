#!/usr/bin/env python3
"""infer_simpleclick.py
---------------------------------
Run **SimpleClick** inference on a single image and save an overlay
(original image blended with the predicted mask).

This script is *headless* (no GUI) and is therefore suitable for
calling from a Streamlit backend or any batch pipeline.

Example (CPU):
    python3 infer_simpleclick.py \
        --input ./sample.jpg \
        --output ./results \
        --checkpoint ./weights/simpleclick_models/cocolvis_vit_huge.pth \
        --gpu -1

Key points
-----------
* If the checkpoint file does **not** exist, it will be downloaded
  automatically from the corresponding HuggingFace URL.
* A **single centre positive click** is generated just to demonstrate
  end‑to‑end inference. When integrating with Streamlit you can pass
  arbitrary click lists instead.
* We import SimpleClick/isegm directly from the *local* `SimpleClick/`
  folder shipped with your repo.  No need to `pip install` the package.
"""

# -------- standard libs --------
import argparse
import os, sys, pathlib, urllib.request, shutil
from pathlib import Path

# -------- third‑party --------
import cv2
import numpy as np
import torch

# -------- make local SimpleClick importable --------
PROJ_ROOT = pathlib.Path(__file__).resolve().parent
SIMPLECLICK_DIR = PROJ_ROOT / "SimpleClick"
sys.path.append(str(SIMPLECLICK_DIR))

# -------- now import SimpleClick helper libs --------
from isegm.inference import utils as pred_utils       # type: ignore
from isegm.inference.clicker import Clicker, Click   # type: ignore
from isegm.utils.vis import draw_with_blend_and_contour  # type: ignore

# Map filename → official download URL (HuggingFace release)
WEIGHT_URLS = {
    "cocolvis_vit_huge.pth": "https://huggingface.co/uncbiag/SimpleClick/resolve/main/cocolvis_vit_huge.pth",
    "cocolvis_vit_base.pth": "https://huggingface.co/uncbiag/SimpleClick/resolve/main/cocolvis_vit_base.pth",
    "cocolvis_vit_large.pth": "https://huggingface.co/uncbiag/SimpleClick/resolve/main/cocolvis_vit_large.pth",
}


def download_checkpoint(ckpt_path: Path):
    """Download weights if missing."""
    fname = ckpt_path.name
    url = WEIGHT_URLS.get(fname)
    if url is None:
        raise ValueError(f"No default URL for checkpoint '{fname}'. Please update WEIGHT_URLS dict.")

    ckpt_path.parent.mkdir(parents=True, exist_ok=True)
    print(f"[infer_simpleclick] Downloading {fname} …")
    with urllib.request.urlopen(url) as resp, open(ckpt_path, "wb") as out_file:
        shutil.copyfileobj(resp, out_file)
    print("[infer_simpleclick] Download complete →", ckpt_path)


def parse_args():
    p = argparse.ArgumentParser(description="SimpleClick one‑shot inference (no GUI)")
    p.add_argument("--input", required=True, help="Path to input PNG/JPG")
    p.add_argument("--output", required=True, help="Directory to save overlay result")
    p.add_argument("--checkpoint", required=True, help="Path to .pth model weights")
    p.add_argument("--gpu", type=int, default=-1, help="GPU id; -1 = CPU")
    p.add_argument("--model-name", default="vit_huge", help="Model tag (vit_huge / vit_large / vit_base)")
    return p.parse_args()


def main():
    args = parse_args()

    ckpt_path = Path(args.checkpoint)
    if not ckpt_path.exists():
        download_checkpoint(ckpt_path)

    device = (
        f"cuda:{args.gpu}"
        if args.gpu >= 0 and torch.cuda.is_available()
        else "cpu"
    )

    # Build predictor
    predictor = pred_utils.get_predictor(
        checkpoint_path=str(ckpt_path),
        model_name=args.model_name,  # must match weight type
        device=device,
        brs_mode="NoBRS",
    )

    # Load image (BGR as required by predictor)
    img_bgr = cv2.imread(args.input)
    if img_bgr is None:
        raise FileNotFoundError(args.input)
    h, w = img_bgr.shape[:2]

    # Generate a dummy click: centre, positive
    clicker = Clicker((h, w))
    clicker.add_click(Click(y=h // 2, x=w // 2, is_positive=True))

    # Inference
    pred_mask, _ = predictor.get_prediction(clicker, prev_mask=None)
    bin_mask = (pred_mask > 0).astype(np.uint8)

    # Visualise overlay
    overlay = draw_with_blend_and_contour(img_bgr, bin_mask)

    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / f"{Path(args.input).stem}_overlay.png"
    cv2.imwrite(str(out_file), overlay)
    print("Saved overlay to", out_file)


if __name__ == "__main__":
    main()
