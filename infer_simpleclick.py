#!/usr/bin/env python3
"""infer_simpleclick.py  (Google Drive auto-download)
------------------------------------------------
Headless inference for **SimpleClick** using ViT-Huge.
Automatically downloads the checkpoint from Google Drive when missing.

Usage (CPU):
    python3 infer_simpleclick.py \
      --input ./sample.jpg \
      --output ./results \
      --checkpoint ./weights/simpleclick_models/cocolvis_vit_huge.pth \
      --gpu -1
"""
import argparse
import sys, os, pathlib
from pathlib import Path
import shutil

# Third-party
import cv2
import numpy as np
import torch
import gdown

# Add local SimpleClick to PYTHONPATH
ROOT = pathlib.Path(__file__).resolve().parent
sys.path.append(str(ROOT / "SimpleClick"))

from isegm.inference import utils as pred_utils         # type: ignore
from isegm.inference.clicker import Clicker, Click     # type: ignore
from isegm.utils.vis import draw_with_blend_and_contour  # type: ignore

# Google Drive file IDs for weights
WEIGHT_IDS = {
    "cocolvis_vit_huge.pth": "1kMHYLPC8uKaCpiuF3kfrlFQK6LyOpXKZ",
}


def ensure_checkpoint(ckpt_path: Path):
    """Download checkpoint from Google Drive if not present."""
    if ckpt_path.exists():
        return
    fid = WEIGHT_IDS.get(ckpt_path.name)
    if not fid:
        raise ValueError(f"No Google Drive ID for {ckpt_path.name}")
    url = f"https://drive.google.com/uc?export=download&id={fid}"
    ckpt_path.parent.mkdir(parents=True, exist_ok=True)
    print(f"Downloading checkpoint {ckpt_path.name}...")
    gdown.download(url, str(ckpt_path), quiet=False)
    print("Download complete.")


def parse_args():
    p = argparse.ArgumentParser(description="Headless SimpleClick inference")
    p.add_argument("--input", required=True, help="Path to input image (PNG/JPG)")
    p.add_argument("--output", required=True, help="Directory to save overlay")
    p.add_argument("--checkpoint", required=True, help="Path to .pth checkpoint")
    p.add_argument("--gpu", type=int, default=-1, help="GPU id (>=0) or -1 for CPU")
    return p.parse_args()


def main():
    args = parse_args()
    ckpt = Path(args.checkpoint)
    ensure_checkpoint(ckpt)

    device = f"cuda:{args.gpu}" if args.gpu >= 0 and torch.cuda.is_available() else "cpu"

    predictor = pred_utils.get_predictor(
        checkpoint_path=str(ckpt),
        model_name="vit_huge",
        device=device,
        brs_mode="NoBRS",
    )

    img = cv2.imread(args.input)
    if img is None:
        raise FileNotFoundError(args.input)
    h, w = img.shape[:2]

    clicker = Clicker((h, w))
    clicker.add_click(Click(y=h//2, x=w//2, is_positive=True))

    mask, _ = predictor.get_prediction(clicker, prev_mask=None)
    bin_mask = (mask > 0).astype(np.uint8)

    overlay = draw_with_blend_and_contour(img, bin_mask)

    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / f"{Path(args.input).stem}_overlay.png"
    cv2.imwrite(str(out_file), overlay)
    print(f"Saved overlay to {out_file}")


if __name__ == "__main__":
    main()
