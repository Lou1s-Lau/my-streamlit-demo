#!/usr/bin/env python3
<<<<<<< HEAD
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
=======
# infer_simpleclick.py
"""
Run SimpleClick inference on a single image and save an overlay result.

示例（CPU）:
    python3 infer_simpleclick.py \
        --input ./sample.jpg \
        --output ./results \
        --checkpoint ./weights/simpleclick_models/cocolvis_vit_huge.pth \
        --gpu -1
"""

import argparse
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image

# SimpleClick / RITM imports
from isegm.inference import utils as pred_utils
from isegm.inference.clicker import Clicker, Click
from isegm.utils.vis import draw_with_blend_and_contour  # Simple visualiser


def parse_args():
    p = argparse.ArgumentParser(description="SimpleClick one‑shot inference")
    p.add_argument("--input", required=True, help="Path to input PNG/JPG")
    p.add_argument("--output", required=True, help="Folder to save overlay")
    p.add_argument("--checkpoint", required=True, help="Path to .pth weights")
    p.add_argument("--gpu", type=int, default=-1, help="GPU id (‑1 = CPU)")
>>>>>>> 62ddd49e922b94ebafaddb171b68fe702639a06d
    return p.parse_args()


def main():
    args = parse_args()
<<<<<<< HEAD
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
=======

    device = f"cuda:{args.gpu}" if args.gpu >= 0 and torch.cuda.is_available() else "cpu"

    # -------------------------------------------------
    # 1. 构建 predictor
    # -------------------------------------------------
    predictor = pred_utils.get_predictor(
        checkpoint_path=args.checkpoint,
        model_name="vit_huge",          # 与权重对应的模型串 (SimpleClick 默认为 vit_huge)
        device=device,
        brs_mode="NoBRS",               # SimpleClick 论文和 demo 用 NoBRS
    )

    # -------------------------------------------------
    # 2. 读取图像并设置到 predictor
    # -------------------------------------------------
    img_bgr = cv2.imread(args.input)                # H×W×3, BGR
    if img_bgr is None:
        raise FileNotFoundError(args.input)
    predictor.set_input_image(img_bgr)

    # -------------------------------------------------
    # 3. 构造一次点击 (中心正点击示例)
    # -------------------------------------------------
    h, w = img_bgr.shape[:2]
    clicker = Clicker((h, w))
    clicker.add_click(Click(y=h // 2, x=w // 2, is_positive=True))

    # -------------------------------------------------
    # 4. 推理
    # -------------------------------------------------
    pred_mask, _ = predictor.get_prediction(clicker, prev_mask=None)  # H×W float
    bin_mask = (pred_mask > 0).astype(np.uint8)                        # 0/1

    # -------------------------------------------------
    # 5. 叠加可视化并保存
    # -------------------------------------------------
    overlay = draw_with_blend_and_contour(img_bgr, bin_mask)
    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{Path(args.input).stem}_overlay.png"
    cv2.imwrite(str(out_path), overlay)
    print("Saved overlay to", out_path)
>>>>>>> 62ddd49e922b94ebafaddb171b68fe702639a06d


if __name__ == "__main__":
    main()
