#!/usr/bin/env python3
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
    return p.parse_args()


def main():
    args = parse_args()

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


if __name__ == "__main__":
    main()
