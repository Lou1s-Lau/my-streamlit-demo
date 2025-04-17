#!/usr/bin/env python3
"""infer_simpleclick.py  (Google‑Drive version)
-------------------------------------------------
Headless inference for **SimpleClick** that automatically downloads the
ViT‑Huge checkpoint from Google Drive when missing.

Usage example (CPU):
    python3 infer_simpleclick.py \
        --input ./sample.jpg \
        --output ./results \
        --checkpoint ./weights/simpleclick_models/cocolvis_vit_huge.pth \
        --gpu -1

*Default behaviour* uses **one centre positive click** just to show the
end‑to‑end pipeline; integrate your own click list in Streamlit for true
interactive mode.
"""

# ------------------------- standard libs --------------------------
import argparse, sys, os, pathlib, shutil
from pathlib import Path

# ------------------------- third‑party ----------------------------
import cv2, numpy as np, torch, gdown   # ensure gdown in requirements.txt

# ------------------------- local import ---------------------------
ROOT = pathlib.Path(__file__).resolve().parent
sys.path.append(str(ROOT / "SimpleClick"))

from isegm.inference import utils as pred_utils        # type: ignore
from isegm.inference.clicker import Clicker, Click    # type: ignore
from isegm.utils.vis import draw_with_blend_and_contour  # type: ignore

# --------------------------------------------------
# Map filename → Google Drive **file id** (not full URL)
# --------------------------------------------------
WEIGHT_IDS = {
    "cocolvis_vit_huge.pth": "1kMHYLPC8uKaCpiuF3kfrlFQK6LyOpXKZ",
    # add more ids if needed
}


def ensure_weights(path: Path):
    """Download weight file from Google Drive if it is absent."""
    if path.exists():
        return
    file_id = WEIGHT_IDS.get(path.name)
    if file_id is None:
        raise ValueError(f"No Google‑Drive id for {path.name}. Please update WEIGHT_IDS.")
    url = f"https://drive.google.com/uc?export=download&id={file_id}"
    path.parent.mkdir(parents=True, exist_ok=True)
    print(f"[infer_simpleclick] Downloading {path.name} …")
    gdown.download(url, str(path), quiet=False)
    print("[infer_simpleclick] Done →", path)


def build_argparser():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True, help="input PNG/JPG")
    ap.add_argument("--output", required=True, help="output folder")
    ap.add_argument("--checkpoint", required=True, help=".pth path")
    ap.add_argument("--gpu", type=int, default=-1, help="GPU id, -1=CPU")
    ap.add_argument("--model-name", default="vit_huge", choices=["vit_huge"], help="match weight type")
    return ap


def main():
    args = build_argparser().parse_args()

    ckpt = Path(args.checkpoint)
    ensure_weights(ckpt)

    device = f"cuda:{args.gpu}" if args.gpu >= 0 and torch.cuda.is_available() else "cpu"

    predictor = pred_utils.get_predictor(
        checkpoint_path=str(ckpt),
        model_name=args.model_name,
        device=device,
        brs_mode="NoBRS",
    )

    img_bgr = cv2.imread(args.input)
    if img_bgr is None:
        raise FileNotFoundError(args.input)
    h, w = img_bgr.shape[:2]

    # one positive click in centre
    clicker = Clicker((h, w))
    clicker.add_click(Click(y=h//2, x=w//2, is_positive=True))

    mask, _ = predictor.get_prediction(clicker, prev_mask=None)
    overlay = draw_with_blend_and_contour(img_bgr, (mask > 0).astype(np.uint8))

    out_dir = Path(args.output); out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{Path(args.input).stem}_overlay.png"
    cv2.imwrite(str(out_path), overlay)
    print("Saved overlay to", out_path)


if __name__ == "__main__":
    main()
