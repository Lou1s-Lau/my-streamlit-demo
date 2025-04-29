# infer_simpleclick.py

import os, sys

# —— 1. 把 SimpleClick-1.0 加入到模块搜索路径 —— 
BASE_DIR = os.path.dirname(__file__)
SIMPLECLICK_PATH = os.path.join(BASE_DIR, "SimpleClick-1.0")
if SIMPLECLICK_PATH not in sys.path:
    sys.path.insert(0, SIMPLECLICK_PATH)

# 然后再导入
import argparse
import gdown
import torch
import numpy as np
from PIL import Image

from isegm.inference import utils as pred_utils
from isegm.config import cfg as default_cfg

import argparse
import os
import gdown
import torch
import numpy as np
from PIL import Image

from isegm.inference import utils as pred_utils
from isegm.config import cfg as default_cfg

GDRIVE_ID = "1kMHYLPC8uKaCpiuF3kfrlFQK6LyOpXKZ"
WEIGHTS_DIR = os.path.join(os.path.dirname(__file__), "weights", "simpleclick_models")
WEIGHTS_PATH = os.path.join(WEIGHTS_DIR, "cocolvis_vit_huge.pth")

def download_checkpoint():
    os.makedirs(WEIGHTS_DIR, exist_ok=True)
    if not os.path.exists(WEIGHTS_PATH):
        url = f"https://drive.google.com/uc?export=download&id={GDRIVE_ID}"
        gdown.download(url, WEIGHTS_PATH, quiet=False)
    return WEIGHTS_PATH

@torch.no_grad()
def build_predictor(checkpoint_path: str, device: torch.device):
    ckpt = download_checkpoint() if checkpoint_path == WEIGHTS_PATH else checkpoint_path
    cfg = default_cfg.clone()
    cfg.MODEL.WEIGHTS = ckpt
    cfg.MODEL.DEVICE = "cuda" if device.type == "cuda" else "cpu"
    return pred_utils.get_predictor(cfg, device)

@torch.no_grad()
def get_prediction(predictor, image_np: np.ndarray, clicks: list):
    # clicks: [(x,y,is_pos), ...]
    clicks_list = [[(c[0], c[1], c[2]) for c in clicks]]
    output = predictor.get_prediction(image_np, clicks_list, None)
    # assume output['instances'] is [H,W] logits or mask
    mask = output["instances"].argmax(0).cpu().numpy().astype(np.uint8) \
           if "instances" in output \
           else output[0].cpu().numpy().astype(np.uint8)
    return mask

def run_cli():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input",     required=True)
    parser.add_argument("--output",    required=True)
    parser.add_argument("--checkpoint",default=WEIGHTS_PATH)
    parser.add_argument("--gpu",       type=int, default=-1)
    args = parser.parse_args()

    device = torch.device(f"cuda:{args.gpu}" if args.gpu >= 0 else "cpu")
    predictor = build_predictor(args.checkpoint, device)

    # load with PIL
    img = Image.open(args.input).convert("RGB")
    img_np = np.array(img)
    h, w = img_np.shape[:2]
    # single center click example
    clicks = [(w//2, h//2, True)]
    mask = get_prediction(predictor, img_np, clicks)

    # overlay red
    overlay = img_np.copy()
    overlay[mask > 0] = [255, 0, 0]
    out_img = Image.fromarray(overlay)
    os.makedirs(args.output, exist_ok=True)
    base = os.path.splitext(os.path.basename(args.input))[0]
    out_path = os.path.join(args.output, f"{base}_overlay.png")
    out_img.save(out_path)
    print(f"Saved overlay to {out_path}")

if __name__ == "__main__":
    run_cli()
