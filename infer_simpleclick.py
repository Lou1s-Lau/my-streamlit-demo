# infer_simpleclick.py

import argparse
import os
import gdown
import torch
import numpy as np
import cv2

# 以下 import 取决于你的 SimpleClick 代码布局，可能需要适当调整
from isegm.inference import utils as pred_utils
from isegm.config import cfg as default_cfg

# Google Drive 上 cocolvis_vit_huge.pth 的文件 ID
GDRIVE_ID = "1kMHYLPC8uKaCpiuF3kfrlFQK6LyOpXKZ"
# 本地存放权重的路径
WEIGHTS_DIR = os.path.join(os.path.dirname(__file__), "weights", "simpleclick_models")
WEIGHTS_PATH = os.path.join(WEIGHTS_DIR, "cocolvis_vit_huge.pth")


def download_checkpoint():
    """如果本地不存在，就从 Google Drive 下载权重."""
    os.makedirs(WEIGHTS_DIR, exist_ok=True)
    if not os.path.exists(WEIGHTS_PATH):
        print(f"Downloading weights to {WEIGHTS_PATH} …")
        url = f"https://drive.google.com/uc?export=download&id={GDRIVE_ID}"
        gdown.download(url, WEIGHTS_PATH, quiet=False)
    return WEIGHTS_PATH


@torch.no_grad()
def build_predictor(checkpoint_path: str, device: torch.device):
    """
    构建并返回一个 predictor 对象，供 web.py 调用：
      predictor.get_prediction(image_np, clicks: List[(x,y,is_positive)]) -> mask_np
    """
    # 下载（如果需要）
    ckpt = download_checkpoint() if checkpoint_path == WEIGHTS_PATH else checkpoint_path

    # 加载配置
    cfg = default_cfg.clone()
    # 这里假设你有一个 config 文件；如果没有可跳过
    # cfg.merge_from_file("SimpleClick-1.0/configs/cocolvis_vit_huge.yaml")
    cfg.MODEL.WEIGHTS = ckpt
    cfg.MODEL.DEVICE = "cuda" if device.type == "cuda" else "cpu"

    # 获取 predictor
    predictor = pred_utils.get_predictor(cfg, device)
    return predictor


@torch.no_grad()
def get_prediction(predictor, image_np: np.ndarray, clicks: list):
    """
    clicks: List of tuples (x:int, y:int, is_positive:bool)
    返回： H×W 二值 mask (np.uint8 0/1)
    """
    # 转成 predictor 需要的格式
    clicks_lists = [[(c[0], c[1], c[2]) for c in clicks]]  # batch of one
    # predictor 返回一个字典，里面含 'instances' 或者直接返回 mask
    output = predictor.get_prediction(image_np, clicks_lists, None)
    # 根据具体返回格式可能不同，这里假设 output['instances'] 是分割 logits
    if "instances" in output:
        mask = output["instances"].argmax(0).cpu().numpy().astype(np.uint8)
    else:
        mask = output[0].cpu().numpy().astype(np.uint8)
    return mask


def run_cli():
    parser = argparse.ArgumentParser(description="SimpleClick Inference")
    parser.add_argument("--input",     required=True,  help="Path to input image")
    parser.add_argument("--output",    required=True,  help="Dir to save overlay")
    parser.add_argument("--checkpoint",default=WEIGHTS_PATH, help="Path to .pth file")
    parser.add_argument("--gpu",       type=int, default=-1, help="GPU id or -1 for CPU")
    args = parser.parse_args()

    # 准备
    device = torch.device(f"cuda:{args.gpu}" if args.gpu >= 0 else "cpu")
    predictor = build_predictor(args.checkpoint, device)

    # 读图、做一个中心点击示例
    img = cv2.imread(args.input)
    h, w = img.shape[:2]
    clicks = [(w // 2, h // 2, True)]

    # 推理
    mask = get_prediction(predictor, img, clicks)

    # 生成 overlay
    overlay = img.copy()
    overlay[mask > 0] = [0, 0, 255]  # 蓝色叠加
    basename = os.path.splitext(os.path.basename(args.input))[0]
    out_path = os.path.join(args.output, f"{basename}_overlay.png")
    cv2.imwrite(out_path, overlay)
    print(f"Saved overlay to {out_path}")


if __name__ == "__main__":
    run_cli()
