# infer_simpleclick.py

import os
import sys
import argparse
import gdown
import torch
import numpy as np
from PIL import Image

# ─── 1. 把 SimpleClick-1.0 放到模块搜索路径最前面 ───
BASE_DIR = os.path.dirname(__file__)
SCC_DIR = os.path.join(BASE_DIR, "SimpleClick-1.0")
if SCC_DIR not in sys.path:
    sys.path.insert(0, SCC_DIR)

# ─── 2. 正确导入 v1.0 接口 ───
from isegm.utils.exp       import load_config_file
from isegm.inference.utils import find_checkpoint, load_is_model

# ─── 3. Google Drive 权重下载设定 ───
GDRIVE_ID    = "1kMHYLPC8uKaCpiuF3kfrlFQK6LyOpXKZ"
WEIGHTS_DIR  = os.path.join(BASE_DIR, "weights", "simpleclick_models")
WEIGHTS_PATH = os.path.join(WEIGHTS_DIR, "cocolvis_vit_huge.pth")

def download_checkpoint():
    os.makedirs(WEIGHTS_DIR, exist_ok=True)
    if not os.path.exists(WEIGHTS_PATH):
        url = f"https://drive.google.com/uc?export=download&id={GDRIVE_ID}"
        gdown.download(url, WEIGHTS_PATH, quiet=False)
    return WEIGHTS_PATH

# ─── 4. 构建模型函数 ───
@torch.no_grad()
def build_predictor(checkpoint: str, device: torch.device):
    # 4.1 下载权重（若本地不存在）
    ckpt = download_checkpoint() if checkpoint == WEIGHTS_PATH else checkpoint

    # 4.2 读取 config.yml
    cfg_path = os.path.join(SCC_DIR, "config.yml")
    # return_edict=True 会得到一个类似 dict 的配置对象
    cfg = load_config_file(cfg_path, return_edict=True)

    # 4.3 自动在 cfg.INTERACTIVE_MODELS_PATH 下查找权重文件
    checkpoint_path = find_checkpoint(cfg.INTERACTIVE_MODELS_PATH, os.path.basename(ckpt))

    # 4.4 加载模型（eval_ritm=False, cpu_dist_maps=True 同 demo.py）
    model = load_is_model(checkpoint_path, device, eval_ritm=False, cpu_dist_maps=True)
    return model

# ─── 5. 推理函数（静态中心点击示例） ───
@torch.no_grad()
def get_prediction(model, image_np: np.ndarray, clicks: list):
    """
    model: load_is_model 返回的模型
    image_np: H×W×3 uint8 的 numpy 图像
    clicks: List of (x:int, y:int, is_positive:bool)
    """
    # 这里只做一个“中央点击”demo，如果你要完全交互可以改这里：
    h, w = image_np.shape[:2]
    # 强制用中央正点
    pts = [(w//2, h//2, True)]

    # 单图批次化
    clicks_lists = [pts]
    # 模型调用接口：直接把 ndarray 和点击列表扔进去
    output = model(image_np, clicks_lists)  # ITRMaskModel 的 __call__
    # output 是一个 dict，里面 key=“instances” 是 logits tensor
    mask = output["instances"].argmax(0).cpu().numpy().astype(np.uint8)
    return mask

# ─── 6. 命令行调用保持兼容 ───
def run_cli():
    parser = argparse.ArgumentParser(description="SimpleClick Inference")
    parser.add_argument("--input",      required=True,  help="输入图像路径")
    parser.add_argument("--output",     required=True,  help="输出 overlay 目录")
    parser.add_argument("--checkpoint", default=WEIGHTS_PATH)
    parser.add_argument("--gpu",        type=int, default=-1, help="GPU id，-1 表示 CPU")
    args = parser.parse_args()

    device = torch.device(f"cuda:{args.gpu}" if args.gpu >= 0 else "cpu")
    model = build_predictor(args.checkpoint, device)

    # 读图
    img = Image.open(args.input).convert("RGB")
    img_np = np.array(img)

    # 推理
    mask = get_prediction(model, img_np, [])

    # 叠加红色 overlay
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
