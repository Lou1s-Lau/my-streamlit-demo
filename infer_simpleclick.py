# infer_simpleclick.py
import torch
# 备份原始 torch.load
_orig_torch_load = torch.load
# 新的 torch.load：如果调用里没传 weights_only，就强制加上 weights_only=False
def _patched_torch_load(f, *args, **kwargs):
    if "weights_only" not in kwargs:
        kwargs["weights_only"] = False
    return _orig_torch_load(f, *args, **kwargs)
# 覆盖
torch.load = _patched_torch_load

import os
import sys
import argparse
import gdown
import torch
import numpy as np
from PIL import Image

# ─── 1. 把 SimpleClick-1.0 源码加入 sys.path ───
BASE_DIR   = os.path.dirname(__file__)
SCC_DIR    = os.path.join(BASE_DIR, "SimpleClick-1.0")
if SCC_DIR not in sys.path:
    sys.path.insert(0, SCC_DIR)

# ─── 2. 从 v1.0 接口导入配置和模型加载工具 ───
from isegm.utils.exp       import load_config_file
from isegm.inference.utils import load_is_model

# ─── 3. Google Drive 权重下载配置 ───
GDRIVE_ID    = "1kMHYLPC8uKaCpiuF3kfrlFQK6LyOpXKZ"
WEIGHTS_DIR  = os.path.join(BASE_DIR, "weights", "simpleclick_models")
WEIGHTS_PATH = os.path.join(WEIGHTS_DIR, "cocolvis_vit_huge.pth")

def download_checkpoint():
    """如果本地没有，就从 Google Drive 下到 weights/simpleclick_models 文件夹里"""
    os.makedirs(WEIGHTS_DIR, exist_ok=True)
    if not os.path.exists(WEIGHTS_PATH):
        print(f"Downloading weights to {WEIGHTS_PATH} …")
        url = f"https://drive.google.com/uc?export=download&id={GDRIVE_ID}"
        gdown.download(url, WEIGHTS_PATH, quiet=False)
    return WEIGHTS_PATH

# ─── 4. 构建 Predictor 的函数 ───
@torch.no_grad()
def build_predictor(checkpoint: str, device: torch.device):
    # ——— 1. 无条件下载／返回本地 checkpoint 路径 ———
    # download_checkpoint() 会在 weights/simpleclick_models 下下载或返回已有 .pth
    ckpt = download_checkpoint()

    # ——— 2. 读取 config.yml ———
    cfg_path = os.path.join(SCC_DIR, "config.yml")
    cfg = load_config_file(cfg_path, return_edict=True)

    # ——— 3. 直接用 ckpt 加载模型 ———
    model = load_is_model(
        ckpt,
        device,
        eval_ritm=False,
        cpu_dist_maps=True
    )
    return model


# ─── 5. 推理函数：根据 clicks 返回二值 mask ───
@torch.no_grad()
def get_prediction(model, image_np: np.ndarray, clicks: list):
    """
    model: load_is_model 返回的模型实例
    image_np: H×W×3 ndarray
    clicks: List[ (x:int, y:int, is_positive:bool) ]
    """
    # 直接把用户传来的 clicks 用作交互
    clicks_lists = [clicks]  # batch of one
    output = model(image_np, clicks_lists)    # ITRMaskModel __call__
    mask = output["instances"].argmax(0).cpu().numpy().astype(np.uint8)
    return mask

# ─── 6. 保留 CLI 模式 ───
def run_cli():
    parser = argparse.ArgumentParser(description="SimpleClick Inference")
    parser.add_argument("--input",      required=True,  help="输入图像路径")
    parser.add_argument("--output",     required=True,  help="输出 overlay 目录")
    parser.add_argument("--checkpoint", default=WEIGHTS_PATH)
    parser.add_argument("--gpu",        type=int, default=-1, help="GPU id，-1 表示 CPU")
    args = parser.parse_args()

    device = torch.device(f"cuda:{args.gpu}" if args.gpu >= 0 else "cpu")
    model  = build_predictor(args.checkpoint, device)

    img    = Image.open(args.input).convert("RGB")
    img_np = np.array(img)

    # 测试用：中心点正点击
    h, w  = img_np.shape[:2]
    clicks = [(w//2, h//2, True)]
    mask   = get_prediction(model, img_np, clicks)

    # 叠加红色 overlay
    overlay = img_np.copy()
    overlay[mask > 0] = [255,0,0]
    out_img = Image.fromarray(overlay)

    os.makedirs(args.output, exist_ok=True)
    base = os.path.splitext(os.path.basename(args.input))[0]
    out_path = os.path.join(args.output, f"{base}_overlay.png")
    out_img.save(out_path)
    print(f"Saved overlay to {out_path}")

if __name__ == "__main__":
    run_cli()
