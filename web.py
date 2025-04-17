import streamlit as st
from PIL import Image
import torch
from simpleclick.models import ITRMaskModel
from simpleclick.utils import visualize

st.set_page_config(page_title="SimpleClick Interactive Web Demo", layout="wide")

st.title("SimpleClick: Web-based Interactive Segmentation Demo")

# 侧边栏参数
st.sidebar.header("Demo 设置")
checkpoint = st.sidebar.text_input(
    "Checkpoint 路径",
    value="./weights/simpleclick_models/cocolvis_vit_huge.pth"
)
use_gpu = st.sidebar.checkbox("使用 GPU", value=True)

# 上传图像
st.header("上传图像进行分割")
uploaded_file = st.file_uploader("请选择一张图片（PNG/JPEG）", type=["png","jpg","jpeg"])

if uploaded_file:
    img = Image.open(uploaded_file).convert("RGB")
    st.image(img, caption="原始图像", use_container_width=True)

    # 捕获点击
    st.write("请在下面的图像上点击标注：红色标注正样本，蓝色标注负样本。")
    # Note: Streamlit currently does not support direct image click coords;
    # using st.canvas or custom component to collect click coords; placeholder here.

    if st.button("运行分割 Demo"):
        device = "cuda" if (use_gpu and torch.cuda.is_available()) else "cpu"
        model = ITRMaskModel(checkpoint=checkpoint, device=device)

        # 示例：中心点点击
        w, h = img.size
        clicks_list = [(h//2, w//2, 1)]  # (y, x, is_positive)

        mask = model.predict([img], clicks=[clicks_list])[0]
        overlay = visualize(img, mask)

        st.image(overlay, caption="Segmentation Result", use_container_width=True)
else:
    st.info("请先上传图片后再运行 Demo。")

# 版权信息
st.markdown("---\n*基于 SimpleClick 实现（ICCV 2023）。示例仅供演示，需根据实际组件完善点击交互逻辑。*")
