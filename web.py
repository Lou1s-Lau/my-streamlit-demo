import streamlit as st
from PIL import Image
import numpy as np
import tempfile, subprocess, uuid, os

# 新增：绘图画布组件
from streamlit_drawable_canvas import st_canvas

# 页面全局配置
st.set_page_config(
    page_title="Interactive Medical Image Segmentation",
    layout="wide",
    initial_sidebar_state="expanded",
)

# 侧边栏导航，新增 Interactive Demo，Static Demo 改名 Demo
st.sidebar.markdown("# 🏥 Medical AI")
page = st.sidebar.radio("Navigate", [
    "Overview", "Background", "iSegFormer", "Interactive Demo", "Demo", "References"
])

def load_asset(name, caption=None):
    path = os.path.join("SimpleClick-1.0", "assets", name)
    if os.path.exists(path):
        st.image(path, caption=caption, use_container_width=True)
    else:
        st.warning(f"Asset `{name}` not found at `{path}`. Please upload it there.")

# 1. Overview
if page == "Overview":
    st.title("Accelerating Clinical Diagnosis with Interactive Segmentation")
    st.markdown("""
    In modern radiology, doctors rely on **CT**, **MRI**, and other scans
    to spot subtle abnormalities. While fully automated AI tools can
    segment organs or lesions at scale, they sometimes struggle with
    fuzzy boundaries or rare pathologies.

    **Interactive segmentation** bridges the gap by letting physicians
    guide the algorithm—clicking or marking key regions—combining
    **AI speed** with **clinical expertise** for **higher accuracy**
    and **faster decision‐making**.
    """)
    load_asset("mri_example.jpg", caption="Figure 1: Knee MRI scan")
    st.markdown("""
    **Figure 1: Tri‑planar MRI of the Knee**

    This composite image shows three standard MRI views of the knee:

    1. **Axial (left):** Horizontal slice through the femoral condyles and tibial plateau.  
    2. **Coronal (center):** Front‑to‑back slice, ideal for cartilage thickness and joint space.  
    3. **Sagittal (right):** Side slice highlighting cruciate ligaments and patellofemoral joint.  
    """)

# 2. Background
elif page == "Background":
    st.title("Background: Deep Learning & Medical Image Segmentation")
    st.markdown("""
    **1. What is Deep Learning?**  
    - Deep learning uses layered neural networks to learn features from data (edges → textures → shapes).  
    - Training adjusts weights to minimize errors on labeled examples.

    **2. Key Architectures**  
    - **CNN** (Convolutional Neural Network): uses kernels to detect local patterns; backbone of UNet.  
    - **ViT** (Vision Transformer): splits images into patches, uses self‐attention for global context; ideal for 3D volumes.

    **3. Image Segmentation**  
    - Assigns a label to every pixel (e.g., “cartilage,” “bone,” “background”), yielding precise outlines.

    **4. Fully Automated vs. Interactive**  
    - **Fully Automated:** batch processing, no user input; <em>Pros:</em> high throughput; <em>Cons:</em> may fail on edge cases.  
    - **Interactive:** user provides clicks/scribbles; <em>Pros:</em> corrects mistakes with few interactions.

    **5. Why It Matters**  
    - Medical scans are large 3D volumes; manual annotation is tedious.  
    - Interactive methods let clinicians quickly refine results in challenging cases.
    """)

    st.markdown("### Fully Automatic vs. Interactive Segmentation Workflow")
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("**Fully Automatic Segmentation**")
        img_auto = os.path.join("SimpleClick-1.0", "assets", "automatic.png")
        if os.path.exists(img_auto):
            st.image(img_auto, use_container_width=True)
        else:
            st.warning(f"Image not found: {img_auto}")
    with col2:
        st.markdown("**Interactive Segmentation**")
        img_inter = os.path.join("SimpleClick-1.0", "assets", "interactive.png")
        if os.path.exists(img_inter):
            st.image(img_inter, use_container_width=True)
        else:
            st.warning(f"Image not found: {img_inter}")

# 3. iSegFormer
elif page == "iSegFormer":
    st.title("iSegFormer (Liu et al., MICCAI 2022)")
    st.markdown("""
    iSegFormer introduces **interactive segmentation** for **3D knee MRI**:
    - **Encoder:** Swin Transformer captures both local and global context.  
    - **Decoder:** Lightweight MLP outputs masks slice by slice.  
    - **Workflow:** Label a few slices → model refines across the volume.  
    - **Result:** >90% Dice with minimal input; <em>trade‐off:</em> higher GPU memory.
    """)
    load_asset("architecture.jpg", caption="Figure 3: iSegFormer Architecture")

# 4. Interactive Demo
elif page == "Interactive Demo":
    st.title("Interactive Segmentation Demo")

    # 上传图像
    uploaded = st.file_uploader("Upload a medical image", type=["png","jpg","jpeg"])
    if not uploaded:
        st.info("Please upload an image to begin.")
        st.stop()

    # 转为 NumPy
    img = Image.open(uploaded).convert("RGB")
    img_np = np.array(img)

    # 延迟加载模型预测器（假设 infer_simpleclick.build_predictor 存在）
    @st.cache_resource
    def load_predictor(path):
        from infer_simpleclick import build_predictor
        return build_predictor(path)

    predictor = load_predictor("./weights/simpleclick_models/cocolvis_vit_huge.pth")

    # 选择点击类型
    click_type = st.radio("Click type", ["Positive (foreground)", "Negative (background)"])

    # 初始化点击列表
    if "clicks" not in st.session_state:
        st.session_state.clicks = []  # list of (x,y,is_positive)

    # 可绘制画布，仅支持点击(point)
    canvas_result = st_canvas(
        fill_color="rgba(0,0,0,0)",
        stroke_color="#0f0" if click_type.startswith("Positive") else "#f00",
        stroke_width=20,
        background_image=img,
        drawing_mode="point",
        key="canvas",
        update_streamlit=True,
        height=img_np.shape[0],
        width=img_np.shape[1],
    )

    # 记录最新点击
    if canvas_result.json_data and canvas_result.json_data.get("objects"):
        for obj in canvas_result.json_data["objects"][-1:]:
            x, y = obj["path"][-1]
            st.session_state.clicks.append(
                (int(x), int(y), click_type.startswith("Positive"))
            )

    # 展示点击历史 & 运行分割
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("#### Click history")
        for i, (x, y, is_pos) in enumerate(st.session_state.clicks, 1):
            emoji = "🟢" if is_pos else "🔴"
            st.write(f"{i}. {emoji} ({x}, {y})")
        if st.button("🔄 Reset Clicks"):
            st.session_state.clicks = []

    with col2:
        st.markdown("#### Segmentation result")
        if st.button("Run / Update Segmentation"):
            # predictor.get_prediction(image_np, clicks) → 返回二值 mask
            mask = predictor.get_prediction(img_np, st.session_state.clicks)
            # 叠加：红色高亮
            overlay = img_np.copy()
            overlay[mask > 0] = [255, 0, 0]
            st.image(
                [img_np, overlay],
                caption=["Input Image", "Overlay Result"],
                use_container_width=True
            )

# 5. Demo（保持静态中心点Demo）
elif page == "Demo":
    st.title("Static Demo: Center‑Click Segmentation")

    video_path = os.path.join("SimpleClick-1.0", "assets", "demo_oaizib_stcn_with_cycle.mp4")
    if os.path.exists(video_path):
        st.video(video_path, format="video/mp4", start_time=0)
    else:
        st.warning(f"Demo video not found at {video_path}")

    st.info("*The full interactive version is in the 'Interactive Demo' tab.*")
    use_gpu = st.checkbox("Use GPU", value=False)
    uploaded = st.file_uploader("Upload an image", type=["jpg","jpeg","png"])
    if uploaded:
        tmp_dir = tempfile.mkdtemp()
        in_path = os.path.join(tmp_dir, f"{uuid.uuid4()}.png")
        with open(in_path, "wb") as f:
            f.write(uploaded.read())
        st.image(in_path, caption="Input Image", use_container_width=True)

        if st.button("Run Static Demo"):
            st.write("Running inference…")
            cmd = [
                "python3", "infer_simpleclick.py",
                "--input", in_path,
                "--output", tmp_dir,
                "--checkpoint", "./weights/simpleclick_models/cocolvis_vit_huge.pth",
                "--gpu", "0" if use_gpu else "-1"
            ]
            try:
                subprocess.run(cmd, check=True)
            except subprocess.CalledProcessError as e:
                st.error(f"Inference error:\n{e}")
            else:
                base = os.path.splitext(os.path.basename(in_path))[0]
                out = os.path.join(tmp_dir, f"{base}_overlay.png")
                if os.path.exists(out):
                    st.image(out, caption="Overlay", use_container_width=True)
                else:
                    st.error("Overlay not found. Please check server logs.")

# 6. References
elif page == "References":
    st.title("References")
    st.markdown("""
    1. Liu, Q., Xu, Z., Jiao, Y., & Niethammer, M. (2022). *iSegFormer: Interactive segmentation via transformers with application to 3D knee MRI images*. MICCAI 2022.  
    2. Zhang, X., Li, Z., Shi, H., Deng, Y., Zhou, G., & Tang, S. (2021). *A deep learning‑based method for knee articular cartilage segmentation in MRI images*. ICCAIS 2021.  
    3. Liu, Q., Xu, Z., Bertasius, G., & Niethammer, M. (2023). *SimpleClick: Interactive Image Segmentation with Simple Vision Transformers*. ICCVW 2023.  
    4. Marinov, Z., Jäger, P. F., Egger, J., Kleesiek, J., & Stiefelhagen, R. (2024). *Deep interactive segmentation of medical images: A systematic review and taxonomy*. IEEE TPAMI, 46(12), 10998–11039.  
    5. Huang, M., Zou, J., Zhang, Y., Bhatti, U. A., & Chen, J. (2024). *Efficient click‑based interactive segmentation for medical image with improved Plain‑ViT*. IEEE JBHI.  
    6. Luo, X., Wang, G., Song, T., Zhang, J., Aertsen, M., Deprest, J., Ourselin, S., Vercauteren, T., & Zhang, S. (2021). *MIDeepSeg: Minimally Interactive Segmentation of Unseen Objects from Medical Images Using Deep Learning*. arXiv:2104.12166.
    """)
