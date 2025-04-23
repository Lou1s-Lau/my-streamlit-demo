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
    In modern radiology, doctors rely on **CT**, **MRI** and other scans to detect 
    subtle abnormalities in patient anatomy. Fully automated AI segmentation tools 
    can process large volumes of images quickly, but often **struggle with fuzzy 
    boundaries**—for example, tumor margins obscured by noise or low contrast—and 
    **rare pathologies**, such as atypical glioblastoma shapes or uncommon vascular 
    malformations where training data are sparse (Zhang et al., 2021).

    **Interactive segmentation** addresses these limitations by allowing clinicians 
    to provide targeted feedback. When a physician clicks inside the lesion 
    (“positive click”) or just outside it (“negative click”), the algorithm’s 
    local probability map is updated and the contour is refined in real time—this 
    combines **human expertise** with **AI speed**, reducing correction time 
    significantly (Liu et al., 2022).  

    Below, Figure 1 shows a tri-planar knee MRI where cartilage boundaries can be 
    especially fuzzy.
    """, unsafe_allow_html=True)
    load_asset("mri_example.jpg", caption="Figure 1: Tri-planar MRI of the Knee")


# 2. Background
elif page == "Background":
    st.title("Background: Deep Learning & Medical Image Segmentation")
    st.markdown("""
    Deep learning revolutionized image analysis by introducing multi-layer neural networks 
    that automatically learn hierarchical features—from simple edges in early layers to 
    complex anatomical structures in deeper layers. In medical imaging, **Convolutional Neural 
    Networks** (CNNs) such as U-Net remain the workhorse for **fully automated** segmentation, 
    detecting local patterns with learnable kernels (Zhang et al., 2021). However, CNNs often 
    falter when lesions span large areas or when image quality is degraded by noise or motion.

    To overcome this, **Vision Transformers** (ViTs) split an image into patches and apply 
    a **self-attention** mechanism to model long-range dependencies. This global context 
    awareness is crucial in 3D scans—like MRI volumes—where adjacent slices share anatomical 
    information (Liu et al., 2023).

    **Image segmentation** assigns every pixel a label (e.g., “bone,” “cartilage,” “background”), 
    yielding precise contours used for surgical planning or treatment monitoring. Yet, even 
    state-of-the-art automated systems can struggle with:
    - **Fuzzy boundaries**, such as low-contrast tumor margins obscured by edema or blood;  
    - **Rare pathologies**, like atypical lesion shapes in glioblastoma or uncommon vascular 
      malformations, where few training examples exist (Marinov et al., 2024).

    **Interactive segmentation** bridges these gaps by letting clinicians guide the algorithm 
    through a few targeted interactions. A **positive click** inside the region of interest 
    and a **negative click** just outside it update the model’s local probability map, refining 
    the mask in real time. This human-in-the-loop approach corrects edge cases—e.g., a tiny 
    metastatic nodule adjacent to bone—while keeping interaction time under a minute per image 
    (Huang et al., 2024; Luo et al., 2021).

    Taken together, fully automated and interactive methods form a spectrum: one end offers 
    high-throughput batch processing, the other provides clinician-driven precision on 
    challenging cases.
    """)


# 3. iSegFormer
elif page == "iSegFormer":
    st.title("iSegFormer (Liu et al., MICCAI 2022)")
    st.markdown("""
    iSegFormer is an **interactive segmentation** framework tailored for **3D knee MRI**, where 
    cartilage surfaces present complex curvatures and often blend into adjacent tissues. In their 
    MICCAI 2022 paper, Liu et al. show that combining a hierarchical **Swin Transformer** encoder 
    with a lightweight **MLP decoder** allows the model to quickly adapt to sparse user annotations 
    (Liu et al., 2022).  

    Rather than processing each slice in isolation, iSegFormer propagates corrections across the 
    entire volume: the user labels a few key slices, and the transformer’s global attention 
    ensures consistency as the model refines segmentation in neighboring slices. This yields Dice 
    scores above 90% with minimal manual effort, demonstrating that interactive refinement can 
    match or exceed fully automated accuracy under limited annotation budgets.  

    The main trade-off is computational: handling 3D volumes with self-attention requires 
    substantial GPU memory and incurs longer model initialization times, which may challenge 
    deployment in resource-constrained clinical settings.
    """, unsafe_allow_html=True)
    load_asset("architecture.jpg", caption="Figure 3: iSegFormer Architecture")

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
