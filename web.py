import streamlit as st
from PIL import Image
import tempfile, subprocess, uuid, os

# 页面全局配置
st.set_page_config(
    page_title="Interactive Medical Image Segmentation",
    layout="wide",
    initial_sidebar_state="expanded",
)

# 侧边栏导航
st.sidebar.markdown("# 🏥 Medical AI")
page = st.sidebar.radio("Navigate", [
    "Overview", "Background", "iSegFormer", "Demo", "References"
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

# …（前面部分保持不变，略）…

# 2. Background
elif page == "Background":
    st.title("Background: Deep Learning & Medical Image Segmentation")
    st.markdown("""
    **1. What is Deep Learning?**  
    - Deep learning is a branch of machine learning where models called _neural networks_ 
      learn patterns from data through multiple layers of mathematical operations.  
    - Each **layer** transforms its input (e.g., an image) into progressively more 
      abstract features (edges → textures → shapes).  
    - During **training**, the network adjusts internal parameters (_weights_) to minimize 
      errors on a labeled dataset.

    **2. Key Architectures**  
    - **Convolutional Neural Networks (CNNs):**  
      - Use small, learnable filters (kernels) that _convolve_ across the image to detect 
        local patterns such as edges or corners.  
      - **Receptive field**: the area of input each filter “sees.” Deeper layers see larger contexts.  
      - Popular for image classification and **semantic segmentation** (pixel‐wise labeling).  

    - **Vision Transformers (ViT):**  
      - Divide an image into fixed‐size **patches** (like tokens in language).  
      - Use **self‐attention**, where each patch weighs its relationship to every other patch, 
        capturing _global context_.  
      - Well‐suited for complex or high‐dimensional data, such as 3D medical volumes.

    **3. What Is Image Segmentation?**  
    - **Image segmentation** assigns a label to every pixel: e.g., “cartilage,” “bone,” or “background.”  
    - Unlike classification (one label per image) or detection (bounding boxes), segmentation 
      yields precise outlines of structures.  
    - In medicine, accurate segmentation of organs, tumors, or vessels is crucial for treatment planning.

    **4. Automated vs. Interactive Segmentation**  
    - **Fully Automated:**  
      - The model processes images in a batch with no user input.  
      - **Pros:** Fast, scalable, minimal human effort.  
      - **Cons:** May fail on unusual anatomy, low‐contrast regions, or artifacts.  

    - **Interactive / Semi‑automatic:**  
      - The user (e.g., radiologist) provides _hints_—clicks, scribbles, or bounding boxes—  
        to guide the algorithm.  
      - **Positive clicks** mark areas _inside_ a structure; **negative clicks** mark outside regions.  
      - **Pros:** Combines human expertise with AI, corrects edge‐cases, requires only a few interactions.  
      - **Cons:** Slightly slower per image, but often still under a minute.

    **5. Why It Matters in Medical Imaging**  
    - Medical scans (MRI, CT) are often 3D with hundreds of slices—manual annotation is tedious.  
    - Fully automatic tools speed up routine tasks but can’t handle every patient’s unique anatomy.  
    - Interactive methods empower clinicians to **quickly refine** results in challenging cases, 
      improving diagnostic accuracy and saving time.
    """)
# —— 左右两列并排放图 —— 
    st.markdown("### 全自动 vs. 交互式 分割流程对比")
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("**全自动分割**")
        img_auto = os.path.join("SimpleClick-1.0", "assets",
                                "automatic.jpg")
        if os.path.exists(img_auto):
            st.image(img_auto, use_container_width=True)
        else:
            st.warning(f"找不到图像：{img_auto}")

    with col2:
        st.markdown("**交互式分割**")
        img_inter = os.path.join("SimpleClick-1.0", "assets",
                                 "interactive.jpg")
        if os.path.exists(img_inter):
            st.image(img_inter, use_container_width=True)
        else:
            st.warning(f"找不到图像：{img_inter}")



# 3. iSegFormer
elif page == "iSegFormer":
    st.title("iSegFormer (Liu et al., MICCAI 2022)")
    st.markdown("""
    iSegFormer introduces **interactive segmentation** for **3D knee MRI**:
    - **Encoder:** Swin Transformer captures both local and global context.  
    - **Decoder:** Lightweight MLP (Multilayer Perceptron) outputs segmentation masks slice by slice.  
    - **Workflow:** Clinician labels a few key slices → model propagates and refines segmentation  
      across the entire volume.  
    - **Results:** Over 90% Dice score with minimal manual input.  
    - **Trade‑off:** High GPU memory usage due to processing 3D data.
    """)
    load_asset("architecture.jpg", caption="Figure 3: iSegFormer Architecture")

# 4. Demo
elif page == "Demo":
    st.title("Static Demo: Center‑Click Segmentation")

    # 本地视频回归
    video_path = os.path.join("SimpleClick-1.0", "assets", "demo_oaizib_stcn_with_cycle.mp4")
    if os.path.exists(video_path):
        st.video(video_path, format="video/mp4", start_time=0)
    else:
        st.warning(f"Demo video not found at {video_path}")

    st.info("*This demo uses a single center click; full interactive version coming soon.*")
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

# 5. References
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
