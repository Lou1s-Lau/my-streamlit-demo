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
    "Overview", "Background", "iSegFormer", "SimpleClick", "Demo", "Installation"
])

# Helper：安全加载本地 asset
def load_asset(name, caption=None):
    # 新的资产目录：SimpleClick-1.0/assets/
    path = os.path.join("SimpleClick-1.0", "assets", name)
    if os.path.exists(path):
        st.image(path, caption=caption, use_container_width=True)
    else:
        st.warning(f"Asset `{name}` not found at `{path}`. 请检查该路径下是否已上传此文件。")

# 1. Overview
if page == "Overview":
    st.title("Accelerating Clinical Diagnosis with Interactive Segmentation")
    st.markdown("""
    In modern radiology, doctors rely on **CT**, **MRI**, and other scans to spot abnormalities.
    AI segmentation automates much of this, but edge cases and fuzzy boundaries still pose challenges.

    **Interactive segmentation** lets doctors **click** or **mark** regions,
    combining **AI speed** with **human expertise** for **higher accuracy** and **faster decisions**.
    """)
    load_asset("mri_example.jpg", caption="Figure 1: Knee MRI scan")
    st.markdown("""
    **Figure 1: Tri‑planar MRI of the Knee**

    This composite image shows three standard MRI views of the knee:

    1. **Axial Plane (left):** A horizontal slice through the femoral condyles and tibial plateau, highlighting cartilage surfaces, menisci, and potential meniscal tears.  
    2. **Coronal Plane (center):** A vertical, front‑to‑back view of the knee joint, ideal for assessing cartilage thickness, joint space narrowing, and medial/lateral compartment integrity.  
    3. **Sagittal Plane (right):** A side‑view slice that visualizes the anterior and posterior cruciate ligaments (ACL/PCL), femoral trochlea, and patellofemoral joint, critical for diagnosing ligament injuries and patellar tracking issues.

    These multi‑planar MRI images serve as the input for interactive segmentation models like iSegFormer and SimpleClick, enabling precise delineation of cartilage, menisci, and ligaments to support faster and more accurate clinical diagnosis.
    """)

# 2. Background
elif page == "Background":
    st.title("Automated vs. Interactive Segmentation")
    st.markdown("""
    - **Fully Automated**  
      - Pros: Batch processing, zero human effort  
      - Cons: May fail on unusual or low‑contrast cases  

    - **Interactive**  
      - Pros: Human‑in‑loop corrections, handles edge cases  
      - Cons: Requires minimal user input  
    """)
    load_asset("seg_pipeline.png", caption="Figure 2: Workflow comparison")

# 3. iSegFormer
elif page == "iSegFormer":
    st.title("iSegFormer (Liu et al., MICCAI 2022)")
    st.markdown("""
    **iSegFormer** tackles interactive **3D knee MRI** segmentation:
    - **Backbone:** Swin Transformer + lightweight MLP  
    - **Interactive Loop:** Doctor annotates/adjusts a few slices → model refines  
    - **Performance:** >90% Dice with minimal annotations  
    - **Limitation:** High GPU memory usage for 3D volumes
    """)
    load_asset("architecture.png", caption="Figure 3: iSegFormer architecture")

# 4. SimpleClick
elif page == "SimpleClick":
    st.title("SimpleClick (Liu et al., CVPR 2023)")
    st.markdown("""
    **SimpleClick** brings click‑style interaction to **2D images**:
    1. **Positive click** on object → coarse mask  
    2. **Negative click** on background → refine mask  
    3. **Iterate** until satisfactory

    - **Backbone:** Vision Transformer (ViT)  
    - **Speed:** >80 FPS on a single GPU  
    - **Accuracy:** State‑of‑the‑art on both natural & medical images
    """)
    load_asset("simpleclick_workflow.png", caption="Figure 4: SimpleClick workflow")

# 5. Demo (Static center‑click)
elif page == "Demo":
    st.title("Static Demo: Center‑Click Segmentation")
    st.info("*This demo uses a single center click. Full interactive version coming soon.*")

    uploaded = st.file_uploader("Upload an image", type=["jpg","jpeg","png"])
    use_gpu = st.checkbox("Use GPU", value=False)

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

# 6. Installation
elif page == "Installation":
    st.title("Installation & Usage")
    st.markdown("""
    1. **Clone repo**  
       `git clone https://github.com/yourname/my-streamlit-demo.git`  
       `cd my-streamlit-demo`

    2. **Install dependencies**  
       `pip install -r requirements.txt`

    3. **Place your assets** under `SimpleClick-1.0/assets/`  

    4. **Run the app**  
       `streamlit run web.py`
    """)
