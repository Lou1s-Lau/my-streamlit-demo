import streamlit as st
from PIL import Image
import tempfile, subprocess, uuid, os

# 页面配置
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

# Helper：安全加载本地或远程图像
def load_asset(name, caption=None):
    local = os.path.join("assets", name)
    if os.path.exists(local):
        st.image(local, caption=caption, use_container_width=True)
    else:
        st.warning(f"Asset `{name}` not found in `/assets`. You can add it or replace with a URL.")
        # 示例：如果你有线上图片链接，可以这样加载：
        # st.image("https://your.cdn.com/path/to/" + name, caption=caption, use_container_width=True)

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
    - Task: Interactive 3D knee MRI segmentation  
    - Backbone: Swin Transformer + lightweight MLP  
    - Key Insight: Fine‑tune on small data → high accuracy & efficiency  
    """)
    load_asset("architecture.png", caption="Figure 3: iSegFormer architecture")

# 4. SimpleClick
elif page == "SimpleClick":
    st.title("SimpleClick (Liu et al., CVPR 2023)")
    st.markdown("""
    - Task: Click‑based 2D image segmentation  
    - Backbone: Vision Transformer (ViT)  
    - Workflow: Positive/negative clicks → iterative refinement  
    """)
    load_asset("simpleclick_workflow.png", caption="Figure 4: SimpleClick workflow")

# 5. Demo (Static center‑click)
elif page == "Demo":
    st.title("Static Demo: Center‑Click Segmentation")
    st.info("*This demo uses a single center click.*")
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
    1. Clone repo and `cd my-streamlit-demo`  
    2. `pip install -r requirements.txt`  
    3. Place your images in `assets/` or use URLs  
    4. `streamlit run web.py`
    """)
