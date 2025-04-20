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

# Helpers
def load_asset(name, caption=None, width=700):
    path = os.path.join("assets", name)
    return st.image(path, caption=caption, use_column_width=True)

# 1. Overview
if page == "Overview":
    st.title("Accelerating Clinical Diagnosis with Interactive Segmentation")
    st.markdown("""
    In modern radiology, doctors rely on **CT**, **MRI**, and other 3D/2D scans to spot abnormalities.
    Advanced AI segmentation can automate this, but edge cases and fuzzy boundaries still pose challenges.

    **Interactive segmentation** bridges the gap by letting the doctor **click** or **mark** areas of interest,
    combining **AI speed** with **human expertise** for **higher accuracy**, **lower workload**, and **faster decisions**.
    """)
    load_asset("mri_example.jpg", caption="Figure 1: Knee MRI scan")

# 2. Background
elif page == "Background":
    st.title("Automated vs. Interactive Segmentation")
    st.markdown("""
    The landscape of medical image segmentation can be divided into two paradigms:

    1. **Fully Automated**  
       - **Pros:** Batch processing, no human in loop, consistent throughput  
       - **Cons:** May fail on special cases, cannot correct fuzzy boundaries  
    2. **Interactive (Semi‑automatic)**  
       - **Pros:** Human‑in‑the‑loop refinement, handles edge cases, personalized corrections  
       - **Cons:** Requires minimal user input, slightly lower throughput  
    """)
    load_asset("seg_pipeline.png", caption="Figure 2: Workflow comparison")

# 3. iSegFormer
elif page == "iSegFormer":
    st.title("iSegFormer (Liu et al., MICCAI 2022)")
    st.markdown("""
    **iSegFormer** tackles **3D knee MRI segmentation** with an **interactive** approach:
    - **Backbone:** Swin Transformer  
    - **Head:** Lightweight MLP for fast mask prediction  
    - **Interactive Loop:** Doctor annotates/adjusts a few slices → model refines  
    - **Benefits:**  
      - Achieves **>90% Dice** with few annotations  
      - Reduces annotation time by **60%**  
    - **Limitations:** High GPU memory usage for 3D volumes
    """)
    load_asset("architecture.png", caption="Figure 3: iSegFormer architecture")

# 4. SimpleClick
elif page == "SimpleClick":
    st.title("SimpleClick (Liu et al., CVPR 2023)")
    st.markdown("""
    **SimpleClick** brings click‑style interaction to **2D images**:
    - **Backbone:** Vision Transformer (ViT)  
    - **Interaction:**  
      1. **Positive click** on object → coarse mask  
      2. **Negative click** on background → refine mask  
      3. Iterate until satisfied  
    - **Results:**  
      - State‑of‑the‑art on natural & medical datasets  
      - **>80 FPS** on a single GPU  
      - User studies show **80% fewer** clicks needed
    """)
    load_asset("simpleclick_workflow.png", caption="Figure 4: SimpleClick interaction flow")

# 5. Demo
elif page == "Demo":
    st.title("Static Demo: Center‑Click Segmentation")
    st.info("*This demo runs a single center click; full interactive version coming soon.*")
    uploaded = st.file_uploader("Upload an image", type=["jpg","jpeg","png"])
    use_gpu = st.checkbox("Use GPU", value=False)

    if uploaded:
        tmp_dir = tempfile.mkdtemp()
        in_path = os.path.join(tmp_dir, f"{uuid.uuid4()}.png")
        with open(in_path, "wb") as f:
            f.write(uploaded.read())
        st.image(in_path, caption="Input", use_column_width=True)

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
                    st.image(out, caption="Overlay", use_column_width=True)
                else:
                    st.error("Overlay not found. Check logs.")

# 6. Installation
elif page == "Installation":
    st.title("Installation & Usage")
    st.markdown("""
    1. **Clone repository**  
       ```bash
       git clone https://github.com/yourname/my-streamlit-demo.git
       cd my-streamlit-demo
       ```
    2. **Create environment & install**  
       ```bash
       pip install -r requirements.txt
       ```
    3. **Place assets**  
       - `assets/mri_example.jpg`  
       - `assets/seg_pipeline.png`  
       - `assets/architecture.png`  
       - `assets/simpleclick_workflow.png`
    4. **Run the app**  
       ```bash
       streamlit run web.py
       ```
    """)
