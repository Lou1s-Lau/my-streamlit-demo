import streamlit as st
from PIL import Image
import tempfile, subprocess, uuid, os

# page config
st.set_page_config(
    page_title="Interactive Medical Image Segmentation",
    layout="wide",
    initial_sidebar_state="expanded",
)

# sidebar navigation
st.sidebar.markdown("# 🏥 Medical AI")
page = st.sidebar.radio("Navigate", [
    "Overview", "Background", "iSegFormer", "SimpleClick", "Demo", "References"
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
    to spot subtle abnormalities. While fully automated AI tools can
    segment organs or lesions at scale, they sometimes struggle with
    fuzzy boundaries or rare pathologies.

    **Interactive segmentation** bridges the gap by letting physicians
    guide the algorithm—clicking or marking key regions—combining
    **AI speed** with **clinical expertise** for **higher accuracy**
    and **faster decision‑making**.
    """)
    load_asset("mri_example.jpg", caption="Figure 1: Knee MRI scan")
    st.markdown("""
    **Figure 1: Tri‑planar MRI of the Knee**

    This composite image shows three standard MRI views of the knee:

    1. **Axial (left):** Horizontal slice through femoral condyles and tibial plateau.  
    2. **Coronal (center):** Front‑to‑back slice, ideal for cartilage thickness and joint space.  
    3. **Sagittal (right):** Side slice highlighting cruciate ligaments and patellofemoral joint.

    These multi‑planar images serve as inputs to interactive models
    like **iSegFormer** and **SimpleClick**, enabling precise
    delineation of cartilage, menisci, and ligaments.
    """)

# 2. Background
elif page == "Background":
    st.title("Background: Deep Learning & Medical Image Segmentation")
    st.markdown("""
    **Deep Learning Basics**  
    - Deep learning uses layered “neural networks” to automatically
      learn features directly from data.  
    - In imaging, it can detect edges, textures, and complex patterns
      without manual feature design.

    **Key Architectures**  
    - **Convolutional Neural Networks (CNNs):**  
      Use learnable filters to scan images. Great for local details
      like edges or small lesions. UNet and its variants are
      widely adopted for fully automated segmentation.  
    - **Vision Transformers (ViT):**  
      Split an image into patches and apply a self‑attention mechanism
      to capture global context. Powerful for modeling long‑range
      dependencies, especially in 3D data like MRI volumes.

    **Segmentation Paradigms**  
    - **Fully Automatic:**  
      - *Pros:* Zero user effort, batch processing, high throughput.  
      - *Cons:* May missegment in low‑contrast or uncommon cases.  
    - **Interactive / Semi‑automatic:**  
      - *Pros:* Clinician provides clicks or strokes to guide the model,
        quickly corrects mistakes.  
      - *Cons:* Requires minimal input—often only a few clicks.

    **Why Interactive in Medicine?**  
    - Medical scans (e.g. MRI, CT) often involve 3D volumes and high
      resolution; manual annotation is labor‑intensive.  
    - Fully automatic tools speed up routine tasks but can fail on
      rare pathologies or ambiguous boundaries.  
    - Interactive methods empower doctors to merge their domain
      knowledge with AI, achieving accurate results in seconds.

    Below is a schematic comparing the two workflows.
    """)
    load_asset("seg_pipeline.png", caption="Figure 2: Automated vs. Interactive Workflow")

# 3. iSegFormer
elif page == "iSegFormer":
    st.title("iSegFormer (Liu et al., MICCAI 2022)")
    st.markdown("""
    iSegFormer introduces **interactive segmentation** for **3D knee MRI**:
    - **Encoder:** Swin Transformer for powerful global context modeling.  
    - **Decoder:** Lightweight MLP to produce mask predictions efficiently.  
    - **Workflow:** Doctor annotates or adjusts a few slices → model propagates 
      and refines across the volume.  
    - **Results:** Over 90% Dice with minimal manual labeling.  
    - **Trade‑off:** Higher GPU memory requirements due to 3D data.
    """)
    load_asset("architecture.png", caption="Figure 3: iSegFormer Architecture")

# 4. SimpleClick
elif page == "SimpleClick":
    st.title("SimpleClick (Liu et al., CVPR 2023)")
    st.markdown("""
    SimpleClick brings **click‑based** interactive segmentation to **2D images**:
    1. **Positive click** on object region → coarse mask.  
    2. **Negative click** on background → mask refinement.  
    3. **Iterate** until the segmentation is satisfactory.

    - **Backbone:** Vision Transformer (ViT) for global context.  
    - **Efficiency:** >80 FPS on a single GPU.  
    - **Accuracy:** State‑of‑the‑art on both natural and medical datasets.
    """)
    load_asset("simpleclick_workflow.png", caption="Figure 4: SimpleClick Interaction Flow")

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
    1. Liu, Q., Xu, Z., Jiao, Y., & Niethammer, M. (2022). *iSegFormer: Interactive segmentation via transformers with application to 3D knee MRI images*. MICCAI 2022.  
    2. Zhang, X., Li, Z., Shi, H., Deng, Y., Zhou, G., & Tang, S. (2021). *A deep learning‑based method for knee articular cartilage segmentation in MRI images*. ICCAIS 2021.  
    3. Liu, Q., Xu, Z., Bertasius, G., & Niethammer, M. (2023). *SimpleClick: Interactive Image Segmentation with Simple Vision Transformers*. ICCVW 2023.  
    4. Marinov, Z., Jäger, P. F., Egger, J., Kleesiek, J., & Stiefelhagen, R. (2024). *Deep interactive segmentation of medical images: A systematic review and taxonomy*. IEEE TPAMI, 46(12), 10998–11039.  
    5. Huang, M., Zou, J., Zhang, Y., Bhatti, U. A., & Chen, J. (2024). *Efficient click‑based interactive segmentation for medical image with improved Plain‑ViT*. IEEE JBHI.  
    6. Luo, X., Wang, G., Song, T., Zhang, J., Aertsen, M., Deprest, J., Ourselin, S., Vercauteren, T., & Zhang, S. (2021). *MIDeepSeg: Minimally Interactive Segmentation of Unseen Objects from Medical Images Using Deep Learning*. arXiv:2104.12166.
    """)
