import streamlit as st
import subprocess, sys, os, pathlib, tempfile, uuid
from PIL import Image

# ------------------------------------------------------------------
# Add local SimpleClick to PYTHONPATH so that infer_simpleclick can import
# ------------------------------------------------------------------
ROOT = pathlib.Path(__file__).resolve().parent
sys.path.append(str(ROOT / "SimpleClick"))

# ------------------------------------------------------------------
# Streamlit page configuration
# ------------------------------------------------------------------
st.set_page_config(page_title="Interactive Segmentation Demo", layout="wide")

st.title("Interactive Medical Image Segmentation: Transformer vs. CNN")

# Sidebar navigation
st.sidebar.header("Navigation")
page = st.sidebar.radio("Go to", ["Background", "Technique Comparison", "iSegFormer Model", "Demo", "References"])

# ------------------------------ Pages ---------------------------------
if page == "Background":
    st.header("Project Background")
    st.markdown(
        """
We explore **how interactive segmentation accelerates and improves medical diagnosis** by contrasting two paradigms:

* **Fully automatic segmentation** (CNN‑based, no clicks)
* **Interactive segmentation** (Transformer‑based, few user clicks)

Key papers analysed:
1. *iSegFormer* — Liu *et al.* 2022  
2. *UNet + Spatial Attention* — Zhang *et al.* 2021  
3. *SimpleClick* — Liu *et al.* 2023
"""
    )

elif page == "Technique Comparison":
    st.header("Automatic vs. Interactive Segmentation")
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Fully Automatic (UNet + Attention)")
        st.markdown("- Batch processing, stable\n- No human input, less flexible\n- Example: Zhang 2021")
    with col2:
        st.subheader("Interactive (Click‑based)")
        st.markdown("- Few clicks refine mask\n- High accuracy with sparse labels\n- Examples: iSegFormer, SimpleClick")

elif page == "iSegFormer Model":
    st.header("Inside iSegFormer")
    st.markdown("- **Core**: Swin Transformer encoder + MLP decoder\n- **Highlights**: memory‑efficient, slice propagation, good with limited data")
    st.image("https://raw.githubusercontent.com/uncbiag/iSegFormer/v1.0/figures/demo_gui.png", caption="iSegFormer interactive GUI", use_container_width=True)

elif page == "Demo":
    st.header("SimpleClick Online Demo")
    st.markdown("Upload a medical image (PNG/JPG) and run SimpleClick inference right in the browser.")

    uploaded = st.file_uploader("Choose image", type=["png","jpg","jpeg"])
    gpu_flag = st.checkbox("Use GPU (if available)", value=False)

    if uploaded:
        img = Image.open(uploaded).convert("RGB")
        st.image(img, caption="Input image", use_container_width=True)

        # save to temp
        tmp_dir = tempfile.mkdtemp()
        img_path = os.path.join(tmp_dir, f"{uuid.uuid4()}.png")
        img.save(img_path)

        if st.button("Run SimpleClick demo"):
            st.info("Running inference … please wait for the first time (weights may be downloaded).")
            ckpt_path = "./weights/simpleclick_models/cocolvis_vit_huge.pth"
            cmd = [
                "python3", "infer_simpleclick.py",
                "--input", img_path,
                "--output", tmp_dir,
                "--checkpoint", ckpt_path,
                "--model-name", "vit_huge",
            ]
            if gpu_flag:
                cmd += ["--gpu", "0"]
            else:
                cmd += ["--gpu", "-1"]

            result = subprocess.run(cmd, capture_output=True, text=True)
            if result.returncode != 0:
                st.error("Inference failed:\n" + result.stderr)
            else:
                # find overlay image
                overlay_file = None
                for f in os.listdir(tmp_dir):
                    if f.endswith("_overlay.png"):
                        overlay_file = os.path.join(tmp_dir, f)
                        break
                if overlay_file and os.path.exists(overlay_file):
                    st.image(overlay_file, caption="Segmentation Result", use_container_width=True)
                else:
                    st.warning("Inference finished but overlay not found. Check server logs.")
    else:
        st.info("Please upload an image.")

elif page == "References":
    st.header("References")
    st.markdown(
        """```bibtex
@inproceedings{Liu2022_iSegFormer,
  author    = {Liu, Qin and others},
  title     = {iSegFormer: Interactive Segmentation for 3D Knee MRI},
  booktitle = {MICCAI},
  year      = {2022}
}

@article{Zhang2021_UNetAttention,
  author  = {Zhang, ...},
  title   = {CNN-Based Fully Automated Segmentation with Spatial Attention},
  journal = {IEEE Trans. Med. Imaging},
  year    = {2021}
}

@inproceedings{Liu2023_SimpleClick,
  author    = {Liu, Qin and others},
  title     = {SimpleClick: Interactive Image Segmentation with Simple Vision Transformers},
  booktitle = {ICCV},
  year      = {2023}
}
```"""
    )

# Footer
<<<<<<< HEAD
st.markdown("---\n*Demo built with SimpleClick & iSegFormer. Author: **Yusen Liu***")
=======
st.markdown("---\n*Demo built with SimpleClick & iSegFormer. Author: **Yusen Liu***")
>>>>>>> 62ddd49e922b94ebafaddb171b68fe702639a06d
