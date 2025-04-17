import streamlit as st

# Page configuration
st.set_page_config(page_title="Interactive Medical Segmentation Overview", layout="wide")

# Main title
st.title("Interactive Medical Image Segmentation: Transformer vs. CNN")

# Sidebar navigation
st.sidebar.header("Navigation")
page = st.sidebar.radio("Go to", ["Background", "Technique Comparison", "iSegFormer Model", "Demo", "References"])

# Content for each page
if page == "Background":
    st.header("Project Background")
    st.markdown(
        """
When addressing **"How can interactive segmentation help doctors diagnose faster and more accurately?"** we analysed research from two perspectives—**fully automatic segmentation** and **interactive segmentation**—to examine how AI assists diagnosis in medical imaging.

- **Fully automatic segmentation**: CNN‑based models that process images without human involvement.
- **Interactive segmentation**: combines quick user clicks/annotations with model inference, allowing doctors to refine results and boost precision.

Key papers discussed:

1. **iSegFormer** – Liu *et al.* (2022)
2. **UNet + Spatial Attention** – Zhang *et al.* (2021)
3. **SimpleClick** – Liu *et al.* (2023)
        """
    )

elif page == "Technique Comparison":
    st.header("Automatic vs. Interactive Segmentation")
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Fully Automatic (UNet + Attention)")
        st.markdown(
            """
- No human intervention; efficient batch processing  
- High stability but less robust to fuzzy boundaries  
- Representative work: Zhang *et al.* (2021)
"""
        )

    with col2:
        st.subheader("Interactive (Click‑based)")
        st.markdown(
            """
- Clinicians quickly refine segmentation with a few clicks  
- Achieves high accuracy with minimal annotations  
- Representative work: iSegFormer, SimpleClick
"""
        )

elif page == "iSegFormer Model":
    st.header("Inside the iSegFormer Architecture")
    st.markdown(
        """
- **Core**: Swin Transformer encoder + lightweight MLP decoder  
- **Highlights**: memory‑efficient, slice‑to‑volume propagation, fine‑tunes well with sparse labels  
- **Target**: 3‑D knee MRI interactive segmentation
"""
    )
    img_url = "https://raw.githubusercontent.com/uncbiag/iSegFormer/v1.0/figures/demo_gui.png"
    st.image(img_url, caption="iSegFormer interactive GUI", use_container_width=True)

elif page == "Demo":
    st.header("Quick Demo")
    st.markdown(
        """
The GIF below demonstrates click‑based interaction with **SimpleClick**:
"""
    )
    gif_url = "https://github.com/uncbiag/SimpleClick/raw/v1.0/assets/demo_sheep.gif"
    st.image(gif_url, caption="SimpleClick click‑based segmentation", use_container_width=True)
    st.markdown(
        "Run locally with:\n```
python3 demo.py --checkpoint ./weights/simpleclick_models/cocolvis_vit_huge.pth --gpu 0
```"
    )

elif page == "References":
    st.header("References")
    st.markdown(
        """
```bibtex
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
```
"""
    )

# Footer
st.markdown(
    "---\n*Page built from iSegFormer (v1.0) and SimpleClick examples. Author: **Yusen Liu***"
)
