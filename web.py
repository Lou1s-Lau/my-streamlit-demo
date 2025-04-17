import streamlit as st

# 页面配置
st.set_page_config(page_title="Interactive Medical Segmentation Overview", layout="wide")

# 顶部标题
st.title("基于 Transformer 与 CNN 的交互式医学图像分割比较与演示")

# 侧边导航
st.sidebar.header("导航")
page = st.sidebar.radio("选择页面", ["项目背景", "技术对比", "iSegFormer 模型", "Demo 演示", "参考文献"])

# 各页面内容
if page == "项目背景":
    st.header("项目背景")
    st.markdown(
        """
在回答“交互式分割如何帮助医生更快更准地诊断？”这一问题时，我们参考了多篇研究，从自动分割与交互式分割两种技术方向进行对比，探讨 AI 在医学图像处理中的应用。

- **自动分割**：依赖 CNN 等模型无人工干预。
- **交互式分割**：结合医生点击/标注与模型推理，提高准确性与灵活性。

本文重点分析了以下方法：
1. iSegFormer (Liu et al., 2022)
2. UNet + 空间注意力 (Zhang et al., 2021)
3. SimpleClick (Liu et al., 2023)
        """
    )

elif page == "技术对比":
    st.header("自动分割 vs 交互式分割 对比")
    st.beta_columns(2)
    col1, col2 = st.beta_columns(2)
    with col1:
        st.subheader("自动分割 (UNet + Attention)")
        st.markdown(
            """
- 无需人工干预，批量处理高效
- 稳定性强，但对模糊边界鲁棒性差
- 典型代表：Zhang et al. (2021)
"""
        )
    with col2:
        st.subheader("交互式分割 (点击式交互)")
        st.markdown(
            """
- 医生可快速调整分割结果
- 少量标注即可获得高精度
- 典型代表：iSegFormer, SimpleClick
"""
        )

elif page == "iSegFormer 模型":
    st.header("iSegFormer 模型解析")
    st.markdown(
        """
- **核心构建**：Swin Transformer 编码器 + 轻量 MLP 解码器
- **特点**：内存高效、支持 3D 切片传播、少数据微调即可高精度
- **应用**：膝关节 MRI 交互式分割
"""
    )
    # 展示 iSegFormer GUI 演示截图
    img_url = "https://raw.githubusercontent.com/uncbiag/iSegFormer/v1.0/figures/demo_gui.png"
    st.image(img_url, caption="iSegFormer 交互式 GUI 示例", use_container_width=True)

elif page == "Demo 演示":
    st.header("Demo 演示")
    st.markdown(
        """
下面展示 SimpleClick 的交互式演示 GIF，供快速体验点击式标注的效果：
"""
    )
    gif_url = "https://github.com/uncbiag/SimpleClick/raw/v1.0/assets/demo_sheep.gif"
    st.image(gif_url, caption="SimpleClick 点击式分割示例", use_container_width=True)
    st.markdown(
        "`python3 demo.py --checkpoint=./weights/simpleclick_models/cocolvis_vit_huge.pth --gpu 0`"
    )

elif page == "参考文献":
    st.header("参考文献")
    st.markdown(
        """
```bibtex
@InProceedings{Liu2022_iSegFormer,
  author = {Liu, Qin and ...},
  title = {iSegFormer: Interactive Segmentation for 3D Knee MRI},
  booktitle = {MICCAI},
  year = {2022}
}

@article{Zhang2021_UNetAttention,
  author = {Zhang, ...},
  title = {CNN-Based Fully Automated Segmentation with Spatial Attention},
  journal = {IEEE Trans. Med. Imag.},
  year = {2021}
}

@InProceedings{Liu2023_SimpleClick,
  author = {Liu, Qin and ...},
  title = {SimpleClick: Interactive Image Segmentation with Simple Vision Transformers},
  booktitle = {ICCV},
  year = {2023}
}
```"""
    )

# 底部说明
st.markdown(
    "---\n*本页面基于 iSegFormer (v1.0) 与 SimpleClick 项目示例构建，用于展示交互式医学图像分割技术；作者：Meijia Wang*"
)
