import streamlit as st

# 页面配置
st.set_page_config(page_title="SimpleClick: Interactive Segmentation Demo", layout="wide")

# 标题和项目链接
st.title("SimpleClick: Interactive Image Segmentation Demo")
st.markdown(
    """
**SimpleClick** 是基于简单 Vision Transformer 的交互式图像分割方法，于 ICCV 2023 发表。

- 论文链接：[ICCV 2023 Paper](https://openaccess.thecvf.com/content/ICCV2023/html/Liu_SimpleClick_Interactive_Image_Segmentation_With_Simple_Vision_Transformers_ICCV_2023_paper.html)
- 代码仓库：[SimpleClick v1.0 (GitHub)](https://github.com/uncbiag/SimpleClick/tree/v1.0)
"""
)

# 侧边导航
st.sidebar.header("导航")
section = st.sidebar.radio("选择页面", ["简介", "示例演Demo", "安装指南", "使用方法", "引用"])

if section == "简介":
    st.header("方法简介")
    st.markdown(
        """
SimpleClick 结合了简洁的 Vision Transformer 架构和 BRS 交互策略，支持点击式前后台提示，生成高质量分割结果。架构图如下：
"""
    )
    framework_url = "https://github.com/uncbiag/SimpleClick/raw/v1.0/assets/simpleclick_framework.png"
    st.image(framework_url, caption="SimpleClick 架构示意图", use_column_width=True)

elif section == "示例演Demo":
    st.header("交互式分割演示")
    st.markdown(
        """
下面是一个示例演示 GIF：
"""
    )
    demo_url = "https://github.com/uncbiag/SimpleClick/raw/v1.0/assets/demo_sheep.gif"
    st.image(demo_url, caption="点击式分割示例 (Demo Sheep)", use_column_width=True)
    st.markdown(
        """
本地运行示例：
```bash
python3 demo.py --checkpoint=./weights/simpleclick_models/cocolvis_vit_huge.pth --gpu 0
```"""
    )

elif section == "安装指南":
    st.header("安装环境与依赖")
    st.markdown(
        """
```bash
# 克隆仓库
git clone https://github.com/uncbiag/SimpleClick.git
cd SimpleClick
# 安装依赖
pip3 install -r requirements.txt
# （可选）配置 CUDA 驱动与 Docker
```"""
    )
    st.markdown(
        """
依赖示例：Python3.8.8、PyTorch1.11.0、CUDA11.0 + torchvision。建议使用虚拟环境管理。
"""
    )

elif section == "使用方法":
    st.header("训练与评估")
    st.markdown(
        """
```bash
# 训练模型示例（Huge 模型）
python train.py models/iter_mask/plainvit_huge448_cocolvis_itermask.py \
  --batch-size=32 --ngpus=4

# 评估模型
python scripts/evaluate_model.py NoBRS --gpu=0 \
  --checkpoint=./weights/simpleclick_models/cocolvis_vit_huge.pth \
  --eval-mode=cvpr --datasets=GrabCut,Berkeley,...
```"""
    )
    st.markdown(
        "更多使用细节请参考 `config.yml` 和项目 README。"
    )

elif section == "引用":
    st.header("引用格式")
    st.markdown(
        """
```bibtex
@InProceedings{Liu_2023_ICCV,
    author    = {Liu, Qin and Xu, Zhenlin and Bertasius, Gedas and Niethammer, Marc},
    title     = {SimpleClick: Interactive Image Segmentation with Simple Vision Transformers},
    booktitle = {Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV)},
    month     = {October},
    year      = {2023},
    pages     = {22290-22300}
}
```"""
    )

# 底部版权信息
st.markdown(
    "---\n*本页面基于 [SimpleClick](https://github.com/uncbiag/SimpleClick/tree/v1.0) 项目构建，作者：Qin Liu 等。*"
)
