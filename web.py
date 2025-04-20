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
    st.title("Background: Deep Learning & Medical Image Segmentation")
    st.markdown("""
    **1. 什么是深度学习？**  
    - 深度学习是一种通过“神经网络”模拟人脑信息处理的技术。  
    - 它可以自动**从大量数据中学习特征**，例如图像中的边缘、纹理甚至复杂结构。  
    - 在医学影像领域，深度学习能帮助我们**精确定位并分割**身体器官、病灶等。

    **2. 经典模型：CNN vs. Transformer**  
    - **卷积神经网络（CNN）**  
      - 通过“卷积核”滑动扫描图像，善于捕捉局部图案（如肿块边缘）。  
      - UNet、ResNet 等是常见架构，已广泛应用于自动化分割任务。  
    - **视觉 Transformer (ViT)**  
      - 最初用于自然语言处理，将图像**切成小块**后，用“自注意力”机制学习全局关联。  
      - 擅长捕捉远距离像素之间的依赖，适用于复杂结构（如 3D MRI 的多层次信息）。

    **3. 图像分割：自动 vs. 交互式**  
    - **全自动分割**  
      - 医生只需上传批量影像，模型一次性跑完。  
      - 优点：效率高，不需人工干预。  
      - 缺点：遇到**模糊边界**或**少见病灶**时，结果可能不准确。  
    - **交互式分割**  
      - 医生在关键点**点击**或**涂抹**（正/负样本）进行微调。  
      - 优点：可在**特殊病例**、**边界不清晰**时快速校正，提高准确率。  
      - 缺点：需要少量用户操作，但通常只需几次点击即可达成。

    **4. 为什么在医学影像中尤为重要？**  
    - 医学影像 (MRI、CT) 属于**高分辨率、三维体积**数据，手工标注耗时长。  
    - 自动化虽能加速，但难以应对所有**异常情况**。  
    - 交互式工具让医生用极少量操作结合专业知识，**在几秒钟内**获得高质量分割。

    **5. 小结示意图**  
    - 下图展示了全自动与交互式分割在常见流程中的对比：  
    """)
    load_asset("seg_pipeline.png", caption="Figure 2: Automated vs. Interactive Workflow")

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
