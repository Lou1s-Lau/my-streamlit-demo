import os, sys
# 假设 web.py 和 SimpleClick-1.0 在同一级目录
ROOT = os.path.dirname(__file__)
sys.path.insert(0, os.path.join(ROOT, "SimpleClick-1.0"))

import streamlit as st
from PIL import Image
import numpy as np
import tempfile, subprocess, uuid, os
from streamlit_drawable_canvas import st_canvas
import tempfile, uuid, os
# 不要在顶部 import cv2/torch/gdown，留到后面 Demo 里再导入


# 新增：绘图画布组件
from streamlit_drawable_canvas import st_canvas

# 页面全局配置
st.set_page_config(
    page_title="Interactive Medical Image Segmentation",
    layout="wide",
    initial_sidebar_state="expanded",
)

# 侧边栏导航，新增 Interactive Demo，Static Demo 改名 Demo
st.sidebar.markdown("# 🏥 Medical AI")
page = st.sidebar.radio("Navigate", [
    "Overview", "Background", "iSegFormer", "Interactive Demo", "Demo", "References"
])

def load_asset(name, caption=None):
    path = os.path.join("SimpleClick-1.0", "assets", name)
    if os.path.exists(path):
        st.image(path, caption=caption, use_container_width=True)
    else:
        st.warning(f"Asset `{name}` not found at `{path}`. Please upload it there.")
import torch
import streamlit as st

@st.cache_resource(show_spinner=False)
def load_predictor(checkpoint_path: str, use_gpu: bool):
    """
    checkpoint_path: 权重文件路径
    use_gpu:       用户是否勾选使用 GPU
    """
    # 根据 use_gpu 决定用 CPU 还是 GPU
    device = torch.device("cuda" if use_gpu and torch.cuda.is_available() else "cpu")
    from infer_simpleclick import build_predictor
    # 这里要传入 device
    return build_predictor(checkpoint_path, device)



# 1. Overview
if page == "Overview":
    st.title("Accelerating Clinical Diagnosis with Interactive Segmentation")
    st.markdown("""
    Medical imaging techniques like **Computed Tomography (CT)** and **Magnetic Resonance Imaging (MRI)** are cornerstones of modern medicine, providing invaluable visual information for diagnosis, treatment planning, and monitoring disease progression. A critical step in analyzing these images is **segmentation**—precisely outlining anatomical structures or pathological findings (like tumors or lesions). Accurate segmentation allows doctors to measure volumes, assess morphology, and track changes over time, directly impacting clinical decisions.

    **Artificial Intelligence (AI)**, particularly deep learning, has driven significant progress in **fully automated segmentation**. These tools promise to analyze large volumes of scan data rapidly and consistently. However, purely automated approaches often encounter difficulties in real-world clinical scenarios:
    * They can **struggle with ambiguous or fuzzy boundaries**, where the contrast between tissues is low, or where noise and artifacts obscure details (e.g., infiltrative tumor margins).
    * They may perform poorly on **rare pathologies** or anatomical variations not well-represented in their training datasets, as their knowledge is limited to the patterns they have previously seen (Zhang et al., 2021). Relying solely on potentially inaccurate automated results in such cases can risk misdiagnosis or flawed treatment planning.

    **Interactive segmentation** emerges as a powerful solution, creating a synergy between **human expertise** and **AI efficiency**. Instead of relying on a fully automated output or resorting to time-consuming manual outlining slice-by-slice, this approach allows clinicians to guide the AI algorithm with minimal, targeted feedback.
    * Typically, a physician provides simple inputs, such as a **positive click** inside the region of interest (e.g., a lesion) and perhaps a **negative click** just outside it.
    * The AI algorithm uses these cues to instantly update its understanding and **refine the segmentation contour in real-time**. This human-in-the-loop process leverages the clinician's anatomical knowledge and ability to interpret subtle or unusual findings, while the AI handles the laborious task of delineating the precise boundary.

    This collaborative approach significantly reduces the time needed for accurate segmentation, especially in complex cases, compared to manual methods, while ensuring higher reliability than fully automated techniques in challenging situations (Liu et al., 2022). It empowers doctors to achieve high-quality segmentations faster, leading to more timely and confident diagnoses.

    Below, Figure 1 illustrates a tri-planar knee MRI. This type of imaging often presents challenges for segmentation due to the complex arrangement of tissues like cartilage and bone, where boundaries can sometimes be indistinct. Interactive tools can be particularly helpful in accurately delineating these structures.
    """, unsafe_allow_html=True) # Keep unsafe_allow_html=True if you have specific HTML needs, otherwise it might be optional.

    # Assuming load_asset function loads and displays the image
    # load_asset("mri_example.jpg", caption="Figure 1: Tri-planar MRI of the Knee")
    load_asset("mri_example.jpg", caption="Figure 1: Tri-planar MRI of the Knee")
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
    Deep learning has profoundly transformed medical image analysis. By employing multi-layer neural networks, these techniques automatically learn hierarchical features directly from image data—progressing from simple edges and textures in early layers to complex anatomical structures or pathological patterns in deeper layers. This capability is central to advancing image segmentation for clinical applications.

    Within this domain, two primary paradigms exist: fully automated segmentation and interactive segmentation, often leveraging distinct deep learning architectures like Convolutional Neural Networks (CNNs) and Vision Transformers (ViTs).

    ### Fully Automated Segmentation: Efficiency and Challenges

    **Convolutional Neural Networks (CNNs)** have long been the workhorse for **fully automated** medical image segmentation. Architectures like the U-Net excel at capturing local spatial patterns using learnable convolutional kernels. A notable example is the work by **Zhang et al. (2021)**, who utilized a CNN-based U-Net enhanced with a **spatial attention mechanism** for automated knee cartilage segmentation. This attention mechanism mimics human visual focus, allowing the model to concentrate on critical image regions, thereby improving accuracy.

    The primary advantage of fully automated methods lies in their efficiency and consistency. They can process large batches of images without human intervention, streamlining workflows. However, their performance can degrade when faced with certain challenges:
    * **Ambiguous or Fuzzy Boundaries:** Low contrast between tissues, such as tumor margins obscured by edema, can confuse automated algorithms.
    * **Image Quality Issues:** Noise or motion artifacts can hinder accurate segmentation.
    * **Data Scarcity for Rare Conditions:** Uncommon pathologies or anatomical variations might not be well-represented in the training data, leading to suboptimal performance (Marinov et al., 2024).
    * **Limited Global Context:** Traditional CNNs primarily focus on local features and may struggle with lesions spanning large areas or requiring understanding of long-range spatial relationships within the image volume.

    ### The Rise of Transformers and Interactive Segmentation

    To address the limitations related to long-range dependencies and global context, **Vision Transformers (ViTs)** have emerged as a powerful alternative. ViTs operate by dividing an image into patches and utilizing a **self-attention mechanism**. This allows the model to weigh the importance of different image regions relative to each other, effectively capturing global context. This capability is particularly beneficial in volumetric medical images like 3D MRI scans, where anatomical structures span multiple slices (Liu et al., 2023).

    While ViTs can be used for fully automated segmentation, they have also become foundational for advanced **interactive segmentation** methods. Image segmentation itself involves assigning a class label (e.g., "bone," "cartilage," "tumor," "background") to every pixel or voxel, generating precise outlines crucial for diagnosis, treatment planning, or monitoring disease progression.

    **Interactive segmentation** bridges the gap between automated efficiency and clinical nuance by incorporating expert knowledge directly into the segmentation loop. It allows clinicians to guide the algorithm, particularly for challenging cases. This "human-in-the-loop" approach typically involves simple user interactions:
    * Providing **positive clicks** inside the target region of interest.
    * Providing **negative clicks** in the background area just outside the target.

    These interactions provide real-time feedback to the model, which then updates its predictions to refine the segmentation mask. This process allows for correction of errors in edge cases—like distinguishing a tiny metastatic nodule adjacent to bone—often requiring only a few clicks and minimal time per image (Huang et al., 2024; Luo et al., 2021).

    ### Interactive Models in Practice: Balancing Accuracy, Speed, and Resources

    Several studies highlight the potential and nuances of interactive methods:

    * The **iSegFormer** model (**Liu et al., 2022**) exemplifies an interactive approach for 3D knee MRI segmentation. Built upon the **Swin Transformer**, it uses interaction to achieve high accuracy even with limited annotated data, requiring only fine-tuning. This demonstrates how interaction can compensate for smaller datasets. However, the authors note that such sophisticated models might demand significant computational resources (e.g., large memory), potentially posing implementation barriers in resource-constrained clinical settings.
    * In contrast, the **SimpleClick** method (**Liu et al., 2023**) focuses on efficiency and usability. Using a standard Vision Transformer, it enables clinicians to achieve high-quality segmentation with just a few intuitive clicks. The study highlights its state-of-the-art performance and strong generalizability to medical images, crucially noting its efficiency in terms of computation speed and resource consumption. This makes it particularly promising for busy hospital environments where diagnostic speed is critical for timely treatment decisions.

    ### The Spectrum of Segmentation Approaches

    In summary, fully automated and interactive segmentation methods represent a spectrum of tools. Fully automated systems, exemplified by the CNN-based approach of Zhang et al. (2021), offer high throughput and consistency for standardized tasks. Interactive systems, such as iSegFormer and SimpleClick leveraging Transformer architectures, provide clinician-driven precision, adaptability to ambiguous cases, and the ability to leverage expert knowledge dynamically. They excel where automated methods might falter but require user input. The choice between them—or a combination thereof—depends on the specific clinical application, the complexity of the images, and the available resources. Ultimately, both approaches aim to enhance the speed and accuracy of medical diagnoses derived from imaging data.
    """)
        # —— Side-by-side comparison: two columns for images —— 
    st.markdown("### Fully Automatic vs. Interactive Segmentation Workflow")
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("**Fully Automatic Segmentation**")
        img_auto = os.path.join("SimpleClick-1.0", "assets",
                                "automatic.png")
        if os.path.exists(img_auto):
            st.image(img_auto, use_container_width=True)
        else:
            st.warning(f"Image not found: {img_auto}")

    with col2:
        st.markdown("**Interactive Segmentation**")
        img_inter = os.path.join("SimpleClick-1.0", "assets",
                                 "interactive.png")
        if os.path.exists(img_inter):
            st.image(img_inter, use_container_width=True)
        else:
            st.warning(f"Image not found: {img_inter}")



# 3. iSegFormer
elif page == "iSegFormer":
    st.title("iSegFormer: Interactive 3D Segmentation (Liu et al., MICCAI 2022)")
    st.markdown("""
    The **iSegFormer** framework, presented by Liu et al. at the top-tier **MICCAI 2022** conference, represents a significant advancement in **interactive segmentation**, specifically designed for the complexities of **3D knee MRI scans**. Segmenting structures like cartilage in 3D MRI is challenging due to intricate surface curvatures, thin structures, and often indistinct boundaries where cartilage blends into adjacent tissues like bone or meniscus. Traditional 2D slice-by-slice methods often fail to capture the full 3D context, while fully automated 3D methods can struggle with accuracy, especially when training data is limited.

    **Architectural Highlights:**

    iSegFormer tackles these challenges by intelligently combining cutting-edge deep learning components:
    * **Hierarchical Swin Transformer Encoder:** Instead of a standard CNN, iSegFormer leverages the **Swin Transformer**. This architecture is particularly well-suited for medical images because it processes image data hierarchically (capturing features at different scales, similar to CNNs) but uses a shifted-window self-attention mechanism. This allows it to model **long-range dependencies** across the entire 3D volume more efficiently than standard Vision Transformers, capturing the global context crucial for understanding anatomical continuity across slices.
    * **Lightweight MLP Decoder:** Complementing the powerful encoder is a simple, **lightweight Multi-Layer Perceptron (MLP) decoder**. The choice of a lightweight decoder is crucial for interactivity; it allows the model to **rapidly generate or update the segmentation mask** in response to user input without introducing significant computational delay during the interactive refinement process.

    **Interactive Mechanism and Efficiency:**

    The core idea behind iSegFormer's interactivity is **efficiency through sparse guidance**. Rather than requiring dense annotations or corrections on every slice, the clinician provides minimal input:
    * The user typically provides **positive and negative clicks** on only a **few key slices** within the 3D volume where the initial automated segmentation might be inaccurate.
    * The Swin Transformer's **global attention mechanism** then takes over. It effectively **propagates the contextual information** from these sparse user corrections throughout the interconnected 3D volume. A correction made on one slice informs the segmentation refinement on adjacent and even more distant slices, ensuring anatomical consistency.

    **Performance and Value Proposition:**

    As demonstrated by Liu et al. (2022), this approach yields impressive results:
    * It achieves high segmentation accuracy, with reported **Dice Similarity Coefficients (DSC) often exceeding 90%** for knee cartilage.
    * Crucially, this accuracy is obtained with **minimal manual effort**, significantly reducing the clinician's interaction time compared to fully manual segmentation or slice-by-slice refinement.
    * It shows that interactive methods can effectively **compensate for limited annotated training data**. By fine-tuning with just a small amount of interactive guidance, iSegFormer can reach accuracy levels comparable to or even exceeding fully automated models trained on larger datasets.

    **Computational Considerations:**

    Despite its effectiveness, iSegFormer highlights a common trade-off with advanced 3D transformer models:
    * Processing entire 3D volumes with self-attention mechanisms is **computationally intensive**. It demands **substantial GPU memory**, often necessitating high-end graphics cards typically found in research settings rather than standard clinical workstations.
    * The **model initialization time** can also be longer compared to simpler models.
    * These resource requirements could pose a **barrier to widespread adoption** in routine clinical workflows, particularly in environments with limited computational infrastructure.

    In essence, iSegFormer showcases the power of Transformers for context-aware, interactive 3D segmentation, offering high accuracy with minimal user input, but its practical deployment requires careful consideration of the necessary hardware resources.
    """, unsafe_allow_html=True)

    # Assuming load_asset function loads and displays the image
    # load_asset("architecture.jpg", caption="Figure 3: iSegFormer Architecture")
    load_asset("architecture.jpg", caption="Figure 3: iSegFormer Architecture")
#4   
elif page == "Interactive Demo":
    st.title("Interactive Segmentation Demo")

    # —— 用户选择是否用 GPU —— 
    use_gpu = st.checkbox("Use GPU for interactive demo", value=False)

    # —— 1. 上传图像 —— 
    uploaded = st.file_uploader("Upload a medical image", type=["png","jpg","jpeg"])
    if not uploaded:
        st.info("Please upload an image to begin.")
        st.stop()

    img = Image.open(uploaded).convert("RGB")
    img_np = np.array(img)

    # —— 2. 延迟加载模型 —— 
    # 注意：现在 load_predictor 需要两个参数
    predictor = load_predictor(
        "./weights/simpleclick_models/cocolvis_vit_huge.pth",
        use_gpu
    )

    # —— 3. 点击类型选择 —— 
    click_type = st.radio("Click type", ["Positive (foreground)", "Negative (background)"])

    # 4. 初始化点击列表
    if "clicks" not in st.session_state:
        st.session_state.clicks = []

    # 5. 绘图画布：只允许“点”操作
    canvas_result = st_canvas(
        background_image=img,
        update_streamlit=True,
        drawing_mode="point",
        stroke_color="#0f0" if click_type.startswith("Positive") else "#f00",
        stroke_width=20,
        key="seg_canvas",
        height=img_np.shape[0],
        width=img_np.shape[1],
    )

    # 6. 记录最新一次点击
    if canvas_result.json_data and canvas_result.json_data.get("objects"):
        for obj in canvas_result.json_data["objects"][-1:]:
            x, y = obj["path"][-1]
            st.session_state.clicks.append((int(x), int(y), click_type.startswith("Positive")))

    # 7. 布局：左侧点击历史，右侧分割预览
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Click History")
        for i, (x, y, is_pos) in enumerate(st.session_state.clicks, 1):
            mark = "🟢" if is_pos else "🔴"
            st.write(f"{i}. {mark} at ({x}, {y})")
        if st.button("🔄 Reset Clicks"):
            st.session_state.clicks = []

    with col2:
        st.subheader("Segmentation Preview")
        if st.button("Run / Update Segmentation"):
            mask = predictor.get_prediction(img_np, st.session_state.clicks)
            overlay = img_np.copy()
            overlay[mask > 0] = [255, 0, 0]
            st.image(
                [img_np, overlay],
                caption=["Input Image", "Overlay Result"],
                use_container_width=True
            )



# 5. Demo（保持静态中心点Demo）
elif page == "Demo":
    st.title("Static Demo: Center‑Click Segmentation")

    video_path = os.path.join("SimpleClick-1.0", "assets", "demo_oaizib_stcn_with_cycle.mp4")
    if os.path.exists(video_path):
        st.video(video_path, format="video/mp4", start_time=0)
    else:
        st.warning(f"Demo video not found at {video_path}")

    st.info("*The full interactive version is in the 'Interactive Demo' tab.*")
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

# 6. References
elif page == "References":
    st.title("References")
    st.markdown("""
    1. Liu, Q., Xu, Z., Jiao, Y., & Niethammer, M. (2022). *iSegFormer: Interactive segmentation via transformers with application to 3D knee MRI images*. MICCAI 2022.  
    2. Zhang, X., Li, Z., Shi, H., Deng, Y., Zhou, G., & Tang, S. (2021). *A deep learning‑based method for knee articular cartilage segmentation in MRI images*. ICCAIS 2021.  
    3. Liu, Q., Xu, Z., Bertasius, G., & Niethammer, M. (2023). *SimpleClick: Interactive Image Segmentation with Simple Vision Transformers*. ICCV 2023.  
    4. Marinov, Z., Jäger, P. F., Egger, J., Kleesiek, J., & Stiefelhagen, R. (2024). *Deep interactive segmentation of medical images: A systematic review and taxonomy*. IEEE TPAMI, 46(12), 10998–11039.  
    5. Huang, M., Zou, J., Zhang, Y., Bhatti, U. A., & Chen, J. (2024). *Efficient click‑based interactive segmentation for medical image with improved Plain‑ViT*. IEEE JBHI.  
    6. Luo, X., Wang, G., Song, T., Zhang, J., Aertsen, M., Deprest, J., Ourselin, S., Vercauteren, T., & Zhang, S. (2021). *MIDeepSeg: Minimally Interactive Segmentation of Unseen Objects from Medical Images Using Deep Learning*. arXiv:2104.12166.
    """)
