import streamlit as st
from PIL import Image
import numpy as np
import tempfile, subprocess, uuid, os

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


# 3. iSegFormer
elif page == "iSegFormer":
    st.title("iSegFormer (Liu et al., MICCAI 2022)")
    st.markdown("""
    iSegFormer is an **interactive segmentation** framework tailored for **3D knee MRI**, where 
    cartilage surfaces present complex curvatures and often blend into adjacent tissues. In their 
    MICCAI 2022 paper, Liu et al. show that combining a hierarchical **Swin Transformer** encoder 
    with a lightweight **MLP decoder** allows the model to quickly adapt to sparse user annotations 
    (Liu et al., 2022).  

    Rather than processing each slice in isolation, iSegFormer propagates corrections across the 
    entire volume: the user labels a few key slices, and the transformer’s global attention 
    ensures consistency as the model refines segmentation in neighboring slices. This yields Dice 
    scores above 90% with minimal manual effort, demonstrating that interactive refinement can 
    match or exceed fully automated accuracy under limited annotation budgets.  

    The main trade-off is computational: handling 3D volumes with self-attention requires 
    substantial GPU memory and incurs longer model initialization times, which may challenge 
    deployment in resource-constrained clinical settings.
    """, unsafe_allow_html=True)
    load_asset("architecture.jpg", caption="Figure 3: iSegFormer Architecture")

# 4. Interactive Demo
elif page == "Interactive Demo":
    st.title("Interactive Segmentation Demo")

    # 上传图像
    uploaded = st.file_uploader("Upload a medical image", type=["png","jpg","jpeg"])
    if not uploaded:
        st.info("Please upload an image to begin.")
        st.stop()

    # 转为 NumPy
    img = Image.open(uploaded).convert("RGB")
    img_np = np.array(img)

    # 延迟加载模型预测器（假设 infer_simpleclick.build_predictor 存在）
    @st.cache_resource
    def load_predictor(path):
        from infer_simpleclick import build_predictor
        return build_predictor(path)

    predictor = load_predictor("./weights/simpleclick_models/cocolvis_vit_huge.pth")

    # 选择点击类型
    click_type = st.radio("Click type", ["Positive (foreground)", "Negative (background)"])

    # 初始化点击列表
    if "clicks" not in st.session_state:
        st.session_state.clicks = []  # list of (x,y,is_positive)

    # 可绘制画布，仅支持点击(point)
    canvas_result = st_canvas(
        fill_color="rgba(0,0,0,0)",
        stroke_color="#0f0" if click_type.startswith("Positive") else "#f00",
        stroke_width=20,
        background_image=img,
        drawing_mode="point",
        key="canvas",
        update_streamlit=True,
        height=img_np.shape[0],
        width=img_np.shape[1],
    )

    # 记录最新点击
    if canvas_result.json_data and canvas_result.json_data.get("objects"):
        for obj in canvas_result.json_data["objects"][-1:]:
            x, y = obj["path"][-1]
            st.session_state.clicks.append(
                (int(x), int(y), click_type.startswith("Positive"))
            )

    # 展示点击历史 & 运行分割
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("#### Click history")
        for i, (x, y, is_pos) in enumerate(st.session_state.clicks, 1):
            emoji = "🟢" if is_pos else "🔴"
            st.write(f"{i}. {emoji} ({x}, {y})")
        if st.button("🔄 Reset Clicks"):
            st.session_state.clicks = []

    with col2:
        st.markdown("#### Segmentation result")
        if st.button("Run / Update Segmentation"):
            # predictor.get_prediction(image_np, clicks) → 返回二值 mask
            mask = predictor.get_prediction(img_np, st.session_state.clicks)
            # 叠加：红色高亮
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
    3. Liu, Q., Xu, Z., Bertasius, G., & Niethammer, M. (2023). *SimpleClick: Interactive Image Segmentation with Simple Vision Transformers*. ICCVW 2023.  
    4. Marinov, Z., Jäger, P. F., Egger, J., Kleesiek, J., & Stiefelhagen, R. (2024). *Deep interactive segmentation of medical images: A systematic review and taxonomy*. IEEE TPAMI, 46(12), 10998–11039.  
    5. Huang, M., Zou, J., Zhang, Y., Bhatti, U. A., & Chen, J. (2024). *Efficient click‑based interactive segmentation for medical image with improved Plain‑ViT*. IEEE JBHI.  
    6. Luo, X., Wang, G., Song, T., Zhang, J., Aertsen, M., Deprest, J., Ourselin, S., Vercauteren, T., & Zhang, S. (2021). *MIDeepSeg: Minimally Interactive Segmentation of Unseen Objects from Medical Images Using Deep Learning*. arXiv:2104.12166.
    """)
