
import base64
import html
import io
import mimetypes
import os
from pathlib import Path

import requests
import streamlit as st
from PIL import Image

# ============================================================
# Configuration
# ============================================================
API_URL = os.getenv(
    "LUNGSIGHT_API_URL",
    "http://localhost:8000",
).rstrip("/")

REQUEST_TIMEOUT = int(os.getenv("LUNGSIGHT_TIMEOUT", "120"))

LABELS = [
    "Cardiomegaly",
    "Edema",
    "Consolidation",
    "Atelectasis",
    "Pleural Effusion",
]

APP_VERSION = "1.1"

# Research metrics from the dissertation / final dataset slide.
RESEARCH = {
    "original_images": "224,316",
    "patients": "65,240",
    "final_augmented_images": "374,075",
    "dataset_size": "12.6 GB",
    "augmentation_target": "60,000 / label",
    "target_pathologies": "5",
    "input_size": "224 × 224",
    "epochs": "15",
    "batch_size": "32",
    "gpu": "NVIDIA RTX 5090",
    "mean_auroc": "0.9520",
    "mean_accuracy": "89.0%",
    "sota_baseline": "0.940",
    "auc_improvement": "+1.20 pp",
}

BENCHMARKS = [
    ("DenseNet-121", "Single CNN", "0.889"),
    ("EfficientNet-B4", "Single CNN", "0.911"),
    ("ViT-Base/16", "Transformer", "0.921"),
    ("Swin-Tiny", "Transformer", "0.931"),
    ("ConvNeXt-Tiny", "Single CNN", "0.929"),
    ("Ensemble CNN", "Ensemble", "0.940"),
    ("SwinConvNeXt Hybrid (Ours)", "Hybrid / Single model", "0.9520"),
]

# ============================================================
# Page configuration
# ============================================================
st.set_page_config(
    page_title="LungSight AI",
    page_icon="🩻",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ============================================================
# Session state
# ============================================================
DEFAULT_STATE = {
    "image_bytes": None,
    "filename": None,
    "preview_image": None,
    "predictions": None,
    "heatmap": None,
    "explanation": None,
    "gradcam_label": None,
    "gradcam_probability": None,
    "analysis_complete": False,
}

for key, value in DEFAULT_STATE.items():
    if key not in st.session_state:
        st.session_state[key] = value


def reset_results() -> None:
    st.session_state["predictions"] = None
    st.session_state["heatmap"] = None
    st.session_state["explanation"] = None
    st.session_state["gradcam_label"] = None
    st.session_state["gradcam_probability"] = None
    st.session_state["analysis_complete"] = False


def save_uploaded_file(uploaded_file) -> None:
    new_bytes = uploaded_file.getvalue()
    new_filename = uploaded_file.name

    is_new_file = (
        st.session_state["image_bytes"] != new_bytes
        or st.session_state["filename"] != new_filename
    )

    if not is_new_file:
        return

    st.session_state["image_bytes"] = new_bytes
    st.session_state["filename"] = new_filename
    reset_results()

    try:
        st.session_state["preview_image"] = (
            Image.open(io.BytesIO(new_bytes)).convert("RGB")
        )
    except Exception:
        st.session_state["preview_image"] = None


# ============================================================
# API helpers
# ============================================================
def post_image(
    path: str,
    image_bytes: bytes,
    filename: str,
    **params,
) -> requests.Response:
    """Send the image using the correct JPEG/PNG MIME type."""
    mime_type, _ = mimetypes.guess_type(filename)

    if mime_type not in {"image/jpeg", "image/png"}:
        suffix = Path(filename).suffix.lower()

        if suffix in {".jpg", ".jpeg"}:
            mime_type = "image/jpeg"
        elif suffix == ".png":
            mime_type = "image/png"
        else:
            raise ValueError(
                "Only JPEG and PNG images are supported."
            )

    return requests.post(
        f"{API_URL}{path}",
        params=params or None,
        files={
            "file": (
                filename,
                image_bytes,
                mime_type,
            )
        },
        timeout=REQUEST_TIMEOUT,
    )


def backend_error(response: requests.Response | None) -> str:
    if response is None:
        return "Unable to reach the backend."

    try:
        payload = response.json()
        detail = payload.get("detail", payload)

        if isinstance(detail, dict):
            return str(detail.get("message", detail))

        return str(detail)
    except Exception:
        return response.text or f"HTTP {response.status_code}"


def decode_heatmap(value: str) -> Image.Image:
    raw = base64.b64decode(value)
    return Image.open(
        io.BytesIO(raw)
    ).convert("RGB")


def check_backend() -> tuple[bool, dict]:
    try:
        response = requests.get(
            f"{API_URL}/health",
            timeout=5,
        )

        if response.ok:
            try:
                return True, response.json()
            except ValueError:
                return True, {"status": "ok"}

        return False, {}

    except requests.RequestException:
        return False, {}


# ============================================================
# Styling
# ============================================================
st.markdown(
    """
<style>
    .stApp {
        background: #f5f7fa;
    }

    .block-container {
        max-width: 1280px;
        padding-top: 1.7rem;
        padding-bottom: 3rem;
    }

    #MainMenu,
    footer {
        visibility: hidden;
    }

    section[data-testid="stSidebar"] {
        background: #ffffff;
        border-right: 1px solid #e4e7ec;
    }

    .brand {
        font-size: 2.65rem;
        line-height: 1;
        font-weight: 850;
        letter-spacing: -0.045em;
        color: #101828;
        margin-bottom: .35rem;
    }

    .subtitle {
        color: #667085;
        font-size: 1.03rem;
        margin-bottom: 1.3rem;
    }

    .micro-label {
        color: #667085;
        font-size: .76rem;
        font-weight: 700;
        letter-spacing: .08em;
        text-transform: uppercase;
    }

    .section-title {
        font-size: 1.42rem;
        font-weight: 780;
        color: #101828;
        margin: 1.5rem 0 .75rem;
    }

    .section-subtitle {
        color: #667085;
        font-size: .92rem;
        margin-top: -.45rem;
        margin-bottom: 1rem;
    }

    .card {
        background: #ffffff;
        border: 1px solid #e4e7ec;
        border-radius: 18px;
        padding: 1.25rem;
        box-shadow: 0 2px 12px rgba(16, 24, 40, .045);
    }

    .metric-card {
        background: #ffffff;
        border: 1px solid #e4e7ec;
        border-radius: 15px;
        padding: 1.05rem;
        min-height: 115px;
        box-shadow: 0 2px 9px rgba(16, 24, 40, .035);
    }

    .metric-label {
        color: #667085;
        font-size: .78rem;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: .04em;
    }

    .metric-value {
        color: #101828;
        font-size: 1.55rem;
        line-height: 1.1;
        font-weight: 850;
        margin-top: .45rem;
    }

    .metric-note {
        color: #98a2b3;
        font-size: .76rem;
        margin-top: .3rem;
    }

    .hero-metric {
        background: #eff8ff;
        border: 1px solid #b2ddff;
        border-radius: 18px;
        padding: 1.25rem;
    }

    .hero-number {
        color: #175cd3;
        font-size: 2.65rem;
        line-height: 1;
        font-weight: 900;
        margin-top: .35rem;
    }

    .hero-caption {
        color: #475467;
        font-size: .88rem;
        margin-top: .45rem;
    }

    .finding {
        background: #f9fafb;
        border: 1px solid #eaecf0;
        border-left: 4px solid #175cd3;
        border-radius: 10px;
        padding: .9rem 1rem;
        margin-bottom: .65rem;
        color: #344054;
        line-height: 1.55;
    }

    .warning {
        background: #fffaeb;
        border: 1px solid #fedf89;
        color: #7a5a00;
        border-radius: 13px;
        padding: .95rem 1rem;
        font-size: .84rem;
        line-height: 1.55;
    }

    .result-card {
        background: #ffffff;
        border: 1px solid #e4e7ec;
        border-radius: 15px;
        padding: 1rem;
        min-height: 118px;
        box-shadow: 0 2px 9px rgba(16, 24, 40, .035);
    }

    .result-name {
        color: #475467;
        font-size: .85rem;
        font-weight: 680;
        min-height: 2.2rem;
        display: flex;
        align-items: flex-start;
    }

    .result-value {
        color: #101828;
        font-size: 1.7rem;
        line-height: 1;
        font-weight: 850;
        margin-top: .45rem;
    }

    .result-caption {
        color: #98a2b3;
        font-size: .74rem;
        margin-top: .38rem;
    }

    .highlight {
        background: #eff8ff;
        border: 1px solid #b2ddff;
        border-radius: 15px;
        padding: 1rem 1.15rem;
        margin: 1rem 0;
    }

    .highlight-title {
        color: #344054;
        font-size: .83rem;
        font-weight: 700;
    }

    .highlight-value {
        color: #175cd3;
        font-weight: 850;
        font-size: 1.35rem;
        margin-top: .2rem;
    }

    .explanation-card {
        background: #ffffff;
        border: 1px solid #e4e7ec;
        border-radius: 15px;
        padding: 1.2rem 1.3rem;
        color: #344054;
        line-height: 1.75;
        box-shadow: 0 2px 9px rgba(16, 24, 40, .035);
    }

    .backend-status {
        border-radius: 12px;
        padding: .8rem .9rem;
        border: 1px solid #e4e7ec;
        background: #f9fafb;
        margin-top: .8rem;
    }

    .stButton > button {
        min-height: 2.75rem;
        border-radius: 10px;
        font-weight: 720;
    }

    @media (max-width: 768px) {
        .brand {
            font-size: 2.1rem;
        }

        .block-container {
            padding-left: 1rem;
            padding-right: 1rem;
        }
    }
</style>
""",
    unsafe_allow_html=True,
)

# ============================================================
# Sidebar
# ============================================================
backend_online, health_data = check_backend()

with st.sidebar:
    st.markdown(
        """
        <div class="micro-label">LungSight</div>
        <h2 style="margin:.15rem 0 0;color:#101828;">
            AI Chest X-ray
        </h2>
        """,
        unsafe_allow_html=True,
    )

    st.caption("Research & explainability interface")

    page = st.radio(
        "Navigation",
        [
            "Analyze X-ray",
            "Research & Model",
            "About",
        ],
        label_visibility="collapsed",
    )

    st.divider()
    st.markdown("**Backend status**")

    if backend_online:
        st.success("API online", icon="✅")
    else:
        st.error("API unavailable", icon="❌")

    st.markdown(
        f"""
        <div class="backend-status">
            <div style="font-size:.76rem;color:#667085;">
                API endpoint
            </div>
            <div style="font-size:.78rem;color:#344054;
                        word-break:break-all;margin-top:.25rem;">
                {html.escape(API_URL)}
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.divider()
    st.caption(f"LungSight UI v{APP_VERSION}")

# ============================================================
# Global header
# ============================================================
st.markdown(
    '<div class="brand">LungSight AI</div>',
    unsafe_allow_html=True,
)

st.markdown(
    '<div class="subtitle">'
    "Explainable AI-assisted chest X-ray analysis"
    "</div>",
    unsafe_allow_html=True,
)

# ============================================================
# Analyze page
# ============================================================
if page == "Analyze X-ray":

    st.markdown(
        '<div class="section-title">Analyze a chest X-ray</div>',
        unsafe_allow_html=True,
    )

    st.markdown(
        """
        <div class="section-subtitle">
            Upload an X-ray, review model scores, and inspect
            class-specific visual explanations.
        </div>
        """,
        unsafe_allow_html=True,
    )

    upload_col, info_col = st.columns(
        [1.25, .75],
        gap="large",
    )

    with upload_col:
        st.markdown('<div class="card">', unsafe_allow_html=True)

        uploaded = st.file_uploader(
            "Upload X-ray image",
            type=["png", "jpg", "jpeg"],
            help="JPEG and PNG images are supported.",
        )

        if uploaded is not None:
            save_uploaded_file(uploaded)

        if st.session_state["preview_image"] is not None:
            st.image(
                st.session_state["preview_image"],
                caption=st.session_state["filename"],
                width="stretch",
            )

        if st.button(
            "Analyze X-ray",
            type="primary",
            width="stretch",
            disabled=st.session_state["image_bytes"] is None,
        ):
            with st.spinner("Running chest X-ray inference..."):
                try:
                    response = post_image(
                        "/predict",
                        st.session_state["image_bytes"],
                        st.session_state["filename"],
                    )
                    response.raise_for_status()

                    payload = response.json()

                    st.session_state["predictions"] = payload.get(
                        "predictions",
                        {},
                    )
                    st.session_state["heatmap"] = None
                    st.session_state["explanation"] = None
                    st.session_state["analysis_complete"] = True

                except requests.RequestException as exc:
                    st.error(
                        "Prediction failed: "
                        + backend_error(exc.response)
                    )
                except (ValueError, TypeError) as exc:
                    st.error(
                        f"Invalid prediction response: {exc}"
                    )

        st.markdown('</div>', unsafe_allow_html=True)

    with info_col:
        st.subheader("Prediction + explainability")
        st.caption("What you get")

        st.markdown(
            """
            - ✓ Five pathology scores
            - ✓ Class-specific Grad-CAM
            - ✓ Optional Qwen explanation
            - ✓ Separate inference backend
            """
        )

        st.warning(
            "Research prototype. LungSight is intended for educational "
            "and research use. It is not a medical device and does not "
            "provide a clinical diagnosis."
        )

    predictions = st.session_state["predictions"]

    if predictions:
        st.markdown(
            '<div class="section-title">Prediction results</div>',
            unsafe_allow_html=True,
        )

        result_cols = st.columns(
            len(LABELS),
            gap="small",
        )

        for col, label in zip(result_cols, LABELS):
            value = float(
                predictions.get(label, 0.0)
            )

            with col:
                st.markdown(
                    f"""
                    <div class="result-card">
                        <div class="result-name">
                            {html.escape(label)}
                        </div>
                        <div class="result-value">
                            {value * 100:.2f}%
                        </div>
                        <div class="result-caption">
                            model score
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

        top_label = max(
            predictions,
            key=predictions.get,
        )
        top_score = float(
            predictions[top_label]
        )

        st.markdown(
            f"""
            <div class="highlight">
                <div class="highlight-title">
                    Highest model score
                </div>
                <div class="highlight-value">
                    {html.escape(top_label)}
                    &nbsp;·&nbsp;
                    {top_score * 100:.2f}%
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.markdown(
            '<div class="section-title">Explainability</div>',
            unsafe_allow_html=True,
        )

        selected = st.selectbox(
            "Pathology for Grad-CAM",
            LABELS,
            index=LABELS.index(top_label),
        )

        explain = st.checkbox(
            "Generate AI explanation",
            value=True,
        )

        if st.button(
            "Generate Grad-CAM",
            width="stretch",
        ):
            with st.spinner("Generating Grad-CAM..."):
                try:
                    response = post_image(
                        "/gradcam",
                        st.session_state["image_bytes"],
                        st.session_state["filename"],
                        disease=selected,
                        explain=explain,
                    )
                    response.raise_for_status()

                    if explain:
                        payload = response.json()

                        st.session_state["heatmap"] = (
                            decode_heatmap(
                                payload[
                                    "heatmap_image_base64"
                                ]
                            )
                        )

                        st.session_state["explanation"] = (
                            payload.get(
                                "explanation",
                                "",
                            )
                        )

                        st.session_state["gradcam_label"] = (
                            payload.get(
                                "label",
                                selected,
                            )
                        )

                        st.session_state[
                            "gradcam_probability"
                        ] = float(
                            payload.get(
                                "probability",
                                predictions.get(
                                    selected,
                                    0,
                                ),
                            )
                        )

                        explanation_error = payload.get(
                            "explanation_error"
                        )

                        if explanation_error:
                            st.session_state[
                                "explanation"
                            ] = ""
                            st.warning(
                                "AI explanation unavailable: "
                                f"{explanation_error}"
                            )

                    else:
                        st.session_state["heatmap"] = (
                            Image.open(
                                io.BytesIO(
                                    response.content
                                )
                            ).convert("RGB")
                        )

                        st.session_state[
                            "explanation"
                        ] = ""

                        st.session_state[
                            "gradcam_label"
                        ] = selected

                        st.session_state[
                            "gradcam_probability"
                        ] = float(
                            predictions.get(
                                selected,
                                0,
                            )
                        )

                except requests.RequestException as exc:
                    st.error(
                        "Grad-CAM failed: "
                        + backend_error(exc.response)
                    )
                except (
                    KeyError,
                    ValueError,
                    base64.binascii.Error,
                ) as exc:
                    st.error(
                        f"Could not read the Grad-CAM response: {exc}"
                    )

        if st.session_state["heatmap"] is not None:
            st.markdown(
                '<div class="section-title">Visual explanation</div>',
                unsafe_allow_html=True,
            )

            image_col, result_col = st.columns(
                [1.15, .85],
                gap="large",
            )

            with image_col:
                st.markdown(
                    "**Grad-CAM heatmap**"
                )
                st.image(
                    st.session_state["heatmap"],
                    caption=(
                        "Highlighted regions indicate areas that "
                        "influenced the model prediction."
                    ),
                    width="stretch",
                )

            with result_col:
                label = (
                    st.session_state["gradcam_label"]
                    or selected
                )
                probability = float(
                    st.session_state[
                        "gradcam_probability"
                    ] or 0
                )

                st.markdown("### Selected pathology")
                st.markdown(f"#### {label}")
                st.metric(
                    "Model score",
                    f"{probability * 100:.2f}%",
                )

                if st.session_state["explanation"]:
                    st.markdown(
                        '<div class="section-subtitle" '
                        'style="margin-top:1.1rem;">'
                        "AI-generated explanation"
                        "</div>",
                        unsafe_allow_html=True,
                    )

                    explanation = html.escape(
                        st.session_state["explanation"]
                    )

                    st.markdown(
                        f"""
                        <div class="explanation-card">
                            {explanation}
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )

            st.markdown(
                """
                <div class="warning" style="margin-top:1rem;">
                    <strong>Interpretation note:</strong>
                    Grad-CAM shows regions that influenced the model;
                    it does not establish that a disease is present.
                    AI-generated explanations are not medical diagnoses.
                </div>
                """,
                unsafe_allow_html=True,
            )

# ============================================================
# Research & Model
# ============================================================
elif page == "Research & Model":

    st.markdown(
        '<div class="section-title">Research & model</div>',
        unsafe_allow_html=True,
    )

    st.markdown(
        """
        <div class="section-subtitle">
            LungSight extends the undergraduate dissertation research
            into a deployable, explainable AI system.
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown("### Dissertation")

    st.info(
        "Advancing Chest Radiograph Interpretation: A Hybrid CNN–Transformer "
        "Architecture with Label-Aware Augmentation on CheXpert"
    )

    st.write(
        "The research investigates a hybrid CNN–Transformer approach for "
        "multi-label chest X-ray classification on CheXpert, combining "
        "ConvNeXt-Tiny and Swin-Tiny with label-aware augmentation."
    )

    st.markdown("### Performance at a glance")

    c1, c2, c3 = st.columns(3)

    with c1:
        st.metric(
            "Mean AUROC",
            RESEARCH["mean_auroc"],
            help="Five-condition validation result.",
        )

    with c2:
        st.metric(
            "Mean Accuracy",
            RESEARCH["mean_accuracy"],
            help="Binary classification accuracy.",
        )

    with c3:
        st.metric(
            "Improvement",
            RESEARCH["auc_improvement"],
            help="Absolute improvement over the reported 0.940 ensemble baseline.",
        )

    st.markdown("### Dataset & training")

    metrics = [
        ("Original radiographs", RESEARCH["original_images"], "CheXpert"),
        ("Patients", RESEARCH["patients"], "CheXpert cohort"),
        (
            "Final augmented dataset",
            RESEARCH["final_augmented_images"],
            "Final experimental dataset",
        ),
        ("Dataset size", RESEARCH["dataset_size"], "Reported storage size"),
        (
            "Augmentation target",
            RESEARCH["augmentation_target"],
            "Per target label",
        ),
        ("Target pathologies", RESEARCH["target_pathologies"], "Multi-label task"),
        ("Input resolution", RESEARCH["input_size"], "Model input"),
        ("Training epochs", RESEARCH["epochs"], "Best checkpoint"),
        ("Batch size", RESEARCH["batch_size"], "Training"),
        ("Training GPU", RESEARCH["gpu"], "CUDA training"),
    ]

    metric_cols = st.columns(5)

    for index, (label, value, note) in enumerate(metrics):
        with metric_cols[index % 5]:
            st.metric(
                label,
                value,
                help=note,
            )

    st.markdown("### Model architecture")

    arch_left, arch_right = st.columns(2)

    with arch_left:
        st.subheader("SwinConvNeXt Hybrid")
        st.code(
            """Chest X-ray
     ↓
 ┌───────────────┐
 │               │
 ▼               ▼
ConvNeXt-Tiny   Swin-Tiny
(Local)        (Global)
 │               │
 ▼               ▼
Adaptive Avg Pooling
       ↓
Feature Fusion
       ↓
Linear Head
       ↓
5-label output""",
            language="text",
        )

    with arch_right:
        st.subheader("Why hybrid?")
        st.write(
            "**CNN branch:** captures fine-grained local features such as "
            "textures, edges, and localized patterns."
        )
        st.write(
            "**Transformer branch:** captures broader spatial relationships "
            "and long-range context."
        )
        st.write(
            "**Fusion:** combines complementary representations in a single "
            "model rather than using a multi-model ensemble."
        )

    st.markdown("### Benchmark comparison")
    st.caption(
        "Mean AUROC values reported in the dissertation."
    )

    for method, architecture, score in BENCHMARKS:
        cols = st.columns([2.4, 1.8, 0.8])

        with cols[0]:
            st.write(f"**{method}**")

        with cols[1]:
            st.write(architecture)

        with cols[2]:
            st.write(f"**{score}**")

    st.success(
        f"Research result: the proposed single SwinConvNeXt model reached "
        f"{RESEARCH['mean_auroc']} mean AUROC versus the reported "
        f"{RESEARCH['sota_baseline']} ensemble baseline — an absolute "
        f"improvement of {RESEARCH['auc_improvement']}."
    )

    st.markdown("### What was actually learned")

    findings = [
        (
            "Hybrid architecture",
            "Combining convolutional and Transformer representations "
            "improved discriminative performance over the individual "
            "Swin-Tiny and ConvNeXt-Tiny baselines reported in the study.",
        ),
        (
            "Single model vs ensemble",
            "The proposed hybrid achieved a higher mean AUROC than the "
            "reported ensemble baseline while remaining a single model.",
        ),
        (
            "Data augmentation mattered most",
            "Removing the offline label-aware augmentation pipeline caused "
            "the largest performance drop among the tested configurations.",
        ),
        (
            "Fine-tuning was important",
            "Pretrained ImageNet features provided a strong starting point, "
            "while end-to-end fine-tuning adapted the model to the medical "
            "imaging domain.",
        ),
        (
            "Convergence",
            "Validation AUROC progressed from 0.7904 at epoch 1 to 0.9520 "
            "at epoch 15, with the best checkpoint selected by validation AUC.",
        ),
    ]

    for title, text in findings:
        st.info(f"**{title}:** {text}")

    st.markdown("### Training trajectory")

    trajectory = [
        ("Epoch 1", "Val AUC 0.7904", "74.2% accuracy"),
        ("Epoch 5", "Val AUC 0.8820", "83.1% accuracy"),
        ("Epoch 10", "Val AUC 0.9310", "87.8% accuracy"),
        ("Epoch 15", "Val AUC 0.9520", "89.0% accuracy"),
    ]

    t1, t2, t3 = st.columns([1, 2, 2])
    t1.markdown("**Epoch**")
    t2.markdown("**Validation AUC**")
    t3.markdown("**Validation accuracy**")

    for epoch, auc, accuracy in trajectory:
        c1, c2, c3 = st.columns([1, 2, 2])
        c1.write(epoch)
        c2.write(auc)
        c3.write(accuracy)

    st.warning(
        "Research caveats: the reported 89% accuracy uses a fixed 0.5 "
        "threshold. The study used a validation split rather than a separate "
        "held-out test set, so real-world generalisation may differ. The "
        "individual per-pathology AUC values in the dissertation are described "
        "as estimates; this page therefore emphasizes the reported mean AUROC "
        "of 0.9520."
    )

# ============================================================
# About
# ============================================================
else:

    st.markdown(
        '<div class="section-title">About LungSight</div>',
        unsafe_allow_html=True,
    )

    st.markdown(
        "### From research to application"
    )

    st.info(
        "LungSight extends the undergraduate dissertation research "
        "into a working application."
    )

    st.write(
        "The research model handles multi-label chest X-ray "
        "classification, while the deployed system adds Grad-CAM "
        "explainability, a vision-language explanation layer, "
        "API-based inference, containerization, cloud deployment, "
        "and a user-facing frontend."
    )

    st.code(
        """Research
   ↓
Model
   ↓
Explainability
   ↓
API
   ↓
Docker
   ↓
Cloud
   ↓
Application""",
        language="text",
    )

    st.markdown("### System stack")

    stack = [
        ("Machine Learning", "PyTorch, ConvNeXt, Swin Transformer, ONNX Runtime"),
        ("Explainability", "Grad-CAM"),
        ("LLM", "Qwen via OpenAI-compatible API"),
        ("Backend", "FastAPI + Uvicorn"),
        ("Frontend", "Streamlit"),
        ("Deployment", "Docker + AWS EC2"),
        ("Artifacts", "Hugging Face"),
    ]

    for category, technologies in stack:
        st.write(f"**{category}:** {technologies}")

    st.warning(
        "Medical disclaimer: LungSight is a research and educational "
        "prototype. It is not a certified medical device and is not "
        "intended to diagnose, treat, or provide medical advice. Real "
        "chest X-rays should be interpreted by a qualified radiologist "
        "or physician."
    )

