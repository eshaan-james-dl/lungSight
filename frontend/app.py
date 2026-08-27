import base64
import io
import os
import requests
import streamlit as st
from PIL import Image

API_URL = os.getenv("LUNGSIGHT_API_URL", "http://localhost:8000").rstrip("/")
TIMEOUT = int(os.getenv("LUNGSIGHT_TIMEOUT", "120"))

LABELS = ["Cardiomegaly", "Edema", "Consolidation", "Atelectasis", "Pleural Effusion"]

# Persist uploaded image and analysis results across Streamlit reruns.
for key, default in {
    "image_bytes": None,
    "filename": None,
    "preview_image": None,
    "predictions": None,
    "heatmap": None,
    "explanation": None,
    "gradcam_label": None,
    "gradcam_probability": None,
}.items():
    if key not in st.session_state:
        st.session_state[key] = default

st.set_page_config(page_title="LungSight AI", page_icon="🩻", layout="wide")

st.markdown("""
<style>
.stApp { background:#f6f8fb; }
.block-container { max-width:1200px; padding-top:2rem; }
#MainMenu, footer { visibility:hidden; }
.brand {font-size:2.4rem;font-weight:800;color:#172033;letter-spacing:-.04em;}
.subtitle {color:#667085;font-size:1.05rem;}
.section {font-size:1.35rem;font-weight:750;color:#172033;margin:1.4rem 0 .8rem;}
.card {background:white;border:1px solid #e4e7ec;border-radius:16px;padding:1.2rem;box-shadow:0 2px 10px rgba(16,24,40,.04);}
.pred {background:white;border:1px solid #e4e7ec;border-radius:14px;padding:1rem;min-height:105px;}
.pred-name {color:#475467;font-size:.85rem;font-weight:650;}
.pred-value {color:#172033;font-size:1.6rem;font-weight:800;margin-top:.4rem;}
.banner {background:#eef4ff;border:1px solid #c7d7fe;border-radius:14px;padding:1rem;margin:1rem 0;}
.explain {background:white;border:1px solid #e4e7ec;border-radius:14px;padding:1.2rem;line-height:1.7;color:#344054;}
.warning {background:#fffaeb;border:1px solid #fedf89;border-radius:12px;padding:.9rem 1rem;color:#7a5a00;font-size:.84rem;line-height:1.5;}
</style>
""", unsafe_allow_html=True)

def post(path, image_bytes, filename, **params):
    return requests.post(
        f"{API_URL}{path}",
        params=params or None,
        files={"file": (filename, image_bytes, "image/png")},
        timeout=TIMEOUT,
    )

def get_error(r):
    try:
        return str(r.json().get("detail", r.json()))
    except Exception:
        return r.text or f"HTTP {r.status_code}"

def as_image(b64):
    return Image.open(io.BytesIO(base64.b64decode(b64))).convert("RGB")

with st.sidebar:
    st.markdown("## 🩻 LungSight")
    st.caption("AI-assisted chest X-ray analysis")
    page = st.radio("Navigation", ["Analyze X-ray", "Model Information", "About"])
    st.divider()
    st.caption(f"Backend: `{API_URL}`")
    try:
        health = requests.get(f"{API_URL}/health", timeout=5)
        st.success("API online" if health.ok else "API error")
    except requests.RequestException:
        st.error("API unavailable")

st.markdown('<div class="brand">LungSight AI</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">Explainable AI-assisted analysis of chest X-rays</div>', unsafe_allow_html=True)

if page == "Analyze X-ray":
    st.markdown('<div class="section">Analyze a chest X-ray</div>', unsafe_allow_html=True)

    left, right = st.columns([1.15, .85], gap="large")
    with left:
        uploaded = st.file_uploader("Upload an X-ray image", type=["png","jpg","jpeg","webp"])

        if uploaded:
            new_bytes = uploaded.getvalue()
            new_filename = uploaded.name

            if (st.session_state["image_bytes"] != new_bytes or
                    st.session_state["filename"] != new_filename):
                st.session_state["image_bytes"] = new_bytes
                st.session_state["filename"] = new_filename
                st.session_state["preview_image"] = Image.open(
                    io.BytesIO(new_bytes)).convert("RGB")
                st.session_state["predictions"] = None
                st.session_state["heatmap"] = None
                st.session_state["explanation"] = None
                st.session_state["gradcam_label"] = None
                st.session_state["gradcam_probability"] = None

        if st.session_state["preview_image"] is not None:
            st.image(st.session_state["preview_image"],
                     caption="Uploaded X-ray", width="stretch")

        analyze = st.button("Analyze X-ray", type="primary",
                            width="stretch",
                            disabled=st.session_state["image_bytes"] is None)

    with right:
        st.markdown("""
        <div class="card">
        <h3 style="margin-top:0;color:#172033;">LungSight provides</h3>
        <ul style="color:#475467;line-height:1.9;">
        <li>Five-pathology classification</li>
        <li>Model probability scores</li>
        <li>Class-specific Grad-CAM</li>
        <li>Optional AI-generated explanation</li>
        </ul></div>
        """, unsafe_allow_html=True)
        st.markdown("""
        <div class="warning" style="margin-top:1rem;">
        <strong>Research prototype:</strong> This system is for educational and
        research purposes only. It is not a medical device or clinical diagnosis.
        </div>
        """, unsafe_allow_html=True)

    if analyze:
        with st.spinner("Running model inference..."):
            try:
                r = post("/predict", st.session_state["image_bytes"],
                         st.session_state["filename"])
                r.raise_for_status()
                predictions = r.json().get("predictions", {})
                st.session_state["predictions"] = predictions
                st.session_state["heatmap"] = None
                st.session_state["explanation"] = None
                st.session_state["gradcam_label"] = None
                st.session_state["gradcam_probability"] = None
            except requests.RequestException as e:
                st.error(f"Prediction failed: {get_error(e.response) if e.response else e}")

    if st.session_state["predictions"] is not None:
        predictions = st.session_state["predictions"]
        st.markdown('<div class="section">Prediction results</div>', unsafe_allow_html=True)

        cols = st.columns(5)
        for col, label in zip(cols, LABELS):
            value = float(predictions.get(label, 0))
            with col:
                st.markdown(
                    f'<div class="pred"><div class="pred-name">{label}</div>'
                    f'<div class="pred-value">{value*100:.1f}%</div></div>',
                    unsafe_allow_html=True,
                )

        top = max(predictions, key=predictions.get)
        st.markdown(
            f'<div class="banner"><b>Highest model score</b><br>'
            f'<span style="font-size:1.3rem;font-weight:800;color:#175cd3;">'
            f'{top} — {float(predictions[top])*100:.1f}%</span></div>',
            unsafe_allow_html=True,
        )

        st.markdown('<div class="section">Explainability</div>', unsafe_allow_html=True)
        selected = st.selectbox("Pathology for Grad-CAM", LABELS, index=LABELS.index(top))
        explain = st.checkbox("Generate AI explanation", value=True)

        if st.button("Generate Grad-CAM", width="stretch"):
            with st.spinner("Generating Grad-CAM..."):
                try:
                    r = post("/gradcam", st.session_state["image_bytes"],
                             st.session_state["filename"],
                             disease=selected, explain=explain)
                    r.raise_for_status()
                    if explain:
                        data = r.json()
                        st.session_state["heatmap"] = as_image(data["heatmap_image_base64"])
                        st.session_state["explanation"] = data.get("explanation", "")
                        st.session_state["gradcam_label"] = data.get("label", selected)
                        st.session_state["gradcam_probability"] = data.get(
                            "probability", predictions.get(selected, 0))
                    else:
                        st.session_state["heatmap"] = Image.open(
                            io.BytesIO(r.content)).convert("RGB")
                        st.session_state["explanation"] = ""
                        st.session_state["gradcam_label"] = selected
                        st.session_state["gradcam_probability"] = predictions.get(selected, 0)
                except requests.RequestException as e:
                    st.error(f"Grad-CAM failed: {get_error(e.response) if e.response else e}")
                except (KeyError, ValueError) as e:
                    st.error(f"Could not read backend response: {e}")

        if st.session_state["heatmap"] is not None:
            img_col, text_col = st.columns([1.2, .8], gap="large")
            with img_col:
                st.markdown("**Grad-CAM visualization**")
                st.image(st.session_state["heatmap"],
                         caption="Regions that influenced the model prediction",
                         width="stretch")
            with text_col:
                label = st.session_state.get("gradcam_label", selected)
                prob = float(st.session_state.get("gradcam_probability", 0))
                st.markdown(
                    f'<div class="card"><b>Selected pathology</b><br>'
                    f'<span style="font-size:1.3rem;font-weight:800;color:#175cd3;">{label}</span>'
                    f'<br><br><b>Model probability:</b> {prob*100:.1f}%</div>',
                    unsafe_allow_html=True)
                if st.session_state.get("explanation"):
                    st.markdown("**AI explanation**")
                    st.markdown(
                        f'<div class="explain">{st.session_state["explanation"]}</div>',
                        unsafe_allow_html=True)

            st.markdown("""
            <div class="warning" style="margin-top:1rem;">
            <strong>Interpretation note:</strong> Grad-CAM highlights regions that
            influenced the model. It does not confirm disease. AI explanations are
            not medical diagnoses.
            </div>
            """, unsafe_allow_html=True)

elif page == "Model Information":
    st.markdown('<div class="section">Model information</div>', unsafe_allow_html=True)
    a,b,c = st.columns(3)
    for col,title,value,detail in [
        (a,"Task","Multi-label classification","Chest X-ray pathology detection"),
        (b,"Input","224 × 224","RGB image"),
        (c,"Explainability","Grad-CAM","Class-specific visualization")]:
        with col:
            st.markdown(
                f'<div class="card"><b>{title}</b><h2 style="color:#172033;">{value}</h2>'
                f'<span style="color:#667085;">{detail}</span></div>',
                unsafe_allow_html=True)
    st.markdown('<div class="section">Output classes</div>', unsafe_allow_html=True)
    for label in LABELS:
        st.markdown(f'<div class="card" style="margin-bottom:.6rem;">{label}</div>',
                    unsafe_allow_html=True)

else:
    st.markdown('<div class="section">About LungSight</div>', unsafe_allow_html=True)
    st.markdown("""
    <div class="card">
    <h2 style="color:#172033;margin-top:0;">LungSight AI</h2>
    <p style="color:#475467;line-height:1.7;">
    LungSight is an AI-assisted chest X-ray research application. The frontend is
    separated from the inference backend: Streamlit handles the interface while
    FastAPI provides the machine-learning services through HTTP.
    </p>
    <p style="color:#475467;line-height:1.7;">
    <b>Architecture:</b> Streamlit → FastAPI → ONNX inference / PyTorch Grad-CAM
    → optional vision-language explanation.
    </p>
    </div>
    """, unsafe_allow_html=True)
    st.markdown("""
    <div class="warning" style="margin-top:1rem;">
    <strong>Medical disclaimer:</strong> LungSight is a research/educational prototype.
    It is not a certified medical device and must not be used for diagnosis or treatment.
    Consult a qualified medical professional for real X-ray interpretation.
    </div>
    """, unsafe_allow_html=True)
