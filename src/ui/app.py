"""Streamlit web interface for AI Product Photo Detector."""

import base64
import io
import os
from datetime import datetime

import httpx
import streamlit as st
from PIL import Image

# Configuration
API_URL = os.getenv("API_URL", "http://localhost:8080")
API_KEY = os.getenv("API_KEY", "")
MAX_DISPLAY_SIZE = 800
MAX_UPLOAD_MB = 20  # Accept larger files, we'll compress


def init_page() -> None:
    """Initialize Streamlit page configuration."""
    st.set_page_config(
        page_title="AI Image Detector",
        page_icon="mag",
        layout="centered",
        initial_sidebar_state="collapsed",
    )

    st.markdown("""
    <style>
        .main > div { max-width: 700px; margin: 0 auto; }
        .stFileUploader > div { border-radius: 12px; }
        div[data-testid="stImage"] { border-radius: 12px; overflow: hidden; }
        .result-card {
            padding: 24px; border-radius: 16px; text-align: center;
            margin: 16px 0; box-shadow: 0 2px 12px rgba(0,0,0,0.1);
        }
        .result-real {
            background: linear-gradient(135deg, #e8f5e9, #c8e6c9);
            border: 2px solid #4caf50;
        }
        .result-ai {
            background: linear-gradient(135deg, #fce4ec, #f8bbd0);
            border: 2px solid #e53935;
        }
        .big-percent { font-size: 48px; font-weight: 800; margin: 8px 0; }
        .result-label { font-size: 20px; font-weight: 600; margin-bottom: 4px; }
        .result-meta { color: #666; font-size: 14px; }
        .confidence-bar {
            height: 8px; border-radius: 4px; margin: 12px auto;
            max-width: 300px; overflow: hidden;
        }
        .heatmap-section { margin-top: 20px; }
    </style>
    """, unsafe_allow_html=True)


# Session state
if "history" not in st.session_state:
    st.session_state.history = []
if "last_result" not in st.session_state:
    st.session_state.last_result = None


def compress_image(image_bytes: bytes, max_size_mb: float = 4.5) -> bytes:
    """Compress image to fit within size limit."""
    if len(image_bytes) <= max_size_mb * 1024 * 1024:
        return image_bytes

    img = Image.open(io.BytesIO(image_bytes))
    if img.mode == "RGBA":
        img = img.convert("RGB")

    # Resize if very large
    max_dim = 2048
    if max(img.size) > max_dim:
        ratio = max_dim / max(img.size)
        new_size = (int(img.size[0] * ratio), int(img.size[1] * ratio))
        img = img.resize(new_size, Image.LANCZOS)

    # Compress with decreasing quality
    for quality in [90, 80, 70, 60]:
        buf = io.BytesIO()
        img.save(buf, format="JPEG", quality=quality, optimize=True)
        if buf.tell() <= max_size_mb * 1024 * 1024:
            return buf.getvalue()

    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=50, optimize=True)
    return buf.getvalue()


def check_api_health() -> dict | None:
    """Check if API is healthy."""
    try:
        response = httpx.get(f"{API_URL}/health", timeout=5.0)
        if response.status_code == 200:
            return response.json()
    except Exception:
        pass
    return None


def predict_image(image_bytes: bytes, filename: str) -> dict | None:
    """Send image to API for prediction."""
    try:
        compressed = compress_image(image_bytes)
        files = {"file": (filename, compressed, "image/jpeg")}
        headers = {"X-API-Key": API_KEY} if API_KEY else {}
        response = httpx.post(
            f"{API_URL}/predict", files=files, headers=headers, timeout=30.0,
        )
        if response.status_code == 200:
            return response.json()
        else:
            try:
                detail = response.json().get("detail", response.text)
            except Exception:
                detail = response.text
            st.error(f"Error: {detail}")
    except Exception as e:
        st.error(f"Connection error: {e}")
    return None


def explain_image(image_bytes: bytes, filename: str) -> dict | None:
    """Get Grad-CAM explanation from API."""
    try:
        compressed = compress_image(image_bytes)
        files = {"file": (filename, compressed, "image/jpeg")}
        headers = {"X-API-Key": API_KEY} if API_KEY else {}
        response = httpx.post(
            f"{API_URL}/predict/explain", files=files, headers=headers, timeout=30.0,
        )
        if response.status_code == 200:
            return response.json()
    except Exception:
        pass
    return None


def display_result(result: dict) -> None:
    """Display prediction result with clear percentage."""
    prediction = result.get("prediction", "unknown")
    probability = result.get("probability", 0)
    inference_time = result.get("inference_time_ms", 0)

    is_ai = prediction == "ai_generated"

    if is_ai:
        percent = probability * 100
        label = "AI-Generated"
        css_class = "result-ai"
        color = "#e53935"
        bar_color = "#e53935"
        icon = "AI"
    else:
        percent = (1 - probability) * 100
        label = "Real Photo"
        css_class = "result-real"
        color = "#4caf50"
        bar_color = "#4caf50"
        icon = "REAL"

    # Confidence text
    if percent >= 95:
        conf_text = "Very High Confidence"
    elif percent >= 80:
        conf_text = "High Confidence"
    elif percent >= 60:
        conf_text = "Medium Confidence"
    else:
        conf_text = "Low Confidence"

    st.markdown(f"""
    <div class="result-card {css_class}">
        <div class="result-label">{icon} {label}</div>
        <div class="big-percent" style="color: {color}">{percent:.1f}%</div>
        <div class="confidence-bar" style="background: #ddd;">
            <div style="width: {percent}%; height: 100%; background: {bar_color}; border-radius: 4px;"></div>
        </div>
        <div class="result-meta">{conf_text} · {inference_time:.0f}ms inference</div>
    </div>
    """, unsafe_allow_html=True)


def main() -> None:
    """Main Streamlit application."""
    init_page()

    # Header
    st.markdown("# AI Image Detector")
    st.markdown("Upload an image to detect if it's **AI-generated** or a **real photograph**.")

    # Status indicator
    health = check_api_health()
    if health and health.get("status") == "healthy":
        st.caption(f"<span style='color:#4caf50;'>●</span> API Online · Model v{health.get('model_version', '?')}", unsafe_allow_html=True)
    else:
        st.caption("<span style='color:#e53935;'>●</span> API Offline", unsafe_allow_html=True)

    st.markdown("---")

    # File upload
    uploaded_file = st.file_uploader(
        "Choose an image",
        type=["jpg", "jpeg", "png", "webp"],
        help="Supports JPEG, PNG, WebP up to 20MB (auto-compressed)",
    )

    if uploaded_file is not None:
        image_bytes = uploaded_file.getvalue()
        file_size_mb = len(image_bytes) / (1024 * 1024)

        if file_size_mb > MAX_UPLOAD_MB:
            st.error(f"File too large ({file_size_mb:.1f}MB). Max: {MAX_UPLOAD_MB}MB")
            return

        # Show image
        image = Image.open(io.BytesIO(image_bytes))
        st.image(image, use_container_width=True)
        st.caption(f"{image.size[0]}×{image.size[1]} · {file_size_mb:.1f}MB")

        # Analyze button
        col1, col2 = st.columns(2)
        with col1:
            analyze = st.button("Analyze", type="primary", use_container_width=True)
        with col2:
            explain = st.button("Explain (Grad-CAM)", use_container_width=True)

        if analyze:
            with st.spinner("Analyzing..."):
                result = predict_image(image_bytes, uploaded_file.name)
                if result:
                    st.session_state.last_result = result
                    st.session_state.last_explain = None
                    st.session_state.history.append({
                        "time": datetime.now().strftime("%H:%M"),
                        "file": uploaded_file.name[:25],
                        "result": result.get("prediction", "?"),
                        "score": f"{(1 - result['probability']) * 100 if result['prediction'] == 'real' else result['probability'] * 100:.1f}%",
                    })

        if explain:
            with st.spinner("Generating heatmap..."):
                result = explain_image(image_bytes, uploaded_file.name)
                if result:
                    st.session_state.last_result = result
                    st.session_state.last_explain = result.get("heatmap_base64")

        # Show result
        if st.session_state.last_result:
            display_result(st.session_state.last_result)

            # Show heatmap if available
            heatmap_b64 = getattr(st.session_state, "last_explain", None)
            if heatmap_b64:
                st.markdown("#### Grad-CAM Heatmap")
                st.markdown("*Highlighted areas show what the model focused on*")
                heatmap_bytes = base64.b64decode(heatmap_b64)
                st.image(heatmap_bytes, use_container_width=True)

    # History
    if st.session_state.history:
        st.markdown("---")
        with st.expander(f"History ({len(st.session_state.history)} predictions)"):
            for entry in reversed(st.session_state.history[-10:]):
                icon = "REAL" if entry["result"] == "real" else "AI"
                st.markdown(f"`{entry['time']}` {icon} **{entry['file']}** → {entry['score']}")

    # Footer
    st.markdown("---")
    st.markdown(
        "<div style='text-align:center; color:#888; font-size:13px;'>"
        "Built by <a href='https://github.com/nolancacheux' style='color:#888;'>Nolan Cacheux</a> · "
        "EfficientNet-B0 · PyTorch · FastAPI · GCP Cloud Run"
        "</div>",
        unsafe_allow_html=True,
    )


if __name__ == "__main__":
    main()
