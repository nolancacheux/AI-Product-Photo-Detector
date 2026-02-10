"""Streamlit web interface for AI Product Photo Detector."""

import os
from datetime import datetime

import httpx
import pandas as pd
import streamlit as st
from PIL import Image

# Configuration
API_URL = os.getenv("API_URL", "http://localhost:8080")
MAX_FILE_SIZE_MB = 5

# Session state initialization
if "history" not in st.session_state:
    st.session_state.history = []
if "last_result" not in st.session_state:
    st.session_state.last_result = None


def init_page() -> None:
    """Initialize Streamlit page configuration."""
    st.set_page_config(
        page_title="AI Product Photo Detector",
        page_icon="search",
        layout="centered",
        initial_sidebar_state="expanded",
    )


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
        files = {"file": (filename, image_bytes, "image/jpeg")}
        response = httpx.post(f"{API_URL}/predict", files=files, timeout=30.0)
        if response.status_code == 200:
            return response.json()
        else:
            try:
                detail = response.json().get("detail", response.text)
            except Exception:
                detail = response.text
            st.error(f"API Error ({response.status_code}): {detail}")
    except Exception as e:
        st.error(f"Connection error: {e}")
    return None


def add_to_history(filename: str, result: dict) -> None:
    """Add prediction to session history."""
    st.session_state.history.append({
        "timestamp": datetime.now().isoformat(),
        "filename": filename,
        "prediction": result.get("prediction", "unknown"),
        "probability": result.get("probability", 0),
        "confidence": result.get("confidence", "unknown"),
    })
    # Keep only last 50 entries
    if len(st.session_state.history) > 50:
        st.session_state.history = st.session_state.history[-50:]


def display_result(result: dict) -> None:
    """Display prediction result."""
    prediction = result.get("prediction", "unknown")
    probability = result.get("probability", 0)
    confidence = result.get("confidence", "unknown")
    inference_time = result.get("inference_time_ms", 0)

    if prediction == "ai_generated":
        color = "red"
        icon = "[AI]"
        label = "AI-Generated"
    else:
        color = "green"
        icon = "[Real]"
        label = "Real Photo"

    st.markdown(
        f"""
        <div style="
            padding: 20px;
            border-radius: 10px;
            background-color: {"#ffebee" if prediction == "ai_generated" else "#e8f5e9"};
            border: 2px solid {color};
            text-align: center;
            margin: 20px 0;
        ">
            <h1 style="color: {color}; margin: 0;">{icon} {label}</h1>
            <p style="font-size: 24px; margin: 10px 0;">
                Probability: <strong>{probability:.1%}</strong>
            </p>
            <p style="color: gray;">
                Confidence: {confidence.upper()} | Inference: {inference_time:.1f}ms
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.progress(probability)


def main() -> None:
    """Main Streamlit application."""
    init_page()

    st.title("AI Product Photo Detector")
    st.markdown("Upload a product image to check if it's **AI-generated** or a **real photo**.")

    # Sidebar
    with st.sidebar:
        st.header("Status")
        health = check_api_health()
        if health and health.get("status") == "healthy":
            st.success("API Online")
            st.caption(f"Model: {health.get('model_version', 'unknown')}")
        else:
            st.error("API Offline")

        st.markdown("---")
        st.header("About")
        st.markdown("""
        Detects AI-generated product photos using deep learning.

        **Supported formats:** JPEG, PNG, WebP
        **Max file size:** 5 MB
        """)
        st.caption("Built by Nolan Cacheux")

    # Main content
    st.markdown("---")

    # File upload
    uploaded_file = st.file_uploader(
        "Upload an image",
        type=["jpg", "jpeg", "png", "webp"],
    )

    if uploaded_file is not None:
        # Reset last result when a new file is uploaded
        if st.session_state.get("last_filename") != uploaded_file.name:
            st.session_state.last_result = None
            st.session_state.last_filename = uploaded_file.name

        file_size_mb = len(uploaded_file.getvalue()) / (1024 * 1024)
        if file_size_mb > MAX_FILE_SIZE_MB:
            st.error(f"File too large! Max size: {MAX_FILE_SIZE_MB}MB")
            return

        col1, col2 = st.columns([1, 1])

        with col1:
            st.subheader("Uploaded Image")
            image = Image.open(uploaded_file)
            st.image(image, use_container_width=True)
            st.caption(f"Size: {image.size[0]}x{image.size[1]} | {file_size_mb:.2f}MB")

        with col2:
            st.subheader("Result")
            if st.button("Analyze Image", type="primary", use_container_width=True):
                with st.spinner("Analyzing..."):
                    uploaded_file.seek(0)
                    image_bytes = uploaded_file.read()
                    result = predict_image(image_bytes, uploaded_file.name)
                    if result:
                        st.session_state.last_result = result
                        add_to_history(uploaded_file.name, result)

            if st.session_state.last_result is not None:
                display_result(st.session_state.last_result)
            else:
                st.info("Click the button above to analyze")

    # History
    if st.session_state.history:
        st.markdown("---")
        st.subheader("Recent Predictions")
        df = pd.DataFrame(st.session_state.history[-10:])
        st.dataframe(df[["filename", "prediction", "probability", "confidence"]], hide_index=True)


if __name__ == "__main__":
    main()
