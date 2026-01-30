"""Streamlit web interface for AI Product Photo Detector."""

import io
import os
from datetime import datetime

import httpx
import pandas as pd
import streamlit as st
from PIL import Image

# Configuration
API_URL = os.getenv("API_URL", "http://localhost:8000")
MAX_FILE_SIZE_MB = 10

# Session state initialization
if "history" not in st.session_state:
    st.session_state.history = []
if "dark_mode" not in st.session_state:
    st.session_state.dark_mode = False


def init_page() -> None:
    """Initialize Streamlit page configuration."""
    st.set_page_config(
        page_title="AI Product Photo Detector",
        page_icon="🔍",
        layout="wide",
        initial_sidebar_state="expanded",
    )


def check_api_health() -> dict | None:
    """Check if API is healthy.

    Returns:
        Health response or None if unavailable.
    """
    try:
        response = httpx.get(f"{API_URL}/health", timeout=5.0)
        if response.status_code == 200:
            return response.json()
    except Exception:
        pass
    return None


def get_explanation(
    image_bytes: bytes, filename: str, alpha: float = 0.5
) -> tuple[Image.Image | None, float | None]:
    """Get GradCAM explanation from API.

    Args:
        image_bytes: Image file bytes.
        filename: Original filename.
        alpha: Heatmap overlay transparency.

    Returns:
        Tuple of (explanation image, probability) or (None, None) if failed.
    """
    try:
        files = {"file": (filename, image_bytes, "image/jpeg")}
        response = httpx.post(
            f"{API_URL}/explain",
            files=files,
            params={"alpha": alpha},
            timeout=30.0,
        )
        if response.status_code == 200:
            # Get probability from header
            probability = float(response.headers.get("X-Prediction-Probability", 0))
            # Load image from response
            explanation_img = Image.open(io.BytesIO(response.content))
            return explanation_img, probability
    except Exception as e:
        st.error(f"Explanation error: {e}")
    return None, None


def add_to_history(filename: str, result: dict) -> None:
    """Add prediction to session history.

    Args:
        filename: Image filename.
        result: Prediction result.
    """
    st.session_state.history.append(
        {
            "timestamp": datetime.now().isoformat(),
            "filename": filename,
            "prediction": result.get("prediction", "unknown"),
            "probability": result.get("probability", 0),
            "confidence": result.get("confidence", "unknown"),
            "inference_time_ms": result.get("inference_time_ms", 0),
        }
    )

    # Keep only last 50 entries
    if len(st.session_state.history) > 50:
        st.session_state.history = st.session_state.history[-50:]


def predict_image(image_bytes: bytes, filename: str) -> dict | None:
    """Send image to API for prediction.

    Args:
        image_bytes: Image file bytes.
        filename: Original filename.

    Returns:
        Prediction response or None if failed.
    """
    try:
        files = {"file": (filename, image_bytes, "image/jpeg")}
        response = httpx.post(
            f"{API_URL}/predict",
            files=files,
            timeout=30.0,
        )
        if response.status_code == 200:
            return response.json()
        else:
            st.error(f"API Error: {response.json()}")
    except Exception as e:
        st.error(f"Connection error: {e}")
    return None


def display_result(result: dict) -> None:
    """Display prediction result.

    Args:
        result: Prediction response from API.
    """
    prediction = result.get("prediction", "unknown")
    probability = result.get("probability", 0)
    confidence = result.get("confidence", "unknown")
    inference_time = result.get("inference_time_ms", 0)

    # Determine color based on prediction
    if prediction == "ai_generated":
        color = "red"
        icon = "🤖"
        label = "AI-Generated"
    else:
        color = "green"
        icon = "📸"
        label = "Real Photo"

    # Display main result
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
            Confidence: {confidence.upper()} |
            Inference: {inference_time:.1f}ms
        </p>
    </div>
    """,
        unsafe_allow_html=True,
    )

    # Probability bar
    st.progress(probability)

    # Confidence explanation
    st.markdown("---")
    with st.expander("📊 How to interpret results"):
        st.markdown("""
        - **Probability**: Likelihood the image is AI-generated (0% = definitely real, 100% = definitely AI)
        - **Confidence Levels**:
            - 🟢 **HIGH**: Very confident in the prediction (prob < 20% or > 80%)
            - 🟡 **MEDIUM**: Moderately confident (prob 20-80%)
            - 🔴 **LOW**: Uncertain, near the decision boundary (~50%)
        """)


def display_history() -> None:
    """Display prediction history."""
    if not st.session_state.history:
        st.info("No predictions yet. Upload an image to get started!")
        return

    df = pd.DataFrame(st.session_state.history)
    df["timestamp"] = pd.to_datetime(df["timestamp"])

    # Summary stats
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Total Predictions", len(df))
    with col2:
        ai_count = (df["prediction"] == "ai_generated").sum()
        st.metric("AI-Generated", ai_count)
    with col3:
        real_count = (df["prediction"] == "real").sum()
        st.metric("Real Photos", real_count)

    # History table
    st.dataframe(
        df[
            [
                "timestamp",
                "filename",
                "prediction",
                "probability",
                "confidence",
                "inference_time_ms",
            ]
        ].sort_values("timestamp", ascending=False),
        use_container_width=True,
        hide_index=True,
    )

    # Export button
    if st.button("📥 Export History (CSV)"):
        csv = df.to_csv(index=False)
        st.download_button(
            label="Download CSV",
            data=csv,
            file_name=f"prediction_history_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
            mime="text/csv",
        )

    # Clear history
    if st.button("🗑️ Clear History"):
        st.session_state.history = []
        st.rerun()


def batch_upload_page() -> None:
    """Batch upload page."""
    st.subheader("📦 Batch Analysis")
    st.markdown("Upload multiple images for batch processing.")

    uploaded_files = st.file_uploader(
        "Upload images",
        type=["jpg", "jpeg", "png", "webp"],
        accept_multiple_files=True,
        key="batch_uploader",
    )

    if uploaded_files:
        st.write(f"📁 {len(uploaded_files)} files selected")

        if st.button("🔍 Analyze All", type="primary"):
            progress_bar = st.progress(0)
            results = []

            for i, file in enumerate(uploaded_files):
                file.seek(0)
                image_bytes = file.read()

                try:
                    files = {"file": (file.name, image_bytes, "image/jpeg")}
                    response = httpx.post(
                        f"{API_URL}/predict",
                        files=files,
                        timeout=30.0,
                    )
                    if response.status_code == 200:
                        result = response.json()
                        results.append(
                            {
                                "filename": file.name,
                                "prediction": result["prediction"],
                                "probability": result["probability"],
                                "confidence": result["confidence"],
                            }
                        )
                        add_to_history(file.name, result)
                except Exception as e:
                    results.append(
                        {
                            "filename": file.name,
                            "prediction": "error",
                            "probability": 0,
                            "confidence": str(e),
                        }
                    )

                progress_bar.progress((i + 1) / len(uploaded_files))

            # Display results
            df = pd.DataFrame(results)
            st.dataframe(df, use_container_width=True, hide_index=True)

            # Summary
            ai_count = (df["prediction"] == "ai_generated").sum()
            real_count = (df["prediction"] == "real").sum()
            st.success(f"✅ Analysis complete: {ai_count} AI-generated, {real_count} real photos")


def main() -> None:
    """Main Streamlit application."""
    init_page()

    # Header
    st.title("🔍 AI Product Photo Detector")
    st.markdown("""
    Upload a product image to check if it's **AI-generated** or a **real photo**.
    Useful for detecting fraudulent e-commerce listings using synthetic images.
    """)

    # Sidebar
    with st.sidebar:
        st.header("⚙️ Settings")

        # API Status
        health = check_api_health()
        if health and health.get("status") == "healthy":
            st.success("🟢 API Online")
            st.caption(f"Model: {health.get('model_version', 'unknown')}")
        else:
            st.error("🔴 API Offline")
            st.caption("Make sure the API is running on port 8000")

        st.markdown("---")

        # Info
        st.header("ℹ️ About")
        st.markdown("""
        This tool uses a deep learning model trained to distinguish
        between real product photos and AI-generated images.

        **Supported formats:**
        - JPEG
        - PNG
        - WebP

        **Max file size:** 10 MB
        """)

        st.markdown("---")
        st.caption("Built with ❤️ by Nolan Cacheux")

    # Main content - Tabs
    st.markdown("---")

    tab1, tab2, tab3 = st.tabs(["🔍 Single Analysis", "📦 Batch Analysis", "📊 History"])

    with tab1:
        # File upload
        uploaded_file = st.file_uploader(
            "📁 Upload an image",
            type=["jpg", "jpeg", "png", "webp"],
            help="Upload a product image to analyze",
            key="single_uploader",
        )

        if uploaded_file is not None:
            # Check file size
            file_size_mb = len(uploaded_file.getvalue()) / (1024 * 1024)
            if file_size_mb > MAX_FILE_SIZE_MB:
                st.error(f"File too large! Max size: {MAX_FILE_SIZE_MB}MB")
            else:
                # Display image
                col1, col2 = st.columns([1, 1])

                with col1:
                    st.subheader("📷 Uploaded Image")
                    image = Image.open(uploaded_file)
                    st.image(image, use_container_width=True)
                    st.caption(f"Size: {image.size[0]}x{image.size[1]} | {file_size_mb:.2f}MB")

                with col2:
                    st.subheader("🔮 Analysis Result")

                    # Options
                    show_explanation = st.checkbox("Show GradCAM explanation", value=False)
                    if show_explanation:
                        alpha = st.slider("Heatmap transparency", 0.1, 0.9, 0.5)

                    # Analyze button
                    if st.button("🔍 Analyze Image", type="primary", use_container_width=True):
                        with st.spinner("Analyzing..."):
                            # Reset file position and read bytes
                            uploaded_file.seek(0)
                            image_bytes = uploaded_file.read()

                            # Get prediction
                            result = predict_image(image_bytes, uploaded_file.name)

                            if result:
                                display_result(result)
                                add_to_history(uploaded_file.name, result)

                                # Show explanation if requested
                                if show_explanation:
                                    st.markdown("---")
                                    st.subheader("🧠 Model Explanation (GradCAM)")
                                    with st.spinner("Generating explanation..."):
                                        uploaded_file.seek(0)
                                        image_bytes = uploaded_file.read()
                                        explanation_img, _ = get_explanation(
                                            image_bytes, uploaded_file.name, alpha
                                        )
                                        if explanation_img:
                                            st.image(explanation_img, use_container_width=True)
                                            st.caption(
                                                "Heatmap shows regions the model focused on. "
                                                "Red = high importance, Blue = low importance."
                                            )
                    else:
                        st.info("👆 Click the button above to analyze the image")

        else:
            # How it works section
            st.subheader("📌 How it works")

            col1, col2, col3 = st.columns(3)

            with col1:
                st.markdown("### 1️⃣ Upload")
                st.markdown("Upload a product image from your device")

            with col2:
                st.markdown("### 2️⃣ Analyze")
                st.markdown("Our AI model processes the image")

            with col3:
                st.markdown("### 3️⃣ Result")
                st.markdown("Get instant classification with confidence score")

    with tab2:
        batch_upload_page()

    with tab3:
        display_history()


if __name__ == "__main__":
    main()
