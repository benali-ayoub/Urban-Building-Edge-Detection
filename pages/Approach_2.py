import streamlit as st
import matplotlib.pyplot as plt
import cv2
import numpy as np
from src.building_detector import BuildingEdgeDetector
from src.rl.approach_2.edge_processor import get_edges
from src.rl.approach_2.pipeline_detector import pipeline_selector
from src.rl.trainer import get_optimal_parameters


def load_image(image_file):
    """Load and convert uploaded image to CV2 format"""
    file_bytes = np.asarray(bytearray(image_file.read()), dtype=np.uint8)
    return cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)


def main():
    # Page configuration
    st.set_page_config(
        page_title="Building Edge Detection AI", page_icon="🏢", layout="wide"
    )

    # Header
    st.title("🏢 Intelligent Building Edge Detection")
    st.markdown(
        """
    This application uses Reinforcement Learning to automatically detect building edges in images.
    The AI model determines optimal preprocessing parameters for each unique image.
    """
    )

    show_params = st.sidebar.checkbox(
        "Show optimal parameters",
        value=True,
        help="Display the AI-selected preprocessing parameters",
    )

    # File uploader
    st.subheader("Upload Image")
    uploaded_file = st.file_uploader(
        "Choose an image...",
        type=["png", "jpg", "jpeg"],
        help="Upload an image containing buildings",
    )

    if uploaded_file is not None:
        # Create columns for before/after comparison
        col1, col2, col3 = st.columns(3)

        # Load and display original image
        image = load_image(uploaded_file)
        with col1:
            st.subheader("Original Image")
            st.image(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

        # Progress indicator
        with st.spinner("AI is analyzing the image..."):
            # Get optimal parameters

            processed, pipeline_str, parameters = pipeline_selector(
                model_path="src/models/pipeline_app_2.pt",
                image=cv2.cvtColor(image, cv2.COLOR_BGR2GRAY),
            )

            result = get_edges(
                model_path="src/models/edge_detection_agent_20241221_181324_episode_0.pth",
                image=processed,
            )

        # Display result

        with col2:
            st.subheader("Preprocessed Image")
            st.image(processed, caption="Enhanced image")

        with col3:
            st.subheader("Detected Edges")
            st.image(result, caption="AI-detected building edges")

        # Display parameters if selected
        if show_params:
            st.subheader("🔧 Optimal Parameters")
            col1, col2 = st.columns(2)

        with col1:
            st.markdown("**Median Blur:**")
            st.write(f"- Kernel Size: {parameters['median_kernel']:.2f}")

            st.markdown("**CLAHE:**")
            st.write(f"- Clip Limit: {parameters['clahe_clip']:.2f}")
            st.write(
                f"- Grid Size: {parameters['clahe_grid']}x{parameters['clahe_grid']}"
            )

            st.markdown("**Adaptive Thresholding:**")
            st.write(f"- Block Size: {parameters['adaptive_block']:.2f}")
            st.write(f"- C Value: {parameters['adaptive_C']:.2f}")

        with col2:
            st.markdown("**Gaussian Blur:**")
            st.write(
                f"- Kernel Size: {parameters['gaussian_kernel']}x{parameters['gaussian_kernel']}"
            )
            st.write(f"- Sigma: {parameters['gaussian_sigma']:.2f}")

        # Download button for processed image
        st.download_button(
            label="Download Processed Image",
            data=cv2.imencode(".png", result)[1].tobytes(),
            file_name="detected_edges.png",
            mime="image/png",
        )

    # Information section
    st.markdown(
        """
    ---
    ### How it works
    1. **Upload** an image containing buildings
    2. The **AI model** analyzes the image and determines optimal preprocessing parameters
    3. These parameters are used to **enhance** the image
    4. Building edges are **detected** using the optimized parameters
    5. **Download** the processed image for your use
    
    ### Tips for best results
    - Use clear images with good lighting
    - Ensure buildings are the main focus
    - Higher resolution images typically work better
    - Try adjusting the number of parameter samples if needed
    """
    )


if __name__ == "__main__":
    main()
