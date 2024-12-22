from src.rl.approach_2.image_processor import ImageProcessor


def pipeline_selector(model_path, image):
    """Process a single image and return the result

    Args:
        model_path (str): Path to the model file
        image (numpy.ndarray): Input image array

    Returns:
        tuple: (processed_image, pipeline_name, parameters)
    """
    processor = ImageProcessor(model_path)

    pipeline_names = [
        "Filtering -> Contrast -> Thresholding",
        "Contrast -> Filtering -> Thresholding",
        "Contrast -> Thresholding -> Filtering",
    ]

    # Process image
    processed, pipeline_id, params = processor.process_image(image)
    print(f"Selected pipeline: {params}")

    # Create parameters dictionary

    parameters = {
        "median_kernel": float(params[0]),
        "gaussian_kernel": float(params[1]),
        "gaussian_sigma": float(params[2]),
        "clahe_clip": float(params[3]),
        "clahe_grid": float(params[4]),
        "adaptive_block": float(params[5]),
        "adaptive_C": float(params[6]),
    }

    return processed, pipeline_names[pipeline_id], parameters
