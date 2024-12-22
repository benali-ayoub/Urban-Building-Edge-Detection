"""
Core edge detection and enhancement functionality.
"""

import cv2
import numpy as np
import numpy as np
import cv2
import torch

from src.rl.edge_agent import DQN
from src.rl.edge_env import EdgeDetectionEnv


def detect_edges(
    original_image: np.ndarray, model_path: str, env_class=EdgeDetectionEnv
) -> np.ndarray:
    """
    Test a trained model on a single image and return the optimal Sobel parameters.

    Args:
        model_path (str): Path to the saved model
        original_image (np.ndarray):  Image
        env_class: Environment class (default: EdgeDetectionEnv)

    Returns:
        tuple: (edge_detected_image, optimal_parameters, original_image)
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load and preprocess the test image
    if original_image is None:
        raise ValueError(f"Could not load image from {image_path}")

    # Create a dummy environment just to get the proper dimensions
    dummy_env = env_class(
        input_images=[original_image],
        ground_truth_images=[np.zeros_like(original_image)],  # Dummy ground truth
    )

    # Load the trained model
    model = DQN(
        dummy_env.observation_space.shape[0],
        dummy_env.observation_space.shape[1],
        dummy_env.action_space.nvec.prod(),
    ).to(device)

    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    # Prepare input
    state = torch.FloatTensor(original_image).unsqueeze(0).unsqueeze(0).to(device)

    # Get model prediction
    with torch.no_grad():
        action = model(state).max(1)[1].item()

    # Convert flat action back to multi-discrete
    action_tuple = np.unravel_index(action, dummy_env.action_space.nvec)

    # Convert actions to Sobel parameters
    ksize = 2 * action_tuple[0] + 3  # Maps to [3, 5, 7, 9]
    scale = action_tuple[1] * 0.1 + 0.1  # Maps to [0.1, 0.2, ..., 2.0]
    delta = action_tuple[2] * 10  # Maps to [0, 10, ..., 250]

    # Apply Sobel edge detection with the optimal parameters
    grad_x = cv2.Sobel(
        original_image, cv2.CV_64F, 1, 0, ksize=ksize, scale=scale, delta=delta
    )
    grad_y = cv2.Sobel(
        original_image, cv2.CV_64F, 0, 1, ksize=ksize, scale=scale, delta=delta
    )
    edge_image = np.sqrt(grad_x**2 + grad_y**2)
    edge_image = np.uint8(np.clip(edge_image, 0, 255))

    optimal_params = {"ksize": ksize, "scale": scale, "delta": delta}

    return edge_image, optimal_params


def enhance_building_edges(image: np.ndarray) -> np.ndarray:
    """
    Enhance edge detection specifically for buildings using multi-scale Laplacian.

    Args:
        image: Input edge image

    Returns:
        Enhanced binary edge image
    """
    blur1 = cv2.GaussianBlur(image, (3, 3), 0)
    blur2 = cv2.GaussianBlur(image, (5, 5), 0)

    laplacian1 = cv2.Laplacian(blur1, cv2.CV_64F)
    laplacian2 = cv2.Laplacian(blur2, cv2.CV_64F)

    combined = cv2.addWeighted(
        np.absolute(laplacian1), 0.5, np.absolute(laplacian2), 0.5, 0
    )
    combined = np.uint8(combined)

    _, binary = cv2.threshold(combined, 30, 255, cv2.THRESH_BINARY)
    return binary
