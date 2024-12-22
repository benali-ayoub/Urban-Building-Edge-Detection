"""
Functions for noise reduction and edge cleanup.
"""

import cv2
import numpy as np


def remove_small_components(
    image: np.ndarray, threshold: int = 50, min_size: int = 500
) -> np.ndarray:
    """
    Process edge detection output by thresholding and removing small components.

    Args:
        image: Input edge detection image
        threshold: Threshold value for edge detection (0-255)
        min_size: Minimum component size to keep

    Returns:
        Processed binary image
    """
    # Ensure image is in correct format
    if image.dtype != np.uint8:
        image = np.uint8(np.clip(image, 0, 255))

    # Apply threshold to create binary image
    _, binary = cv2.threshold(image, threshold, 255, cv2.THRESH_BINARY)

    # Remove small components
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
        binary, connectivity=8
    )
    result = np.zeros_like(binary)

    for label in range(1, num_labels):  # Start from 1 to skip background
        if stats[label, cv2.CC_STAT_AREA] >= min_size:
            result[labels == label] = 255

    return result


def clean_edges(image: np.ndarray, kernel_size: int = 3) -> np.ndarray:
    """
    Apply morphological operations to clean up edges.

    Args:
        image: Binary edge image
        kernel_size: Size of the morphological kernel

    Returns:
        Cleaned edge image
    """
    kernel = np.ones((kernel_size, kernel_size), np.uint8)
    result = cv2.morphologyEx(image, cv2.MORPH_CLOSE, kernel)
    result = cv2.morphologyEx(result, cv2.MORPH_OPEN, kernel)
    return result
