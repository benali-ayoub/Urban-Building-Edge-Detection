"""
Utility functions for basic image processing operations.
"""

import cv2
import numpy as np

def apply_bilateral_filter(image: np.ndarray, d: int = 11, sigma_color: int = 100, sigma_space: int = 100) -> np.ndarray:
    """
    Apply bilateral filter to reduce noise while preserving edges.
    
    Args:
        image: Input grayscale image
        d: Diameter of each pixel neighborhood
        sigma_color: Filter sigma in the color space
        sigma_space: Filter sigma in the coordinate space
    
    Returns:
        Filtered image
    """
    return cv2.bilateralFilter(image, d, sigma_color, sigma_space)

def enhance_contrast(image: np.ndarray, clip_limit: float = 10.0, grid_size: tuple = (8, 8)) -> np.ndarray:
    """
    Enhance image contrast using CLAHE.
    
    Args:
        image: Input grayscale image
        clip_limit: Threshold for contrast limiting
        grid_size: Size of grid for histogram equalization
    
    Returns:
        Contrast enhanced image
    """
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=grid_size)
    return clahe.apply(image)