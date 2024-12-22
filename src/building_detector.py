from enum import Enum
from dataclasses import dataclass
import numpy as np
import cv2
from typing import List, Tuple
from .utils.image_processing import apply_bilateral_filter, enhance_contrast
from .edge_detection.edge_processors import detect_edges, enhance_building_edges
from .edge_detection.noise_reduction import remove_small_components, clean_edges


@dataclass
class ImageCharacteristics:
    """Store image characteristics for processing decision making."""

    brightness_mean: float
    brightness_std: float
    contrast: float
    noise_level: float
    edge_density: float
    texture_strength: float


class ProcessingStep(Enum):
    """Available processing steps."""

    BILATERAL_FILTER = "bilateral_filter"
    CONTRAST_ENHANCEMENT = "contrast_enhancement"
    GAUSSIAN_BLUR = "gaussian_blur"
    SKY_REMOVAL = "sky_removal"
    THRESHOLDING = "thresholding"


class BuildingEdgeDetector:
    def __init__(self, sky_threshold: int = 400, optimal_params: dict = None):
        self.sky_threshold = sky_threshold
        self.optimal_params = optimal_params

    def remove_small_components(
        self, binary_image: np.ndarray, min_size: int = 100
    ) -> np.ndarray:
        """
        Remove small connected components from the binary image.

        Args:
            binary_image: Binary image with edges
            min_size: Minimum size of components to keep

        Returns:
            Cleaned binary image
        """
        # Find connected components
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
            binary_image, connectivity=8
        )

        # Create output image
        cleaned = np.zeros_like(binary_image)

        # Skip background (label 0)
        for label in range(1, num_labels):
            size = stats[label, cv2.CC_STAT_AREA]
            if size >= min_size:
                cleaned[labels == label] = 255

        return cleaned

    def filter_building_edges(self, edges: np.ndarray) -> np.ndarray:
        """
        Filter edges to keep only those likely to belong to buildings.
        """
        # Remove small components
        cleaned = self.remove_small_components(edges, min_size=100)

        # Find contours
        contours, _ = cv2.findContours(
            cleaned, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )

        # Filter contours based on shape and size
        filtered = np.zeros_like(edges)
        for contour in contours:
            # Calculate contour properties
            area = cv2.contourArea(contour)
            perimeter = cv2.arcLength(contour, True)

            # Skip if area is too small
            if area < 100:
                continue

            # Calculate shape metrics
            if perimeter > 0:
                circularity = 4 * np.pi * area / (perimeter * perimeter)

                # Filter based on circularity (buildings typically aren't circular)
                if circularity < 0.8:  # Adjust this threshold as needed
                    # Draw the contour
                    cv2.drawContours(filtered, [contour], -1, 255, 1)

        return filtered

    def _analyze_image(self, image: np.ndarray) -> ImageCharacteristics:
        """
        Analyze image characteristics to determine optimal processing order.

        Args:
            image: Grayscale input image

        Returns:
            ImageCharacteristics object containing analysis results
        """
        # Calculate brightness statistics
        brightness_mean = np.mean(image)
        brightness_std = np.std(image)

        # Calculate contrast using histogram
        hist = cv2.calcHist([image], [0], None, [256], [0, 256])
        hist_norm = hist.ravel() / hist.sum()
        contrast = np.sqrt(np.sum((np.arange(256) - brightness_mean) ** 2 * hist_norm))

        # Estimate noise level using Laplacian
        noise_level = cv2.Laplacian(image, cv2.CV_64F).var()

        # Calculate edge density using Sobel
        sobelx = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=3)
        sobely = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=3)
        edge_density = np.mean(np.sqrt(sobelx**2 + sobely**2))

        # Calculate texture strength using GLCM
        texture_strength = self._calculate_texture_strength(image)

        return ImageCharacteristics(
            brightness_mean=brightness_mean,
            brightness_std=brightness_std,
            contrast=contrast,
            noise_level=noise_level,
            edge_density=edge_density,
            texture_strength=texture_strength,
        )

    def _calculate_texture_strength(self, image: np.ndarray) -> float:
        """
        Calculate texture strength using gradient magnitude statistics.

        Args:
            image: Grayscale input image

        Returns:
            Texture strength score
        """
        gradient_x = cv2.Sobel(image, cv2.CV_64F, 1, 0)
        gradient_y = cv2.Sobel(image, cv2.CV_64F, 0, 1)
        gradient_magnitude = np.sqrt(gradient_x**2 + gradient_y**2)
        return np.std(gradient_magnitude)

    def _determine_processing_order(
        self, characteristics: ImageCharacteristics
    ) -> List[ProcessingStep]:
        """
        Determine optimal processing order based on image characteristics.

        Args:
            characteristics: ImageCharacteristics object

        Returns:
            List of processing steps in optimal order
        """
        steps = []

        # High noise scenario
        if characteristics.noise_level > 100:
            steps.append(ProcessingStep.BILATERAL_FILTER)
            if characteristics.contrast < 40:
                steps.append(ProcessingStep.CONTRAST_ENHANCEMENT)
        # Low noise but poor contrast scenario
        elif characteristics.contrast < 40:
            steps.append(ProcessingStep.CONTRAST_ENHANCEMENT)
            steps.append(ProcessingStep.BILATERAL_FILTER)
        # High texture scenario
        elif characteristics.texture_strength > 50:
            steps.append(ProcessingStep.GAUSSIAN_BLUR)
            steps.append(ProcessingStep.BILATERAL_FILTER)
        # Default scenario
        else:
            steps.append(ProcessingStep.BILATERAL_FILTER)
            steps.append(ProcessingStep.CONTRAST_ENHANCEMENT)

        # Sky removal is always last before edge detection
        steps.append(ProcessingStep.SKY_REMOVAL)
        steps.append(ProcessingStep.THRESHOLDING)

        return steps

    def detect(self, image_path: str) -> np.ndarray:
        """
        Detect building edges in an image with adaptive processing.

        Args:
            image_path: Path to the input image

        Returns:
            Binary image containing building edges
        """
        # Handle different input types
        if isinstance(image_path, str):
            # Input is a file path
            img = cv2.imread(image_path)
            if img is None:
                raise ValueError(f"Could not load image from path: {image_path}")
        elif isinstance(image_path, np.ndarray):
            # Input is already a numpy array
            img = image_path.copy()
        else:
            raise ValueError("Input must be either a file path (str) or numpy array")

        # Convert to grayscale if needed
        if len(img.shape) == 3:
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        else:
            gray = img

        if img is None:
            raise ValueError(f"Could not load image from {image_path}")

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Analyze image and determine processing order
        characteristics = self._analyze_image(gray)
        processing_steps = self._determine_processing_order(characteristics)

        # Apply processing steps in determined order
        processed = gray.copy()
        for step in processing_steps:
            if step == ProcessingStep.BILATERAL_FILTER:
                processed = apply_bilateral_filter(
                    processed,
                    d=self.optimal_params["bilateral_d"],
                    sigma_color=self.optimal_params["bilateral_sigma"],
                    sigma_space=self.optimal_params["bilateral_sigma"],
                )

            elif step == ProcessingStep.CONTRAST_ENHANCEMENT:
                # Adjust CLAHE parameters based on contrast
                clahe_grid_size = self.optimal_params["clahe_grid"]
                processed = enhance_contrast(
                    processed,
                    clip_limit=self.optimal_params["clahe_clip"],
                    grid_size=(clahe_grid_size, clahe_grid_size),
                )

            elif step == ProcessingStep.GAUSSIAN_BLUR:
                kernel_size = self.optimal_params["gaussian_kernel"]
                processed = cv2.GaussianBlur(processed, (kernel_size, kernel_size), 0)

            elif step == ProcessingStep.SKY_REMOVAL:
                threshold = self.optimal_params["threshold_value"]
                _, sky_mask = cv2.threshold(
                    processed, threshold, 255, cv2.THRESH_BINARY
                )
                processed = cv2.bitwise_and(processed, processed, mask=~sky_mask)
            elif step == ProcessingStep.THRESHOLDING:
                processed = cv2.adaptiveThreshold(
                    processed,
                    255,
                    cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                    cv2.THRESH_BINARY,
                    41,
                    5,
                )

        # Edge detection and enhancement with adaptive parameters
        edge_image, optimal_params = detect_edges(
            processed, "src/models/edges_model.pth"
        )
        # edge_image = enhance_building_edges(edge_image)

        edges = remove_small_components(edge_image, min_size=300)

        # Final cleanup
        # edges = clean_edges(edges)
        # final_edges = self.filter_building_edges(edges)

        return edges, processed
