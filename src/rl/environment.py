"""
nvironment for preprocessing parameter optimization.
"""

import numpy as np
import cv2
from typing import Dict, Tuple, Any
import gym
from gym import spaces


class PreprocessingEnv(gym.Env):
    def __init__(self, image: np.ndarray, ground_truth: np.ndarray = None):
        super().__init__()
        self.image = image
        self.ground_truth = ground_truth
        self.is_training = ground_truth is not None

        # Define action spaces for each hyperparameter
        self.action_space = spaces.Dict(
            {
                "bilateral_d": spaces.Discrete(15),  # 1-15
                "bilateral_sigma": spaces.Discrete(150),  # 1-150
                "clahe_clip": spaces.Discrete(40),  # 0.1-4.0
                "clahe_grid": spaces.Discrete(4),  # (4,4) to (16,16)
                "gaussian_kernel": spaces.Discrete(3),  # 3,5,7
                "threshold_value": spaces.Discrete(200),  # 100-300
            }
        )

        # State space includes image features
        self.observation_space = spaces.Box(
            low=0, high=255, shape=(6,), dtype=np.float32
        )

    def _get_state(self) -> np.ndarray:
        """Extract state features from current image."""
        features = []
        gray = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)

        # Basic statistics (6 features)
        features.extend(
            [
                np.mean(gray),
                np.std(gray),
                np.median(gray),
                cv2.Laplacian(gray, cv2.CV_64F).var(),
                np.percentile(gray, 25),  # First quartile
                np.percentile(gray, 75),  # Third quartile
            ]
        )

        return np.array(features, dtype=np.float32)

    def _apply_preprocessing(self, actions: Dict[str, int]) -> np.ndarray:
        """Apply preprocessing with given parameters."""
        img = self.image.copy()
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Apply bilateral filter
        d = actions["bilateral_d"] + 1
        sigma = actions["bilateral_sigma"] + 1
        bilateral = cv2.bilateralFilter(gray, d, sigma, sigma)

        # Apply CLAHE
        clip_limit = (actions["clahe_clip"] + 1) / 10.0
        grid_size = 4 + (actions["clahe_grid"] * 4)
        clahe = cv2.createCLAHE(
            clipLimit=clip_limit, tileGridSize=(grid_size, grid_size)
        )
        enhanced = clahe.apply(bilateral)

        # Apply Gaussian blur
        kernel_size = 2 * actions["gaussian_kernel"] + 3
        blurred = cv2.GaussianBlur(enhanced, (kernel_size, kernel_size), 0)

        # Apply threshold
        threshold = actions["threshold_value"] + 100
        _, processed = cv2.threshold(blurred, threshold, 255, cv2.THRESH_BINARY)

        return processed

    def _calculate_reward(self, processed: np.ndarray) -> float:
        """Calculate reward based on image quality metrics."""
        if not self.is_training or self.ground_truth is None:
            # Quality metrics for inference mode
            # Edge detection quality
            edges = cv2.Canny(processed, 100, 200)
            edge_density = np.mean(edges) / 255.0

            # Contrast
            contrast = np.std(processed) / 128.0

            # Noise estimation (lower is better)
            noise = cv2.Laplacian(processed, cv2.CV_64F).var() / 10000.0
            noise_penalty = np.exp(-noise)  # Convert to reward

            # Histogram spread
            hist = cv2.calcHist([processed], [0], None, [256], [0, 256])
            hist = hist.flatten() / hist.sum()
            hist_entropy = -np.sum(hist * np.log2(hist + 1e-10))
            hist_reward = hist_entropy / 8.0  # Normalize

            # Combine metrics
            reward = (
                0.3 * edge_density  # Clear edges
                + 0.3 * contrast  # Good contrast
                + 0.2 * noise_penalty  # Low noise
                + 0.2 * hist_reward  # Good histogram distribution
            )

            return reward

        else:
            # Training mode with ground truth comparison
            from skimage.metrics import structural_similarity as ssim

            # Ensure same dimensions
            if processed.shape != self.ground_truth.shape:
                processed = cv2.resize(processed, self.ground_truth.shape[::-1])

            # Structural similarity
            similarity = ssim(processed, self.ground_truth)

            # Edge similarity
            edges_processed = cv2.Canny(processed, 100, 200)
            edges_gt = cv2.Canny(self.ground_truth, 100, 200)
            edge_similarity = np.mean(np.equal(edges_processed, edges_gt))

            # Combined reward
            reward = 0.6 * similarity + 0.4 * edge_similarity

            return reward

    def step(self, action: Dict[str, int]) -> Tuple[np.ndarray, float, bool, Dict]:
        """Execute one step in the environment."""
        try:
            processed = self._apply_preprocessing(action)
            reward = self._calculate_reward(processed)
            self.current_state = self._get_state()

            return self.current_state, reward, True, {"processed": processed}
        except Exception as e:
            print(f"Error in step: {str(e)}")
            raise

    def reset(self) -> np.ndarray:
        self.current_state = self._get_state()
        return self.current_state
