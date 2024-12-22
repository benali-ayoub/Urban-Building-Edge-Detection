import numpy as np
import cv2
from gymnasium import Env, spaces
from skimage.metrics import structural_similarity as ssim

class EdgeDetectionEnv(Env):
    def __init__(self, input_images, ground_truth_images):
        super().__init__()
        self.input_images = input_images
        self.ground_truth_images = ground_truth_images
        self.current_image_idx = 0
        
        # Action space: [ksize, scale, delta]
        # ksize: 1, 3, 5, 7
        # scale: 0.1 to 2.0 (discretized to 20 values)
        # delta: 0 to 255 (discretized to 26 values)
        self.action_space = spaces.MultiDiscrete([4, 20, 26])
        
        # Observation space: normalized image
        sample_shape = input_images[0].shape
        self.observation_space = spaces.Box(
            low=0, high=255, shape=sample_shape, dtype=np.uint8
        )
        
    def reset(self, seed=None):
        super().reset(seed=seed)
        self.current_image_idx = 0
        return self.input_images[self.current_image_idx], {}
    
    def step(self, action):
        # Convert actions to Sobel parameters
        ksize = 2 * action[0] + 3  # Maps to [3, 5, 7, 9]
        scale = action[1] * 0.1 + 0.1  # Maps to [0.1, 0.2, ..., 2.0]
        delta = action[2] * 10  # Maps to [0, 10, ..., 250]
        
        # Apply Sobel edge detection
        img = self.input_images[self.current_image_idx]
        grad_x = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=ksize, scale=scale, delta=delta)
        grad_y = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=ksize, scale=scale, delta=delta)
        edge_img = np.sqrt(grad_x**2 + grad_y**2)
        edge_img = np.uint8(np.clip(edge_img, 0, 255))
        
        # Calculate reward using SSIM
        reward = ssim(edge_img, self.ground_truth_images[self.current_image_idx])
        
        # Move to next image
        self.current_image_idx += 1
        done = self.current_image_idx >= len(self.input_images)
        
        return edge_img, reward, done, False, {}