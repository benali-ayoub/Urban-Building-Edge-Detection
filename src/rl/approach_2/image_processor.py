import cv2
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import os
import json
from datetime import datetime

from src.rl.approach_2.pipeline_agent import DQN


class ImageProcessor:
    def __init__(self, model_path):
        self.device = torch.device("cpu")
        self.checkpoint = torch.load(
            model_path, map_location=self.device, weights_only=True
        )

        self.model = DQN(input_shape=(256, 256), n_actions=7, n_pipelines=3).to(
            self.device
        )

        self.model.load_state_dict(self.checkpoint["model_state_dict"])
        self.model.eval()

    def apply_filtering(self, image, params):
        """Apply filtering operations"""
        try:
            # Median Filter
            median_kernel = max(3, int(params[0]))
            if median_kernel % 2 == 0:
                median_kernel += 1
            image = cv2.medianBlur(image, median_kernel)

            # Gaussian Filter
            gaussian_kernel = max(3, int(params[1]))
            if gaussian_kernel % 2 == 0:
                gaussian_kernel += 1
            sigma = float(params[2])
            image = cv2.GaussianBlur(image, (gaussian_kernel, gaussian_kernel), sigma)
            return image
        except Exception as e:
            print(f"Error in filtering: {str(e)}")
            return image

    def apply_contrast(self, image, params):
        """Apply contrast enhancement"""
        try:
            clip_limit = float(params[3])
            grid_size = max(4, int(params[4]))
            clahe = cv2.createCLAHE(
                clipLimit=clip_limit, tileGridSize=(grid_size, grid_size)
            )
            return clahe.apply(image)
        except Exception as e:
            print(f"Error in contrast enhancement: {str(e)}")
            return image

    def apply_thresholding(self, image, params):
        """Apply thresholding"""
        try:
            block_size = int(params[5])
            if block_size % 2 == 0:
                block_size += 1
            return cv2.adaptiveThreshold(
                image,
                255,
                cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                cv2.THRESH_BINARY,
                block_size,
                params[6],
            )
        except Exception as e:
            print(f"Error in thresholding: {str(e)}")
            return image

    def apply_pipeline(self, image, pipeline_id, params):
        """Apply the selected pipeline with parameters"""
        try:
            # Pipeline 1: Filtering -> Contrast -> Thresholding
            if pipeline_id == 0:
                image = self.apply_filtering(image, params)
                image = self.apply_contrast(image, params)
                image = self.apply_thresholding(image, params)

            # Pipeline 2: Contrast -> Filtering -> Thresholding
            elif pipeline_id == 1:
                image = self.apply_contrast(image, params)
                image = self.apply_filtering(image, params)
                image = self.apply_thresholding(image, params)

            # Pipeline 3: Contrast -> Thresholding -> Filtering
            elif pipeline_id == 2:
                image = self.apply_contrast(image, params)
                image = self.apply_thresholding(image, params)
                image = self.apply_filtering(image, params)

            return image

        except Exception as e:
            print(f"Error in pipeline application: {str(e)}")
            return image

    def process_image(self, image):
        """Process a new image using the trained model"""

        with torch.no_grad():
            state = torch.FloatTensor(image).unsqueeze(0).unsqueeze(0).to(self.device)
            pipeline_logits, parameters = self.model(state)

            # Get pipeline choice
            pipeline_id = torch.argmax(pipeline_logits[0]).item()

            # Get and clip parameters
            parameters = parameters[0].cpu().numpy()
            parameters = np.clip(
                parameters,
                [3.0, 3.0, 0.5, 1.0, 4.0, 11.0, 5.0],
                [11.0, 11.0, 2.0, 3.0, 8.0, 31.0, 15.0],
            )

        processed = self.apply_pipeline(image, pipeline_id, parameters)
        return processed, pipeline_id, parameters
