import gym
from gym import spaces
import numpy as np
import cv2
import torch
import torch.nn as nn
import matplotlib.pyplot as plt

from src.rl.approach_2.image_metrics import ImageMetrics


class EdgeDetectionEnv:
    def __init__(self, input_image, ground_truth=None):
        self.input_image = input_image
        self.input_image_inv = cv2.bitwise_not(input_image)
        self.ground_truth = (
            ground_truth if ground_truth is not None else np.zeros_like(input_image)
        )

        self.action_space = spaces.Box(
            low=np.array([0.1, 0.2, 0.5, 0.1, 0.1], dtype=np.float32),
            high=np.array([0.5, 1.0, 2.0, 0.5, 0.3], dtype=np.float32),
        )

        self.observation_space = spaces.Box(
            low=0, high=255, shape=input_image.shape, dtype=np.uint8
        )

        self.current_state = self.input_image.copy()

    def process_edges(self, params):
        try:
            img = self.input_image_inv.copy()

            # Gaussian blur
            sigma = float(params[2])
            blurred = cv2.GaussianBlur(img, (3, 3), sigma)

            # Canny edge detection
            low_threshold = float(params[0] * 100)
            high_threshold = float(params[1] * 200)
            edges = cv2.Canny(blurred, low_threshold, high_threshold)

            # Dilate
            kernel_size = max(1, int(params[3] * 3))
            kernel = np.ones((kernel_size, kernel_size), np.uint8)
            dilated = cv2.dilate(edges, kernel, iterations=1)

            # Thinning
            thinned = cv2.ximgproc.thinning(dilated)

            # Remove small components
            num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
                thinned, connectivity=8
            )
            min_size = int(params[4] * 50)
            clean_edges = np.zeros_like(thinned)

            for i in range(1, num_labels):
                if stats[i, cv2.CC_STAT_AREA] >= min_size:
                    clean_edges[labels == i] = 255

            result = cv2.ximgproc.thinning(clean_edges)

            if np.mean(self.ground_truth) < 127:
                result = cv2.bitwise_not(result)

            return result

        except Exception as e:
            print(f"Error in edge processing: {str(e)}")
            return np.zeros_like(self.input_image)

    def step(self, action):
        processed = self.process_edges(action)
        self.current_state = processed
        return processed, 0, False, {}

    def reset(self):
        self.current_state = self.input_image.copy()
        return self.current_state


class EdgeDetectionAgent:
    def __init__(self, env):
        self.env = env
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.n_actions = env.action_space.shape[0]
        self.model = DQN(env.observation_space.shape, self.n_actions).to(self.device)

        self.action_low = env.action_space.low
        self.action_high = env.action_space.high
        self.action_mean = (self.action_high + self.action_low) / 2
        self.action_std = (self.action_high - self.action_low) / 4

    def act(self, state):
        with torch.no_grad():
            state = torch.FloatTensor(state).unsqueeze(0).unsqueeze(0).to(self.device)
            actions = self.model(state).cpu().numpy()[0]
            return np.clip(actions, self.action_low, self.action_high)


class DQN(nn.Module):
    def __init__(self, input_shape, n_actions):
        super(DQN, self).__init__()

        self.adaptive_pool = nn.AdaptiveAvgPool2d((32, 32))

        self.conv_block1 = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )

        self.conv_block2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )

        self.conv_block3 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )

        self._feature_size = 128 * 4 * 4

        self.fc = nn.Sequential(
            nn.Linear(self._feature_size, 512),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(256, n_actions),
        )

    def forward(self, x):
        x = self.adaptive_pool(x)
        x = self.conv_block1(x)
        x = self.conv_block2(x)
        x = self.conv_block3(x)
        x = x.view(x.size(0), -1)
        return self.fc(x)
