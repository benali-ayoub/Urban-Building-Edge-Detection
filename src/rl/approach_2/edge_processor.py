import cv2
import numpy as np
import torch
from src.rl.approach_2.edge_agent import EdgeDetectionAgent
from src.rl.approach_2.edge_env import EdgeDetectionEnv


def get_edges(image, model_path):
    # Normalize image
    img = cv2.normalize(image, None, 0, 255, cv2.NORM_MINMAX)

    # Create environment
    env = EdgeDetectionEnv(img, np.zeros_like(img))  # Dummy ground truth

    # Load agent
    agent = load_agent(env, model_path)
    if agent is None:
        raise ValueError("Failed to load agent")

    # Process image
    state = env.reset()
    action = agent.act(state)
    processed_image, _, _, _ = env.step(action)

    inverted_result = cv2.bitwise_not(processed_image)

    return inverted_result


def load_agent(env, model_path):
    agent = EdgeDetectionAgent(env)
    checkpoint = torch.load(model_path, map_location=agent.device)
    agent.model.load_state_dict(checkpoint["model_state"])
    print(f"\nAgent loaded successfully from: {model_path}")
    return agent
