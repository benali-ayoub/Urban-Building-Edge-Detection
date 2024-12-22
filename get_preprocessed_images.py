import os
import cv2
import numpy as np
from src.building_detector import BuildingEdgeDetector
from src.rl.trainer import get_optimal_parameters

def load_image(image_path):
    return cv2.imread(image_path, cv2.IMREAD_COLOR)

def save_image(image, output_path):
    cv2.imwrite(output_path, image)

def process_folder(input_folder, output_folder, agent_path, num_samples=5):
    os.makedirs(output_folder, exist_ok=True)

    for file_name in os.listdir(input_folder):
        file_path = os.path.join(input_folder, file_name)

        if not (file_name.lower().endswith(('.png', '.jpg', '.jpeg'))):
            continue

        image = load_image(file_path)

        optimal_params = get_optimal_parameters(
            agent_path=agent_path,
            image=image,
            num_samples=num_samples,
        )

        detector = BuildingEdgeDetector(optimal_params=optimal_params)
        _, processed = detector.detect(image)

        output_path = os.path.join(output_folder, file_name)
        save_image(processed, output_path)

if __name__ == "__main__":
    input_folder = "images/images_inter"
    output_folder = "images/preprocessed_images"
    agent_path = "src/models/preprocessing_agent2.pth"  # Replace with your model path

    process_folder(input_folder, output_folder, agent_path, num_samples=5)
