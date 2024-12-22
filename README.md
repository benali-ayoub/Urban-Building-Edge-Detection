# Urban Building Edge Detection & Parameter Optimization with DQN

This project implements a Deep Q-Network (DQN) to optimize edge detection parameters for images using reinforcement learning. The model learns to select the best parameters for edge detection methods like Canny and Sobel by training on a dataset of preprocessed and ground truth images.

## Table of Contents

- [Project Overview](#project-overview)
- [Features](#features)
- [Installation](#installation)
- [Usage](#usage)
- [Training](#training)
- [Model Saving and Loading](#model-saving-and-loading)
- [Contributing](#contributing)
- [License](#license)

## Project Overview

The goal of this project is to develop a reinforcement learning model that can automatically determine the optimal parameters for edge detection algorithms. The model is trained using a DQN, which learns from a dataset of preprocessed images and their corresponding ground truth edge maps.

## Features

- **Reinforcement Learning**: Uses a DQN to learn optimal edge detection parameters.
- **Image Feature Extraction**: Extracts intensity, texture, and gradient features from images.
- **Flexible Edge Detection**: Supports both Canny and Sobel edge detection methods.
- **Model Persistence**: Save and load trained models for future use.

## Installation

1. **Clone the repository**:
   ```bash
   git clone https://github.com/benali-ayoub/Urban-Building-Edge-Detection.git
   cd image_segmentation_rl
   ```

2. **Install dependencies**:
   Ensure you have Python 3.7+ and install the required packages:
   ```bash
   pip install -r requirements.txt
   ```

3. **Prepare your dataset**:
   - Place preprocessed images in the `preprocessed/` directory.
   - Place ground truth images in the `ground_truth/` directory.

## Usage

1. **Train the model**:
   Run the training script to start training the DQN model.
   ```bash
   python train.py
   ```

2. **Use the trained model**:
   Load a saved model and use it to predict edge detection parameters for new images.
   ```python
   from model_saver import ModelSaver
   from edge_detection_agent import EdgeDetectionAgent

   model_saver = ModelSaver()
   agent, _, _ = model_saver.load_model('saved_models/your_model_name')
   image = cv2.imread('test_image.jpg', 0)
   params = use_model(agent, image)
   edges = cv2.Canny(image, *params)
   ```

## Training

- **Training Data**: Ensure your dataset is properly organized with matching filenames in the `preprocessed/` and `ground_truth/` directories.
- **Training Parameters**: Adjust the number of episodes, batch size, and other hyperparameters in the `train_model` function as needed.

## Model Saving and Loading

- **Save Model**: The model is automatically saved every 50 episodes during training.
- **Load Model**: Use the `ModelSaver` class to load a saved model for inference or continued training.

## Contributing

Contributions are welcome! Please fork the repository and submit a pull request for any improvements or bug fixes.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.
