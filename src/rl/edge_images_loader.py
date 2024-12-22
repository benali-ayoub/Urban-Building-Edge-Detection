import os
from pathlib import Path
import cv2
import numpy as np
from tqdm import tqdm

def load_image_pairs(preprocessed_dir, ground_truth_dir, img_extensions=('.png', '.jpg', '.jpeg')):
    """
    Load pairs of images from preprocessed and ground truth directories.
    
    Args:
        preprocessed_dir (str): Path to directory containing preprocessed images
        ground_truth_dir (str): Path to directory containing ground truth images
        img_extensions (tuple): Tuple of valid image extensions to load
    
    Returns:
        tuple: (preprocessed_images, ground_truth_images, file_names)
    """
    # Convert directories to Path objects
    prep_path = Path(preprocessed_dir)
    gt_path = Path(ground_truth_dir)
    
    # Verify directories exist
    if not prep_path.exists():
        raise FileNotFoundError(f"Preprocessed images directory not found: {preprocessed_dir}")
    if not gt_path.exists():
        raise FileNotFoundError(f"Ground truth images directory not found: {ground_truth_dir}")
    
    # Get list of image files
    prep_files = []
    for ext in img_extensions:
        prep_files.extend(list(prep_path.glob(f"*{ext}")))
    prep_files.sort()
    
    if not prep_files:
        raise ValueError(f"No images found in {preprocessed_dir}")
    
    preprocessed_images = []
    ground_truth_images = []
    valid_file_names = []
    
    print("Loading image pairs...")
    for prep_file in tqdm(prep_files):
        # Get corresponding ground truth file
        gt_file = gt_path / prep_file.name
        
        # Check if ground truth file exists
        if not gt_file.exists():
            print(f"Warning: No matching ground truth for {prep_file.name}")
            continue
        
        try:
            # Load preprocessed image
            prep_img = cv2.imread(str(prep_file), cv2.IMREAD_GRAYSCALE)
            if prep_img is None:
                print(f"Warning: Could not load {prep_file}")
                continue
                
            # Load ground truth image
            gt_img = cv2.imread(str(gt_file), cv2.IMREAD_GRAYSCALE)
            if gt_img is None:
                print(f"Warning: Could not load {gt_file}")
                continue
            
            # Verify images have same dimensions
            if prep_img.shape != gt_img.shape:
                print(f"Warning: Size mismatch for {prep_file.name}")
                continue
            
            # Add to lists
            preprocessed_images.append(prep_img)
            ground_truth_images.append(gt_img)
            valid_file_names.append(prep_file.name)
            
        except Exception as e:
            print(f"Error processing {prep_file.name}: {str(e)}")
            continue
    
    if not preprocessed_images:
        raise ValueError("No valid image pairs found")
    
    # Convert to numpy arrays
    preprocessed_images = np.array(preprocessed_images)
    ground_truth_images = np.array(ground_truth_images)
    
    print(f"Successfully loaded {len(preprocessed_images)} image pairs")
    print(f"Image shape: {preprocessed_images[0].shape}")
    
    return preprocessed_images, ground_truth_images, valid_file_names