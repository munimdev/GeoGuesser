import json
import os
import cv2
import numpy as np
from typing import Tuple
from tensorflow.keras.preprocessing.image import ImageDataGenerator

def preprocess_images(data_dir: str, metadata_file: str, output_shape: Tuple[int, int], grid_size: int, num_classes: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:    
    # Load metadata
    with open(metadata_file, 'r') as f:
        metadata = json.load(f)

    num_samples = len(metadata)
    image_data = np.zeros((num_samples, output_shape[1], output_shape[0], 3), dtype=np.float32)
    location_data = np.zeros((num_samples, 2), dtype=np.float32)
    # grid_labels = np.zeros((num_samples), dtype=int)
    grid_labels = np.zeros((num_samples, num_classes), dtype=int)


    for i, item in enumerate(metadata):
        # Load and preprocess image
        filename = item['filename']
        # grid_labels[i] = item['grid_label']
        row = int(item['grid_row'])
        col = int(item['grid_col'])
        grid_labels[i] = np.eye(num_classes)[row]*row+np.eye(num_classes)[col]*col
        image = cv2.imread(os.path.join(data_dir, filename))
        image = cv2.resize(image, output_shape)
        image = image.astype(np.float32) / 255.0
        # Store preprocessed image and location data
        image_data[i] = image
        location_data[i] = [item['location']['lat'], item['location']['lng']]

    return image_data, location_data, grid_labels