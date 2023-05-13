import json
import os
import cv2
import numpy as np
from typing import Tuple
from tensorflow.keras.preprocessing.image import ImageDataGenerator

def preprocess_images(data_dir: str, metadata_file: str, output_shape: Tuple[int, int], grid_size: int, total_images: int) -> Tuple[ImageDataGenerator, ImageDataGenerator, ImageDataGenerator]:
    # Load metadata
    with open(metadata_file, 'r') as f:
        metadata = json.load(f)

    image_data = np.zeros((total_images, output_shape[1], output_shape[0], 3), dtype=np.float32)
    location_data = np.zeros((total_images, 2), dtype=np.float32)
    grid_labels = np.zeros((total_images, grid_size*grid_size), dtype=int)

    for i, item in enumerate(metadata):
        # Load and preprocess image
        filename = item['filename']
        image = cv2.imread(os.path.join(data_dir, filename))
        image = cv2.resize(image, (output_shape[1], output_shape[0]))
        image = image.astype(np.float32) / 255.0
        # Store preprocessed image and location data
        image_data[i] = image
        location_data[i] = [item['location']['lat'], item['location']['lng']]
        grid_labels[i] = np.eye(grid_size*grid_size)[item['grid_label']]

    # return image_data, location_data, grid_labels
    return image_data, location_data, grid_labels