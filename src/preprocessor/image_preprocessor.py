import json
import os
import cv2
import numpy as np
from typing import Tuple

def preprocess_images(data_dir: str, metadata_file: str, output_shape: Tuple[int, int]) -> Tuple[np.ndarray, np.ndarray]:

    # Load metadata
    with open(metadata_file, 'r') as f:
        metadata = json.load(f)

    num_samples = len(metadata)
    image_data = np.zeros((num_samples, output_shape[0], output_shape[1], 3), dtype=np.float32)
    location_data = np.zeros((num_samples, 2), dtype=np.float32)

    for i, item in enumerate(metadata):
        # Load and preprocess image
        filename = item['filename']
        image = cv2.imread(os.path.join(data_dir, filename))
        image = cv2.resize(image, output_shape)
        image = image.astype(np.float32) / 255.0

        # Store preprocessed image and location data
        image_data[i] = image
        location_data[i] = [item['location']['lat'], item['location']['lng']]

    return image_data, location_data
