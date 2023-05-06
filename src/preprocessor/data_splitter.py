import json
import os
import random
import shutil
from typing import Tuple

def split_data(metadata_file: str, data_dir: str, train_dir: str, val_dir: str, test_dir: str,
               train_ratio: float = 0.7, val_ratio: float = 0.15, test_ratio: float = 0.15) -> Tuple[int, int, int]:

    # Load metadata
    with open(metadata_file, 'r') as f:
        metadata_json = json.load(f)

    # Shuffle the metadata
    random.shuffle(metadata_json['metadata'])

    # Calculate the number of samples for each set
    total_samples = len(metadata_json['metadata'])
    train_samples = int(total_samples * train_ratio)
    val_samples = int(total_samples * val_ratio)
    test_samples = total_samples - train_samples - val_samples

    # Create the output directories if they don't exist
    os.makedirs(os.path.join(train_dir, 'images'), exist_ok=True)
    os.makedirs(os.path.join(val_dir, 'images'), exist_ok=True)
    os.makedirs(os.path.join(test_dir, 'images'), exist_ok=True)

    # Define helper function to copy files and save metadata
    def copy_and_save_metadata(data, output_dir):
        metadata = []
        for item in data:
            filename = item['filename']
            shutil.copy(os.path.join(data_dir, filename), os.path.join(output_dir, 'images', filename))
            metadata.append(item)
        with open(os.path.join(output_dir, 'metadata.json'), 'w') as f:
            json.dump(metadata, f)

    # Split data and save to respective directories
    copy_and_save_metadata(metadata_json['metadata'][:train_samples], train_dir)
    copy_and_save_metadata(metadata_json['metadata'][train_samples:train_samples + val_samples], val_dir)
    copy_and_save_metadata(metadata_json['metadata'][train_samples + val_samples:], test_dir)

    return train_samples, val_samples, test_samples
