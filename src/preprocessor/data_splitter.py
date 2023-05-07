import json
import os
import random
import shutil
from typing import Tuple

def clear_directory(directory: str):
    if not os.path.exists(directory):
        os.makedirs(directory)

    for filename in os.listdir(directory):
        file_path = os.path.join(directory, filename)
        if os.path.isfile(file_path) or os.path.islink(file_path):
            os.unlink(file_path)
        elif os.path.isdir(file_path):
            shutil.rmtree(file_path)

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

    # Clear output directories and create them if they don't exist
    for output_dir in [train_dir, val_dir, test_dir]:
        clear_directory(output_dir)
        os.makedirs(output_dir, exist_ok=True)

    # Define helper function to copy files and save metadata
    def copy_and_save_metadata(data, output_dir):
        metadata = []
        for item in data:
            src_file_path = os.path.join(data_dir, item['filepath'])
            dst_file_path = os.path.join(output_dir, item['filename'])
            os.makedirs(os.path.dirname(dst_file_path), exist_ok=True)
            shutil.copy(src_file_path, dst_file_path)
            metadata.append(item)
        with open(os.path.join(output_dir, 'metadata.json'), 'w') as f:
            json.dump(metadata, f)

    # Split data and save to respective directories
    copy_and_save_metadata(metadata_json['metadata'][:train_samples], train_dir)
    copy_and_save_metadata(metadata_json['metadata'][train_samples:train_samples + val_samples], val_dir)
    copy_and_save_metadata(metadata_json['metadata'][train_samples + val_samples:], test_dir)

    return train_samples, val_samples, test_samples
