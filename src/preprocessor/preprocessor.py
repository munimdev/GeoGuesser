import json
import random
import shutil
from pathlib import Path
from sklearn.model_selection import train_test_split
from PIL import Image

def preprocess_image(image_path, output_path, size=(224, 224)):
    img = Image.open(image_path)
    img_resized = img.resize(size)
    img_resized.save(output_path)

metadata_file = './scraped_images/metadata.json'
with open(metadata_file, 'r') as f:
    metadata_json = json.load(f)

image_metadata = metadata_json['metadata']
train_metadata, test_metadata = train_test_split(image_metadata, test_size=0.2, random_state=42)
train_metadata, val_metadata = train_test_split(train_metadata, test_size=0.25, random_state=42)

for split, metadata_split in [('train', train_metadata), ('val', val_metadata), ('test', test_metadata)]:
    for metadata in metadata_split:
        filename = metadata['filename']
        input_path = f'./scraped_images/{filename}'
        output_dir = f'./preprocessed_images/{split}'
        output_path = f'{output_dir}/{filename}'
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        preprocess_image(input_path, output_path, size=(640, 640)) # Change the target size to 640x640

    # Save split metadata to separate JSON files
    split_metadata_file = f'./preprocessed_images/{split}_metadata.json'
    with open(split_metadata_file, 'w') as f:
        json.dump(metadata_split, f)
