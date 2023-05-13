import os
import json
import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.preprocessing.image import load_img, img_to_array

def preprocess_images(metadata_file, output_shape=(224, 224), test_size=0.1, validation_size=0.1):
    with open(metadata_file, 'r') as f:
        metadata_json = json.load(f)

    metadata = metadata_json['metadata']
    image_dir = 'data/scraped_images'

    # Load images and labels
    images = []
    grid_labels = []
    lat_lng_labels = []

    for entry in metadata:
        img_path = os.path.join(image_dir, entry['filepath'])
        img = load_img(img_path, target_size=output_shape)
        img_array = img_to_array(img)

        images.append(img_array)
        grid_labels.append(entry['grid_label'])
        lat_lng_labels.append(entry['one_hot_label'])

    images = np.array(images)
    grid_labels = np.array(grid_labels)
    lat_lng_labels = np.array(lat_lng_labels)

    # Split the dataset
    train_indices, test_indices, _, _ = train_test_split(
        range(len(images)), range(len(images)), test_size=test_size, stratify=grid_labels, random_state=42)

    train_images, test_images = images[train_indices], images[test_indices]
    train_grid_labels, test_grid_labels = grid_labels[train_indices], grid_labels[test_indices]
    train_lat_lng_labels, test_lat_lng_labels = lat_lng_labels[train_indices], lat_lng_labels[test_indices]

    train_indices, validation_indices, _, _ = train_test_split(
        range(len(train_images)), range(len(train_images)), test_size=validation_size, stratify=train_grid_labels, random_state=42)

    validation_images = train_images[validation_indices]
    validation_grid_labels = train_grid_labels[validation_indices]
    validation_lat_lng_labels = train_lat_lng_labels[validation_indices]

    train_images = train_images[train_indices]
    train_grid_labels = train_grid_labels[train_indices]
    train_lat_lng_labels = train_lat_lng_labels[train_indices]

    # Image data generators
    train_datagen = ImageDataGenerator(rescale=1./255)
    validation_datagen = ImageDataGenerator(rescale=1./255)
    test_datagen = ImageDataGenerator(rescale=1./255)

    train_generator = train_datagen.flow(train_images, train_lat_lng_labels, batch_size=32)
    validation_generator = validation_datagen.flow(validation_images, validation_lat_lng_labels, batch_size=32)
    test_generator = test_datagen.flow(test_images, test_lat_lng_labels, batch_size=1, shuffle=False)

    return train_generator, validation_generator, test_generator