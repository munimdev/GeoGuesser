import json
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from sklearn.model_selection import train_test_split  # type: ignore
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms  # type: ignore

from geoguesser.preprocessor.grid_preprocessor import one_hot_encode_grid_labels


class ImageDataset(Dataset):
    def __init__(self, images: torch.Tensor, labels: torch.Tensor):
        self.images = images
        self.labels = labels

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = self.images[idx]
        label = self.labels[idx]
        return image, label


def preprocess_images(metadata_file: Path, output_shape: tuple, grid_size: int, test_size=0.15, validation_size=0.15):
    with Path.open(metadata_file, "r") as f:
        metadata_json = json.load(f)

    metadata = metadata_json["metadata"]
    image_dir = Path("data/scraped_images")

    # Define transformations
    transform = transforms.Compose([
        transforms.Resize(output_shape),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    # Load images and labels
    images: list[torch.Tensor] = []
    grid_labels: list[np.ndarray] = []
    lat_lng_labels: list[list[float]] = []

    for entry in metadata:
        img_path = image_dir / entry["filepath"]
        img = Image.open(img_path).convert("RGB")
        img = transform(img)

        images.append(transforms.ToTensor()(img))
        grid_labels.append(one_hot_encode_grid_labels(entry["grid_label"], grid_size * grid_size))
        lat_lng_labels.append([entry["lat"], entry["lng"]])

    images_tensor = torch.stack(images)
    grid_labels_tensor = torch.tensor(grid_labels, dtype=torch.float32)
    lat_lng_labels_tensor = torch.tensor(lat_lng_labels, dtype=torch.float32)

    # Split the dataset
    train_indices, test_indices, _, _ = train_test_split(
        range(len(images_tensor)),
        range(len(images_tensor)),
        test_size=test_size,
        stratify=np.argmax(grid_labels, axis=1),
        random_state=42,
    )

    train_images, test_images = images[train_indices], images[test_indices]
    train_grid_labels, test_grid_labels = grid_labels_tensor[train_indices], grid_labels_tensor[test_indices]
    train_lat_lng_labels, test_lat_lng_labels = (
        lat_lng_labels_tensor[train_indices],
        lat_lng_labels_tensor[test_indices],
    )

    train_indices, validation_indices, _, _ = train_test_split(
        range(len(train_images)),
        range(len(train_images)),
        test_size=validation_size,
        stratify=np.argmax(train_grid_labels, axis=1),
        random_state=42,
    )

    validation_images = train_images[validation_indices]
    validation_grid_labels = train_grid_labels[validation_indices]
    validation_lat_lng_labels = train_lat_lng_labels[validation_indices]

    train_images = train_images[train_indices]
    train_grid_labels = train_grid_labels[train_indices]
    train_lat_lng_labels = train_lat_lng_labels[train_indices]

    # Create datasets
    train_dataset = ImageDataset(train_images, train_grid_labels)
    validation_dataset = ImageDataset(validation_images, validation_grid_labels)
    test_dataset = ImageDataset(test_images, test_grid_labels)

    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    validation_loader = DataLoader(validation_dataset, batch_size=32, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

    return (
        train_loader,
        validation_loader,
        test_loader,
        train_lat_lng_labels,
        validation_lat_lng_labels,
        test_lat_lng_labels,
    )
