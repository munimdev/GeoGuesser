from collections.abc import Callable

import torch
from loguru import logger
from pydantic import BaseModel
from pydantic_settings import SettingsConfigDict
from torch import nn, optim
from torch.utils.data import DataLoader
from torchvision import models  # type: ignore

from geoguesser.utils.geocoding import haversine_distance


class GridClassifier(nn.Module):
    def __init__(self, num_classes: int, input_shape: tuple[int, int, int], *, pretrained=True):
        super().__init__()
        self.base_model = models.resnet50(
            weights="imagenet", include_top=False, input_shape=input_shape, pretrained=pretrained
        )
        # self.base_model = models.efficientnet_b0(weights='imagenet', include_top=False, pretrained=pretrained)

        for param in self.base_model.parameters():
            param.requires_grad = False

        # Unfreeze the last few layers
        for param in list(self.base_model.parameters())[-15:]:
            param.requires_grad = True

        self.fc = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(self.base_model.fc.in_features, 1024),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(1024, num_classes),
            nn.Softmax(dim=1),
        )

    def forward(self, x):
        return self.base_model(x)


class LocationRegressor(nn.Module):
    def __init__(self, grid_size: int):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(grid_size, 1024),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, 2),
        )

    def forward(self, x):
        return self.fc(x)


class GridClassifierReturn(BaseModel):
    model_config = SettingsConfigDict(arbitrary_types_allowed=True)

    classifier: GridClassifier
    optimizer: optim.Optimizer
    criterion: nn.CrossEntropyLoss


def create_grid_classifier(
    num_classes: int, input_shape: tuple[int, int, int], learning_rate=0.001
) -> GridClassifierReturn:
    model = GridClassifier(num_classes, input_shape)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.CrossEntropyLoss()
    return GridClassifierReturn(classifier=model, optimizer=optimizer, criterion=criterion)


class LocationRegressorReturn(BaseModel):
    model_config = SettingsConfigDict(arbitrary_types_allowed=True)

    regressor: LocationRegressor
    optimizer: optim.Optimizer
    criterion: Callable


def create_location_regressor(grid_size: int, learning_rate=0.001) -> LocationRegressorReturn:
    model = LocationRegressor(grid_size)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    criterion = haversine_distance
    return LocationRegressorReturn(regressor=model, optimizer=optimizer, criterion=criterion)


def train_geoguesser(
    train_loader: DataLoader,
    validation_loader: DataLoader,
    num_classes: int,
    input_shape: tuple[int, int, int],
    grid_classifier_epochs=20,
) -> GridClassifier:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    grid_classifier = create_grid_classifier(num_classes, input_shape)
    grid_classifier.classifier.to(device)

    for epoch in range(grid_classifier_epochs):
        grid_classifier.classifier.train()
        running_loss = 0.0
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)  # noqa: PLW2901

            grid_classifier.optimizer.zero_grad()
            outputs = grid_classifier.classifier(inputs)
            loss = grid_classifier.criterion(outputs, labels)
            loss.backward()
            grid_classifier.optimizer.step()

            running_loss += loss.item() * inputs.size(0)

        epoch_loss = running_loss / len(train_loader)
        logger.info(f"Epoch {epoch + 1}/{grid_classifier_epochs}, Loss: {epoch_loss:.4f}")

        grid_classifier.classifier.eval()
        val_loss = 0.0
        with torch.no_grad():
            for inputs, labels in validation_loader:
                inputs, labels = inputs.to(device), labels.to(device)  # noqa: PLW2901
                outputs = grid_classifier.classifier(inputs)
                loss = grid_classifier.criterion(outputs, labels)
                val_loss += loss.item() * inputs.size(0)

        val_loss /= len(validation_loader)
        logger.info(f"Validation Loss: {val_loss:.4f}")

    return grid_classifier.classifier
