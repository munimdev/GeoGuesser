# GeoGuesser Project

## Overview

The GeoGuesser project is a machine learning-based geolocation prediction tool. This project leverages deep learning models to predict geographical locations from images, specifically using grid-based classification and regression techniques. The project involves scraping images, preprocessing them, training a grid classifier to predict grid cells, and a location regressor to predict precise latitudes and longitudes.

## Table of Contents

- [GeoGuesser Project](#geoguesser-project)
  - [Overview](#overview)
  - [Table of Contents](#table-of-contents)
  - [Features](#features)
  - [Installation](#installation)
  - [Usage](#usage)
  - [Project Structure](#project-structure)
  - [Model Training](#model-training)
  - [Evaluation](#evaluation)
  - [Contributing](#contributing)
  - [License](#license)

## Features

- **Image Scraping**: Scrape images from Google Street View based on specified geographical bounds.
- **Image Preprocessing**: Resize, normalize, and prepare images for model training.
- **Grid Classification**: Classify images into grid cells based on their geographical locations.
- **Location Regression**: Predict precise latitudes and longitudes within the classified grid cells.
- **Custom Loss Functions**: Use custom loss functions like haversine distance for training the models.
- **Model Evaluation**: Evaluate the models using accuracy, mean distance error, and median distance error.

## Installation

To install the GeoGuesser project, follow these steps:

1. **Clone the repository**:

   ```sh
   git clone https://github.com/munimdev/git
   cd geoguesser
   ```

2. **Install poetry with the `dotenv` plugin**:

   ```sh
    curl -sSL https://install.python-poetry.org | python3 -
    poetry self add poetry-dotenv-plugin
   ```

   ```sh
    poetry env use python3.12
    poetry install
   ```

3. **Set up the environment variables**:

   Create a `.env` file in the root directory and add the following environment variables:

   ```sh
   GOOGLE_MAPS_API_KEY=<YOUR_GOOGLE_MAPS_API_KEY>
   ```

4. **Run the project**:

   ```sh
    poetry run python -m geoguesser.main
   ```

## Usage

### Scraping Images

To scrape images from Google Street View:

```python
from scrapers.maps_scraper import scrape_images

scrape_images(
    grid_size=10,
    images_per_grid=3,
    image_shape=(640, 640),
    bounding_box=None,
    location_name="London",
    keep_current_images=True
)
```

### Preprocessing Images

To preprocess images for model training:

```python
from preprocessor.image_preprocessor import preprocess_images
from pathlib import Path

metadata_file = Path("data/scraped_images/metadata.json")
train_loader, validation_loader, test_loader, train_lat_lng_labels, validation_lat_lng_labels, test_lat_lng_labels = preprocess_images(
    metadata_file,
    output_shape=(224, 224),
    grid_size=10
)
```

### Training

To train the grid classification and location regression models:

```python
from models.cnn_geoguesser import train_geoguesser, create_location_regressor

# Train grid classifier
grid_classifier = train_geoguesser(
    train_loader,
    validation_loader,
    num_classes=100,
    input_shape=(224, 224, 3),
    grid_classifier_epochs=20
)

# Prepare data for location regressor
train_grid_predictions = grid_classifier.predict(train_loader)
validation_grid_predictions = grid_classifier.predict(validation_loader)

# Train location regressor
location_regressor = create_location_regressor(grid_size=100, learning_rate=0.001)
location_regressor.regressor.fit(
    train_grid_predictions,
    train_lat_lng_labels,
    epochs=50,
    validation_data=(validation_grid_predictions, validation_lat_lng_labels)
)
```

### Evaluation

To evaluate the grid classification and location regression models:

```python
from evaluation.evaluator import evaluate_geoguesser

evaluate_geoguesser(
    grid_classifier,
    location_regressor,
    test_loader,
    test_lat_lng_labels,
    grid_size=100
)
```

## Project Structure

The project structure is as follows:

```plaintext
geoguesser/
├── data/
│   ├── scraped_images/
│   ├── preprocessed_images/
│   └── models/
├── geoguesser/
│   ├── evaluation/
│   ├── models/
│   ├── preprocessor/
│   └── scrapers/
├── .env
├── main.py
├── README.md
└── pyproject.toml
```

## Model Training

The GeoGuesser project uses a grid-based classification and regression approach to predict geographical locations from images. The project involves training two models:

1. **Grid Classifier**: A convolutional neural network (CNN) that classifies images into grid cells based on their geographical locations. The grid classifier is trained using a custom loss function that combines cross-entropy loss and haversine distance loss.

2. **Location Regressor**: A fully connected neural network that predicts precise latitudes and longitudes within the classified grid cells. The location regressor is trained using mean squared error loss.

## Evaluation Criteria

The GeoGuesser project evaluates the grid classification and location regression models using the following metrics:

1. **Accuracy**: The percentage of correctly classified grid cells.
2. **Mean Distance Error**: The mean distance between the predicted and actual latitudes and longitudes.
3. **Median Distance Error**: The median distance between the predicted and actual latitudes and longitudes.

## Contributing

Contributions to the GeoGuesser project are welcome! To contribute, follow these steps:

1. **Fork the repository**.
2. **Create a new branch**:

   ```sh
   git checkout -b feature/my-feature
   ```

3. **Make your changes** and commit them:

   ```sh
    git commit -am "Add new feature"
   ```

4. **Push your branch**:

   ```sh
   git push origin feature/my-feature
   ```

5. **Submit a pull request**.

## License

The GeoGuesser project is licensed under the MIT License. See the [LICENSE](LICENSE) file for more information.
