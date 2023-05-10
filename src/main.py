import os
import numpy as np
from scrapers.maps_scraper import scraper
from utils.geocoding import haversine_distance, calculate_accuracy
from preprocessor.data_splitter import split_data
from preprocessor.image_preprocessor import preprocess_images
from preprocessor.grid_preprocessor import one_hot_encode_grid_labels
from postprocessor.grid_postprocessor import grid_predictions_to_coordinates
from models.cnn_geoguesser import create_geoguesser_model, create_regression_model, train_geoguesser_model

# Constants
EPOCHS_FOR_REGRESSION = 100
BATCH_SIZE_FOR_REGRESSION = 32
EPOCHS_FOR_CLASSIFICATION = 5
BATCH_SIZE_FOR_CLASSIFICATION = 2

# Scrape images and metadata
total_images = 300
keep_current_images = True
num_classes = scraper(total_images, keep_current_images)['num_classes']

# Split the data into train, validation, and test sets
metadata_file = 'data/scraped_images/metadata.json'
data_dir = 'data/scraped_images'
train_dir = 'data/train'
val_dir = 'data/val'
test_dir = 'data/test'
train_samples, val_samples, test_samples = split_data(metadata_file, data_dir, train_dir, val_dir, test_dir)

# Preprocess images and location data
output_shape = (600, 300)
grid_size = 30
train_images, train_locations, train_grid_labels = preprocess_images(train_dir, os.path.join(train_dir, 'metadata.json'), output_shape, grid_size)
val_images, val_locations, val_grid_labels = preprocess_images(val_dir, os.path.join(val_dir, 'metadata.json'), output_shape, grid_size)
test_images, test_locations, test_grid_labels = preprocess_images(test_dir, os.path.join(test_dir, 'metadata.json'), output_shape, grid_size)

# Preprocess grid labels
num_grid_cells = grid_size * grid_size
train_grid_labels = one_hot_encode_grid_labels(train_grid_labels, num_grid_cells)
val_grid_labels = one_hot_encode_grid_labels(val_grid_labels, num_grid_cells)
test_grid_labels = one_hot_encode_grid_labels(test_grid_labels, num_grid_cells)

# Create the geoguesser model
classification_model = create_geoguesser_model(output_shape, grid_size, num_grid_cells)

# Train the classification model
train_geoguesser_model(classification_model, train_images, train_grid_labels, val_images, val_grid_labels, BATCH_SIZE_FOR_CLASSIFICATION, EPOCHS_FOR_CLASSIFICATION)

# Get the grid predictions
train_grid_predictions = classification_model.predict(train_images)
val_grid_predictions = classification_model.predict(val_images)

# Convert grid predictions to coordinates
train_predicted_coordinates = grid_predictions_to_coordinates(train_grid_predictions, grid_size)
val_predicted_coordinates = grid_predictions_to_coordinates(val_grid_predictions, grid_size)

# Create the regression model
regression_model = create_regression_model(train_predicted_coordinates.shape[1:])

# Train the regression model
train_geoguesser_model(regression_model, train_predicted_coordinates, train_locations, val_predicted_coordinates, val_locations, BATCH_SIZE_FOR_REGRESSION, EPOCHS_FOR_REGRESSION)

# Evaluate the model on the test set
test_grid_predictions = classification_model.predict(test_images)
test_predicted_coordinates = grid_predictions_to_coordinates(test_grid_predictions, grid_size)
test_loss, test_accuracy = regression_model.evaluate(test_predicted_coordinates, test_locations)
print(f'Test Loss: {test_loss}, Test Accuracy: {test_accuracy}')

# Get the model's predictions
predicted_locations = []
# predicted_locations = regression_model.predict(test_predicted_coordinates)
for index, _ in enumerate(test_predicted_coordinates):
    predicted_locations.append(regression_model.predict(np.array([test_predicted_coordinates[index]])))

# Calculate distances between actual and predicted coordinates
distances=[]
for index, _ in enumerate(test_locations):
    print(f"Test: {test_locations[index]}, Prediction: {predicted_locations[index]}")
    distances.append(haversine_distance(test_locations[index], predicted_locations[index]))

# Calculate mean and median distance errors
mean_distance_error = np.mean(distances)
median_distance_error = np.median(distances)
print(f'Mean Distance Error: {mean_distance_error} km, Median Distance Error: {median_distance_error} km')

# Save the models
models_dir = 'trained_models'
classification_model.save(os.path.join(models_dir, 'classification_model'))
regression_model.save(os.path.join(models_dir, 'regression_model'))

# Calculate accuracy
accuracy = calculate_accuracy(predicted_locations, test_locations, 5)
print(f'Accuracy: {accuracy}%')