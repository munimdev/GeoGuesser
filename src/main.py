import os
import numpy as np
from scrapers.maps_scraper import scraper
from utils.geocoding import haversine_distance, calculate_accuracy
from preprocessor.data_splitter import split_data
from preprocessor.image_preprocessor import preprocess_images
from models.cnn_geoguesser import create_geoguesser_model, train_geoguesser_model, tune_geoguesser_model
from tensorflow.keras.applications.resnet import preprocess_input

# Scrape images and metadata
total_images = 300
keep_current_images = True
scraper(total_images, keep_current_images)

# Split the data into train, validation, and test sets
metadata_file = 'data/scraped_images/metadata.json'
data_dir = 'data/scraped_images'
train_dir = 'data/train'
val_dir = 'data/val'
test_dir = 'data/test'
train_samples, val_samples, test_samples = split_data(metadata_file, data_dir, train_dir, val_dir, test_dir)

# Preprocess images and location data
output_shape = (600, 300)
train_images, train_locations = preprocess_images(train_dir, os.path.join(train_dir, 'metadata.json'), output_shape)
val_images, val_locations = preprocess_images(val_dir, os.path.join(val_dir, 'metadata.json'), output_shape)
test_images, test_locations = preprocess_images(test_dir, os.path.join(test_dir, 'metadata.json'), output_shape)

# Create the geoguesser model
model = create_geoguesser_model(output_shape)

# Tune and train the model
tuned_model = tune_geoguesser_model(model, train_images, train_locations, val_images, val_locations)

# Evaluate the model on the test set
test_loss, test_accuracy = tuned_model.evaluate(test_images, test_locations)
print(f'Test Loss: {test_loss}, Test Accuracy: {test_accuracy}')

# Get the model's predictions
predicted_locations = model.predict(test_images)

# Calculate distances between actual and predicted coordinates
distances = haversine_distance(test_locations[:, 0], test_locations[:, 1],
                               predicted_locations[:, 0], predicted_locations[:, 1])

# Calculate mean and median distance errors
mean_distance_error = np.mean(distances)
median_distance_error = np.median(distances)
print(f'Mean Distance Error: {mean_distance_error} km, Median Distance Error: {median_distance_error} km')

# After training the model
predictions = model.predict(test_images)
test_accuracy = calculate_accuracy(predictions, test_locations)
print(f'Test Accuracy: {test_accuracy}')
