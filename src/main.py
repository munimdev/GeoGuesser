import os
import numpy as np
from scrapers.maps_scraper import scraper
from utils.geocoding import haversine_distance, calculate_accuracy
from preprocessor.data_splitter import split_data
from preprocessor.image_preprocessor import preprocess_images
from models.cnn_geoguesser import train_geoguesser_model, tune_geoguesser_model
from tensorflow.keras.applications.resnet import preprocess_input

# Scrape images and metadata
total_images = 300
keep_current_images = True
grid_size = 30
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

# Tune and train the model
num_grid_cells = grid_size * grid_size
tuned_model = tune_geoguesser_model(train_images, train_locations, val_images, val_locations, output_shape, num_grid_cells)

# Evaluate the model on the test set
test_loss, test_grid_accuracy, test_lat_lng_accuracy = tuned_model.evaluate(test_images, {'grid': test_locations[:, 2], 'lat_lng': test_locations[:, :2]})
print(f'Test Loss: {test_loss}, Test Grid Accuracy: {test_grid_accuracy}, Test Lat-Lng Accuracy: {test_lat_lng_accuracy}')

# Get the model's predictions
predicted_grid_labels, predicted_locations = tuned_model.predict(test_images)

# Calculate distances between actual and predicted coordinates
# print(test_locations, predicted_locations)
distances=[]
for index, _ in enumerate(test_locations):
  print(f"Test: {test_locations[index]}, Prediction: {predicted_locations[index]}")
  distances.append(haversine_distance(test_locations[index], predicted_locations[index]))
  print(f"Distance: {distances[index]}")

# Calculate mean and median distance errors
mean_distance_error = np.mean(distances)
median_distance_error = np.median(distances)
print(f'Mean Distance Error: {mean_distance_error} km, Median Distance Error: {median_distance_error} km')

# After training the model
predictions = tuned_model.predict(test_images)
test_accuracy = calculate_accuracy(predictions, test_locations)
print(f'Test Accuracy: {test_accuracy}')

# Save the model
model_save_path = 'saved_models'
os.makedirs(model_save_path, exist_ok=True)
model_file_name = f'geoguesser_model_{test_accuracy}'
tuned_model.save(os.path.join(model_save_path, model_file_name))
