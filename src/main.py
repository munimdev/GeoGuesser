import os
import json
import numpy as np
from scrapers.maps_scraper import scraper
from preprocessor.image_preprocessor import preprocess_images
from models.cnn_geoguesser import train_geoguesser, create_grid_classifier, create_location_regressor

# Scrape images
grid_size = 10
images_per_grid = 3
location_name = "London"
INPUT_SHAPE = (640, 640, 3)
OUTPUT_SHAPE = (224, 224, 3)
EPOCHS_CLASSIFIER = 20
EPOCHS_REGRESSION = 20
bounding_box = None  # Example: {'lat_min': 33.5675, 'lat_max': 33.8242, 'lng_min': 72.8245, 'lng_max': 73.2819}
keep_current_images = True

if bounding_box is not None:
    lat_min, lat_max, lng_min, lng_max = bounding_box.values()
    scraper(grid_size, images_per_grid, INPUT_SHAPE, keep_current_images, location_name=location_name, bounding_box=bounding_box)
else:
    scraper(grid_size, images_per_grid, INPUT_SHAPE, keep_current_images, location_name=location_name)

# Preprocess images
metadata_file = 'data/scraped_images/metadata.json'
train_generator, validation_generator, test_generator = preprocess_images(metadata_file, OUTPUT_SHAPE, grid_size)

# Train the models
grid_classifier, location_regressor = train_geoguesser(train_generator, validation_generator, grid_size*grid_size, OUTPUT_SHAPE,
                                                       EPOCHS_CLASSIFIER, EPOCHS_REGRESSION)

# Evaluate the models
test_grid_predictions = grid_classifier.predict(test_generator)
test_lat_lng_labels = np.array([entry['one_hot_label'] for entry in test_generator.labels])

# Location regression evaluation
test_location_predictions = location_regressor.predict(test_grid_predictions)
test_location_loss, test_location_mae = location_regressor.evaluate(test_grid_predictions, test_lat_lng_labels)

# Grid classification evaluation
test_grid_labels = np.argmax(test_lat_lng_labels, axis=1)
test_grid_accuracy = np.mean(np.argmax(test_grid_predictions, axis=1) == test_grid_labels)

print(f'Test grid classification accuracy: {test_grid_accuracy}')
print(f'Test location loss: {test_location_loss}, Test location MAE: {test_location_mae}')

# Save models
grid_classifier.save('models/grid_classifier.h5')
location_regressor.save('models/location_regressor.h5')