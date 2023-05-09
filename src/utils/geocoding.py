import numpy as np
import tensorflow as tf
import math

def lat_lng_to_grid_label(lat, lng, lat_min, lat_max, lng_min, lng_max, grid_size):
    lat_step = (lat_max - lat_min) / grid_size
    lng_step = (lng_max - lng_min) / grid_size

    grid_row = int((lat - lat_min) // lat_step)
    grid_col = int((lng - lng_min) // lng_step)

    grid_label = grid_row * grid_size + grid_col
    return grid_label

def grid_label_to_lat_lng(grid_label, lat_min, lat_max, lng_min, lng_max, grid_size):
    lat_step = (lat_max - lat_min) / grid_size
    lng_step = (lng_max - lng_min) / grid_size

    grid_row = grid_label // grid_size
    grid_col = grid_label % grid_size

    lat_center = lat_min + grid_row * lat_step + lat_step / 2
    lng_center = lng_min + grid_col * lng_step + lng_step / 2

    return lat_center, lng_center

def haversine_distance(y_true, y_pred):
    lat1, lon1 = tf.split(y_true, 2, axis=-1)
    lat2, lon2 = tf.split(y_pred, 2, axis=-1)

    lat1, lon1, lat2, lon2 = map(lambda x: x * math.pi / 180, [lat1, lon1, lat2, lon2])  # Convert to radians

    dlat = lat2 - lat1
    dlon = lon2 - lon1

    a = tf.math.sin(dlat / 2) ** 2 + tf.math.cos(lat1) * tf.math.cos(lat2) * tf.math.sin(dlon / 2) ** 2
    c = 2 * tf.math.atan2(tf.math.sqrt(a), tf.math.sqrt(1 - a))

    return 6371 * c

def calculate_accuracy(predictions, true_locations, threshold_km=1):
    num_samples = predictions.shape[0]
    num_correct = 0
    for i in range(num_samples):
        y_true = np.array([true_locations[i]])
        y_pred = np.array([predictions[i]])
        distance = haversine_distance(y_true, y_pred)
        if distance <= threshold_km:
            num_correct += 1
    accuracy = num_correct / num_samples
    return accuracy