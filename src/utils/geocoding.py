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

    lat1, lon1, lat2, lon2 = map(lambda x: tf.cast(x, tf.float32) * math.pi / 180, [lat1, lon1, lat2, lon2])  # Convert to radians

    dlat = lat2 - lat1
    dlon = lon2 - lon1

    a = tf.math.sin(dlat / 2) ** 2 + tf.math.cos(lat1) * tf.math.cos(lat2) * tf.math.sin(dlon / 2) ** 2
    c = 2 * tf.math.atan2(tf.math.sqrt(a), tf.math.sqrt(1 - a))

    return 6371 * c

def calculate_accuracy(predictions, true_locations, threshold_km=1):
    num_samples = len(true_locations)
    num_correct = 0
    for i in range(num_samples):
        y_true = np.array([true_locations[i]])
        y_pred = np.array([predictions[i]])
        distance = haversine_distance(y_true, y_pred)
        if distance <= threshold_km:
            num_correct += 1
    accuracy = num_correct / num_samples
    return accuracy

def custom_grid_loss(y_true, y_pred, grid_size, alpha=0.1):
    # Standard categorical cross-entropy loss
    ce_loss = tf.keras.losses.CategoricalCrossentropy()(y_true, y_pred)
    
    # Calculate grid centers for true and predicted grids
    y_true_centers = tf.argmax(y_true, axis=-1)
    y_pred_centers = tf.argmax(y_pred, axis=-1)

    y_true_row = tf.math.floordiv(y_true_centers, grid_size)
    y_true_col = tf.math.mod(y_true_centers, grid_size)
    y_pred_row = tf.math.floordiv(y_pred_centers, grid_size)
    y_pred_col = tf.math.mod(y_pred_centers, grid_size)

    # Calculate the distance between true and predicted grid centers as the city block distance
    # row_diff = tf.square(tf.cast(y_true_row, tf.float32) - tf.cast(y_pred_row, tf.float32))
    # col_diff = tf.square(tf.cast(y_true_col, tf.float32) - tf.cast(y_pred_col, tf.float32))
    # distance = tf.sqrt(row_diff + col_diff)
    distance = tf.math.abs(y_true_row - y_pred_row) + tf.math.abs(y_true_col - y_pred_col)

    # Combine the categorical cross-entropy loss and the distance
    alpha = tf.constant(alpha, dtype=tf.float32)
    one = tf.constant(1, dtype=tf.float32)
    distance = tf.cast(distance, tf.float32)  # Cast the distance tensor to float32
    combined_loss = (one - alpha) * ce_loss + alpha * tf.reduce_mean(distance)


    return combined_loss