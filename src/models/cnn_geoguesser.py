import tensorflow as tf
from tensorflow.keras import layers, Model
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping

def haversine_distance(y_true, y_pred):
    lat1, lon1 = tf.split(y_true, 2, axis=-1)
    lat2, lon2 = tf.split(y_pred, 2, axis=-1)

    lat1_rad = tf.math.radians(lat1)
    lon1_rad = tf.math.radians(lon1)
    lat2_rad = tf.math.radians(lat2)
    lon2_rad = tf.math.radians(lon2)

    dlat = lat2_rad - lat1_rad
    dlon = lon2_rad - lon1_rad

    a = tf.math.square(tf.math.sin(dlat / 2)) + tf.math.cos(lat1_rad) * tf.math.cos(lat2_rad) * tf.math.square(tf.math.sin(dlon / 2))
    c = 2 * tf.math.atan2(tf.math.sqrt(a), tf.math.sqrt(1 - a))

    # Earth radius in kilometers
    earth_radius = 6371.0
    distance = earth_radius * c

    return distance

def create_geoguesser_model(input_shape):
    base_model = ResNet50(weights='imagenet', include_top=False, input_shape=input_shape + (3,))

    # Freeze the base model weights
    for layer in base_model.layers:
        layer.trainable = False

    x = base_model.output
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(256, activation='relu')(x)
    x = layers.Dropout(0.5)(x)
    predictions = layers.Dense(2, activation='linear')(x)

    model = Model(inputs=base_model.input, outputs=predictions)
    return model

def train_geoguesser_model(model, train_images, train_locations, val_images, val_locations):
    optimizer = Adam(learning_rate=0.001)
    loss = haversine_distance
    metric = 'mean_absolute_error'

    model.compile(optimizer=optimizer, loss=loss, metrics=[metric])

    early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

    model.fit(train_images, train_locations,
              validation_data=(val_images, val_locations),
              epochs=50, batch_size=32,
              callbacks=[early_stopping])
