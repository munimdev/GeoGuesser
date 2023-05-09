import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, Model
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.optimizers import Adam, RMSprop, Adagrad, Nadam
from sklearn.model_selection import ParameterSampler
from utils.geocoding import haversine_distance

def create_classifier(base_model, grid_size):
    x = base_model.output
    x = layers.GlobalAveragePooling2D()(x)
    classifier = layers.Dense(grid_size * grid_size, activation='softmax', name='grid_classifier')(x)
    return Model(inputs=base_model.input, outputs=classifier)

def create_regressor(base_model, params):
    x = base_model.output
    x = layers.GlobalAveragePooling2D()(x)

    for _ in range(params['hidden_layers']):
        x = layers.Dense(params['hidden_units'], activation='relu', kernel_initializer=params['init'])(x)
        x = layers.Dropout(params['dropout_rate'])(x)

    predictions = layers.Dense(2, activation='linear', name='location_regressor')(x)
    return Model(inputs=base_model.input, outputs=predictions)

def train_geoguesser_model(model, train_images, train_locations, val_images, val_locations, epochs=50):
    train_labels = {'grid': train_locations[:, 2], 'lat_lng': train_locations[:, :2]}
    val_labels = {'grid': val_locations[:, 2], 'lat_lng': val_locations[:, :2]}
    model.fit(train_images, train_labels, validation_data=(val_images, val_labels), batch_size=32, epochs=epochs)


def tune_geoguesser_model(train_images, train_locations, val_images, val_locations, output_shape, grid_size):
    base_model = ResNet50(weights='imagenet', include_top=False, input_shape=(output_shape[1], output_shape[0], 3))

    # Freeze the layers of the base model
    # for layer in base_model.layers:
    #     layer.trainable = False

    param_grid = {
        'dropout_rate': [0.1],
        'init': ['glorot_normal'],
        'optimizer': [Adam],
        'unfreeze_layers': [5],
        'hidden_layers': [1],
        'hidden_units': [128],
    }

    param_combinations = ParameterSampler(param_grid, n_iter=10)
    
    best_val_loss = float('inf')
    best_model = None

    # Create and train grid classifier
    grid_classifier = create_classifier(base_model, grid_size)
    grid_labels = np.argmax(np.reshape(train_locations[:, 2], (-1, 1)), axis=1)
    grid_classifier.compile(optimizer=Adam(learning_rate=0.001), loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    grid_classifier.fit(train_images, grid_labels, validation_split=0.2, batch_size=32, epochs=50)

    # Create and train location regressor
    for params in param_combinations:
        print(f"Training models with params: {params}")
        location_regressor = create_regressor(base_model, params)
        location_regressor.compile(optimizer=params['optimizer'](learning_rate=0.001), loss=haversine_distance, metrics=['mean_absolute_error'])
        train_geoguesser_model(location_regressor, train_images, train_locations[:, :2], val_images, val_locations[:, :2])

        val_loss, _ = location_regressor.evaluate(val_images, val_locations[:, :2])
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model = location_regressor
        return best_model, grid_classifier

def create_tuned_geoguesser_model(base_model, params, num_grid_cells):
    # Unfreeze layers
    for layer in base_model.layers[-params['unfreeze_layers']:]:
        layer.trainable = True

    x = base_model.output
    x = layers.GlobalAveragePooling2D()(x)

    # Add hidden layers for grid classification
    for _ in range(params['hidden_layers']):
        x = layers.Dense(params['hidden_units'], activation='relu', kernel_initializer=params['init'])(x)
        x = layers.Dropout(params['dropout_rate'])(x)

    # Grid classification
    grid_predictions = layers.Dense(num_grid_cells, activation='softmax', name='grid')(x)

    # Regression for latitude and longitude
    lat_lng_predictions = layers.Dense(2, activation='linear', name='lat_lng')(x)

    model = Model(inputs=base_model.input, outputs=[grid_predictions, lat_lng_predictions])
    model.compile(optimizer=params['optimizer'](learning_rate=0.001),
                  loss={'grid': 'sparse_categorical_crossentropy', 'lat_lng': haversine_distance},
                  metrics={'grid': 'accuracy', 'lat_lng': 'mean_absolute_error'})

    return model