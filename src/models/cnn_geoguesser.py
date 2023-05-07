import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, Model
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.optimizers import Adam, RMSprop, Adagrad, Nadam
from sklearn.model_selection import ParameterSampler
from utils.geocoding import haversine_distance

def create_geoguesser_model(output_shape):
    base_model = ResNet50(weights='imagenet', include_top=False, input_shape=(output_shape[1], output_shape[0], 3))
    
    for layer in base_model.layers:
        layer.trainable = False

    x = base_model.output
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(256, activation='relu')(x)
    x = layers.Dropout(0.5)(x)
    predictions = layers.Dense(2, activation='linear')(x)

    model = Model(inputs=base_model.input, outputs=predictions)
    model.compile(optimizer=Adam(learning_rate=0.001), loss=haversine_distance, metrics=['mean_absolute_error'])

    return model

def train_geoguesser_model(model, train_images, train_locations, val_images, val_locations, epochs=100):
    model.fit(train_images, train_locations, validation_data=(val_images, val_locations), batch_size=8, epochs=epochs)

def tune_geoguesser_model(train_images, train_locations, val_images, val_locations, output_shape):
    base_model = ResNet50(weights='imagenet', include_top=False, input_shape=(output_shape[1], output_shape[0], 3))

    # Freeze the layers of the base model
    for layer in base_model.layers:
        layer.trainable = False

    # param_grid = {
    #     'dropout_rate': [0.3, 0.5, 0.7],
    #     'init': ['he_normal', 'glorot_normal'],
    #     'optimizer': [Adam, RMSprop, Adagrad, Nadam],
    #     'unfreeze_layers': [0, 5, 10],
    #     'hidden_layers': [2, 4, 6],
    #     'hidden_units': [128, 256, 512],
    # }
    
    param_grid = {
        'dropout_rate': [0.3],
        'init': ['he_normal', 'glorot_normal'],
        'optimizer': [Adam],
        'unfreeze_layers': [5],
        'hidden_layers': [5],
        'hidden_units': [1024],
    }

    param_combinations = ParameterSampler(param_grid, n_iter=10)
    
    best_val_loss = float('inf')
    best_model = None

    for params in param_combinations:
        tuned_model = create_tuned_geoguesser_model(base_model, params)
        train_geoguesser_model(tuned_model, train_images, train_locations, val_images, val_locations)
        
        val_loss, _ = tuned_model.evaluate(val_images, val_locations)
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model = tuned_model

    return best_model

def create_tuned_geoguesser_model(base_model, params):
    # Unfreeze layers
    for layer in base_model.layers[-params['unfreeze_layers']:]:
        layer.trainable = True

    x = base_model.output
    x = layers.GlobalAveragePooling2D()(x)
    
    # Add hidden layers
    for _ in range(params['hidden_layers']):
        x = layers.Dense(params['hidden_units'], activation='relu', kernel_initializer=params['init'])(x)
        x = layers.Dropout(params['dropout_rate'])(x)

    predictions = layers.Dense(2, activation='linear')(x)

    model = Model(inputs=base_model.input, outputs=predictions)
    model.compile(optimizer=params['optimizer'](learning_rate=0.001), loss=haversine_distance, metrics=['mean_absolute_error'])

    return model