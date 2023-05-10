import numpy as np
import math
import tensorflow as tf
from tensorflow.keras import layers, Model
from tensorflow.keras.applications import ResNet50, VGG16
from tensorflow.keras.optimizers import Adam, RMSprop, Adagrad, Nadam
from sklearn.model_selection import ParameterSampler
from utils.geocoding import haversine_distance, custom_grid_loss

def create_geoguesser_model(output_shape, grid_size, num_grid_cells):
    base_model = ResNet50(weights='imagenet', include_top=False, input_shape=(output_shape[1], output_shape[0], 3))
    # base_model = VGG16(weights='imagenet', include_top=False, input_shape=(output_shape[1], output_shape[0], 3))

    # for layer in base_model.layers[:-4]:
    #     layer.trainable = False
    model = tf.keras.Sequential()
    model.add(base_model)
    model.add(layers.GlobalAveragePooling2D())
    model.add(layers.Dense(num_grid_cells, activation='relu'))
    model.add(layers.Dropout(0.1))
    model.add(layers.Dense(grid_size, activation = 'sigmoid'))
    model.add(layers.Dense(256, activation='softmax'))

    # model.compile(optimizer=Adam(learning_rate=0.001), loss=['binary_crossentropy'], metrics=['mean_absolute_error'])
    # model.compile(optimizer=Adam(learning_rate=0.001), loss=['categorical_crossentropy'], metrics=['accuracy'])
    model.compile(optimizer=Adam(learning_rate=0.001), loss=['categorical_crossentropy'], metrics=['accuracy'])
    # model.compile(optimizer=Adam(learning_rate=0.001), loss=haversine_distance, metrics=['mean_absolute_error'])
    # model.compile(optimizer=Adam(learning_rate=0.001), loss=lambda y_true, y_pred: custom_grid_loss(y_true, y_pred, math.floor(math.sqrt(num_grid_cells)), 0.1) , metrics=['accuracy'])

    return model

def train_geoguesser_model(model, train_images, train_grid_labels, val_images, val_grid_labels, batch_size=8, epochs=50):
    model.fit(train_images, train_grid_labels, validation_data=(val_images, val_grid_labels), batch_size=batch_size, epochs=epochs)

def tune_geoguesser_model(train_images, train_grid_labels, val_images, val_grid_labels, output_shape, epochs=50):
    base_model = ResNet50(weights='imagenet', include_top=False, input_shape=(output_shape[1], output_shape[0], 3))

    # Freeze the layers of the base model
    for layer in base_model.layers:
        layer.trainable = False
    
    param_grid = {
        'dropout_rate': [0.2],
        'init': ['glorot_normal'],
        'optimizer': [Adam],
        'unfreeze_layers': [5],
        'hidden_layers': [3],
        'hidden_units': [128],
    }

    param_combinations = ParameterSampler(param_grid, n_iter=10)
    
    best_val_loss = float('inf')
    best_model = None

    for params in param_combinations:
        print(f"Training models with params: {params}")
        tuned_model = create_tuned_geoguesser_model(base_model, params)
        train_geoguesser_model(tuned_model, train_images, train_grid_labels, val_images, val_grid_labels, epochs)
        
        val_loss, _ = tuned_model.evaluate(val_images, val_grid_labels)
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

def create_regression_model(input_shape):
    model = tf.keras.Sequential([
        layers.Input(shape=input_shape),
        layers.Dense(64, activation='relu'),
        layers.Dense(32, activation='relu'),
        layers.Dense(2, activation='linear')
    ])

    # model.compile(optimizer=Adam(learning_rate=0.001), loss=haversine_distance, metrics=['mean_absolute_error'])
    model.compile(optimizer=Adam(learning_rate=0.001), loss=haversine_distance, metrics=['mean_absolute_error'])
    return model
