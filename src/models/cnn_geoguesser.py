import os
import numpy as np
from tensorflow.keras.applications import ResNet50, EfficientNetB0
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense, Dropout, BatchNormalization
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.optimizers import Adam, SGD, RMSprop
from utils.geocoding import haversine_distance, mean_haversine_distance, median_haversine_distance

def create_grid_classifier(num_classes, input_shape, learning_rate=0.001):
    base_model = ResNet50(weights='imagenet', include_top=False, input_shape=input_shape)
    # base_model = EfficientNetB0(weights='imagenet', include_top=False, input_shape=input_shape)
    
    for layer in base_model.layers:
        layer.trainable = False

    # Unfreeze the last few layers
    for layer in base_model.layers[-15:]:
        if not isinstance(layer, BatchNormalization):
            layer.trainable = True

    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(1024, activation='relu')(x)
    x = Dropout(0.5)(x)  # Increased dropout
    predictions = Dense(num_classes, activation='softmax')(x)

    model = Model(inputs=base_model.input, outputs=predictions)

    # model.compile(optimizer=Adam(learning_rate=learning_rate), loss='categorical_crossentropy', metrics=['accuracy'])
    model.compile(optimizer=Adam(learning_rate=learning_rate), loss='categorical_crossentropy', metrics=['categorical_accuracy'])

    return model

def create_location_regressor(grid_size, learning_rate=0.001):
    model = Sequential([
        Dense(1024, activation='relu', input_shape=(grid_size,)),
        Dropout(0.5),
        Dense(512, activation='relu'),
        Dropout(0.5),
        Dense(2, activation='linear')
    ])

    # model.compile(optimizer=Adam(learning_rate=learning_rate), loss='mse', metrics=['mae'])
    model.compile(loss=haversine_distance, optimizer=Adam(learning_rate=learning_rate), metrics=[mean_haversine_distance, median_haversine_distance])

    return model

def train_geoguesser(train_generator, validation_generator, num_classes, input_shape, grid_classifier_epochs=20):
    # Train the grid classifier
    grid_classifier = create_grid_classifier(num_classes, input_shape)
    grid_classifier.fit(train_generator, epochs=grid_classifier_epochs, validation_data=validation_generator)

    return grid_classifier