import os
import numpy as np
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.optimizers import Adam, SGD, RMSprop

def create_grid_classifier(num_classes, input_shape, learning_rate=0.001):
    base_model = ResNet50(weights='imagenet', include_top=False, input_shape=input_shape)
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(1024, activation='relu')(x)
    x = Dropout(0.5)(x)
    predictions = Dense(num_classes, activation='softmax')(x)

    model = Model(inputs=base_model.input, outputs=predictions)

    for layer in base_model.layers:
        layer.trainable = False

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

    model.compile(optimizer=Adam(learning_rate=learning_rate), loss='mse', metrics=['mae'])

    return model

def train_geoguesser(train_generator, validation_generator, num_classes, input_shape, grid_classifier_epochs=20):
    # Train the grid classifier
    grid_classifier = create_grid_classifier(num_classes, input_shape)
    grid_classifier.fit(train_generator, epochs=grid_classifier_epochs, validation_data=validation_generator)

    return grid_classifier