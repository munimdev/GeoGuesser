import os
import numpy as np
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.optimizers import Adam, SGD, RMSprop

def create_grid_classifier(num_classes, learning_rate=0.001):
    base_model = ResNet50(weights='imagenet', include_top=False, input_shape=(320, 640, 3))
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(1024, activation='relu')(x)
    x = Dropout(0.5)(x)
    predictions = Dense(num_classes, activation='softmax')(x)

    model = Model(inputs=base_model.input, outputs=predictions)

    for layer in base_model.layers:
        layer.trainable = False

    model.compile(optimizer=Adam(learning_rate=learning_rate), loss='categorical_crossentropy', metrics=['accuracy'])

    return model

def create_location_regressor(grid_size, learning_rate=0.001):
    model = Sequential([
        Dense(1024, activation='relu', input_shape=(grid_size * grid_size,)),
        Dropout(0.5),
        Dense(512, activation='relu'),
        Dropout(0.5),
        Dense(2, activation='linear')
    ])

    model.compile(optimizer=Adam(learning_rate=learning_rate), loss='mse', metrics=['mae'])

    return model

def train_geoguesser(train_generator, validation_generator, grid_classifier_epochs=20, location_regressor_epochs=20):
    num_classes = train_generator.num_classes

    # Train the grid classifier
    grid_classifier = create_grid_classifier(num_classes)
    grid_classifier.fit(train_generator, epochs=grid_classifier_epochs, validation_data=validation_generator)

    # Extract grid predictions
    train_grid_predictions = grid_classifier.predict(train_generator)
    val_grid_predictions = grid_classifier.predict(validation_generator)

    # Get true latitude and longitude labels for training and validation
    train_lat_lng_labels = np.array([entry['one_hot_label'] for entry in train_generator.labels])
    val_lat_lng_labels = np.array([entry['one_hot_label'] for entry in validation_generator.labels])

    # Train the location regressor
    location_regressor = create_location_regressor(grid_size=num_classes)
    location_regressor.fit(train_grid_predictions, train_lat_lng_labels,
                           epochs=location_regressor_epochs, validation_data=(val_grid_predictions, val_lat_lng_labels))

    return grid_classifier, location_regressor