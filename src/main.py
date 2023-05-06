from preprocessor.data_splitter import split_data
from preprocessor.image_preprocessor import preprocess_images
from models.cnn_geoguesser import create_cnn_geoguesser
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

metadata_file = './data/scraped_images/metadata.json'
data_dir = './data/scraped_images'
train_dir = './data/train'
val_dir = './data/val'
test_dir = './data/test'

train_samples, val_samples, test_samples = split_data(metadata_file, data_dir, train_dir, val_dir, test_dir)

print(f'Training samples: {train_samples}')
print(f'Validation samples: {val_samples}')
print(f'Test samples: {test_samples}')

output_shape = (224, 224)

train_images, train_locations = preprocess_images(train_dir, os.path.join(train_dir, 'metadata.json'), output_shape)
val_images, val_locations = preprocess_images(val_dir, os.path.join(val_dir, 'metadata.json'), output_shape)
test_images, test_locations = preprocess_images(test_dir, os.path.join(test_dir, 'metadata.json'), output_shape)

print(f'Training data shape: {train_images.shape}, {train_locations.shape}')
print(f'Validation data shape: {val_images.shape}, {val_locations.shape}')
print(f'Test data shape: {test_images.shape}, {test_locations.shape}')

# Create the model
model = create_cnn_geoguesser(input_shape=(224, 224, 3))
print(model.summary())

# Train the model
batch_size = 32
epochs = 50
early_stopping = EarlyStopping(patience=5, restore_best_weights=True)
model_checkpoint = ModelCheckpoint('best_model.h5', save_best_only=True)

history = model.fit(train_images, train_locations, batch_size=batch_size, epochs=epochs,
                    validation_data=(val_images, val_locations),
                    callbacks=[early_stopping, model_checkpoint])

# Evaluate the model
test_loss, test_mae = model.evaluate(test_images, test_locations)
print(f'Test loss: {test_loss}, Test Mean Absolute Error: {test_mae}')
