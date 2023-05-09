import numpy as np

def grid_predictions_to_coordinates(predictions: np.ndarray, grid_size: int) -> np.ndarray:
    # Convert predictions to grid labels
    grid_labels = np.argmax(predictions, axis=1)

    # Calculate latitudes and longitudes based on the grid labels
    latitudes = (grid_labels // grid_size) * (180 / grid_size) - 90
    longitudes = (grid_labels % grid_size)
    longitudes = (grid_labels % grid_size) * (360 / grid_size) - 180
    coordinates = np.column_stack((latitudes, longitudes))
    return coordinates