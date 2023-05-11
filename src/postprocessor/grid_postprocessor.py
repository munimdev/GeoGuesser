import numpy as np

def grid_predictions_to_coordinates(grid_predictions: np.ndarray, min_lat: float, max_lat: float, min_lng: float, max_lng: float, grid_size: int) -> np.ndarray:
    lat_step = (max_lat - min_lat) / grid_size
    lng_step = (max_lng - min_lng) / grid_size

    lat_centers = np.linspace(min_lat + lat_step / 2, max_lat - lat_step / 2, grid_size)
    lng_centers = np.linspace(min_lng + lng_step / 2, max_lng - lng_step / 2, grid_size)

    # Convert grid predictions to row and column indices
    row_indices = np.argmax(grid_predictions[:, :grid_size], axis=1)
    col_indices = np.argmax(grid_predictions[:, grid_size:], axis=1)

    # Get the corresponding latitude and longitude centers for the row and column indices
    lat_centers = lat_centers[row_indices]
    lng_centers = lng_centers[col_indices]

    # Combine the latitude and longitude centers into a single numpy array
    coordinates = np.stack((lat_centers, lng_centers), axis=-1)

    return coordinates