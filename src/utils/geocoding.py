import numpy as np

def haversine_distance(lat1, lon1, lat2, lon2):
    R = 6371  # Earth radius in km

    # Convert latitude and longitude from decimal degrees to radians
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])

    # Haversine formula
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat / 2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2)**2
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))

    return R * c

def calculate_accuracy(predictions, true_locations, threshold_km=1):
    num_samples = predictions.shape[0]
    num_correct = 0
    for i in range(num_samples):
        lat1, lon1 = predictions[i]
        lat2, lon2 = true_locations[i]
        distance = haversine_distance(lat1, lon1, lat2, lon2)
        if distance <= threshold_km:
            num_correct += 1
    accuracy = num_correct / num_samples
    return accuracy