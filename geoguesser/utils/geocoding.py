import math

import torch
import torch.nn.functional as F  # noqa: N812


def lat_lng_to_grid_label(lat, lng, lat_min, lat_max, lng_min, lng_max, grid_size):
    lat_step = (lat_max - lat_min) / grid_size
    lng_step = (lng_max - lng_min) / grid_size

    grid_row = int((lat - lat_min) // lat_step)
    grid_col = int((lng - lng_min) // lng_step)

    grid_label = grid_row * grid_size + grid_col
    return grid_label


def grid_label_to_lat_lng(grid_label, lat_min, lat_max, lng_min, lng_max, grid_size):
    lat_step = (lat_max - lat_min) / grid_size
    lng_step = (lng_max - lng_min) / grid_size

    grid_row = grid_label // grid_size
    grid_col = grid_label % grid_size

    lat_center = lat_min + grid_row * lat_step + lat_step / 2
    lng_center = lng_min + grid_col * lng_step + lng_step / 2

    return lat_center, lng_center


def haversine_distance(y_true, y_pred):
    lat1, lon1 = torch.split(y_true, 1, dim=-1)
    lat2, lon2 = torch.split(y_pred, 1, dim=-1)

    lat1, lon1, lat2, lon2 = (x * math.pi / 180 for x in [lat1, lon1, lat2, lon2])  # Convert to radians

    dlat = lat2 - lat1
    dlon = lon2 - lon1

    a = torch.sin(dlat / 2) ** 2 + torch.cos(lat1) * torch.cos(lat2) * torch.sin(dlon / 2) ** 2
    c = 2 * torch.atan2(torch.sqrt(a), torch.sqrt(1 - a))

    return 6371 * c


def calculate_accuracy(predictions, true_locations, threshold_km=1):
    num_samples = len(true_locations)
    num_correct = 0
    for i in range(num_samples):
        y_true = torch.tensor([true_locations[i]])
        y_pred = torch.tensor([predictions[i]])
        distance = haversine_distance(y_true, y_pred)
        if distance <= threshold_km:
            num_correct += 1
    accuracy = num_correct / num_samples
    return accuracy


def custom_grid_loss(y_true, y_pred, grid_size, alpha=0.1):
    # Standard categorical cross-entropy loss
    ce_loss = F.cross_entropy(y_pred, y_true.argmax(dim=-1))

    # Calculate grid centers for true and predicted grids
    y_true_centers = y_true.argmax(dim=-1)
    y_pred_centers = y_pred.argmax(dim=-1)

    y_true_row = y_true_centers // grid_size
    y_true_col = y_true_centers % grid_size
    y_pred_row = y_pred_centers // grid_size
    y_pred_col = y_pred_centers % grid_size

    # Calculate the distance between true and predicted grid centers as the city block distance
    distance = torch.abs(y_true_row - y_pred_row) + torch.abs(y_true_col - y_pred_col)

    # Combine the categorical cross-entropy loss and the distance
    combined_loss = (1 - alpha) * ce_loss + alpha * distance.mean()

    return combined_loss


def mean_haversine_distance(y_true, y_pred):
    return torch.mean(haversine_distance(y_true, y_pred))


def median_haversine_distance(y_true, y_pred):
    return torch.median(haversine_distance(y_true, y_pred))
