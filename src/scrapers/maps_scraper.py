import os
import random
import requests
import json
from dotenv import load_dotenv
import google_streetview.api

PATH_FROM_SCRAPER = "../../data/scraped_images"
PATH_FROM_ROOT = "data/scraped_images"

load_dotenv()
apiKey = os.getenv('GOOGLE_API')

geocoding_api_url = f'https://maps.googleapis.com/maps/api/geocode/json?address=Islamabad,+Pakistan&key={apiKey}'
response = requests.get(geocoding_api_url)
geocoding_data = response.json()
bounds = geocoding_data['results'][0]['geometry']['bounds']

lat_min = bounds['southwest']['lat']
lat_max = bounds['northeast']['lat']
lng_min = bounds['southwest']['lng']
lng_max = bounds['northeast']['lng']

# Function to create a one-hot encoding for labels
def one_hot_encoding(n, size):
    return [1 if i == n else 0 for i in range(size)]

def get_image_and_metadata(lat, lng, heading, identifier, grid_row, grid_col, grid_size, grid_label):
    params = [{
        'size': '640x320',
        'location': f'{lat},{lng}',
        'heading': heading,
        'pitch': '0',
        'key': apiKey}]

    metadata_api_url = f'https://maps.googleapis.com/maps/api/streetview/metadata?size=640x640&location={lat},{lng}&heading={heading}&pitch=0&key={apiKey}&return_error_code=true'
    metadata_response = requests.get(metadata_api_url)
    metadata = metadata_response.json()

    if metadata['status'] == 'OK':
        results = google_streetview.api.results(params)
        image_url = results.links[0]
        image_data = requests.get(image_url).content
        
        folder_path = f'{PATH_FROM_ROOT}/{grid_row}_{grid_col}/{lat:.6f}_{lng:.6f}'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        filename = f'{heading}_{identifier}.png'
        with open(f'{folder_path}/{filename}', 'wb') as f:
            f.write(image_data)

        metadata['filepath'] = f'{grid_row}_{grid_col}/{lat:.6f}_{lng:.6f}/{filename}'
        metadata['grid_row'] = grid_row
        metadata['grid_col'] = grid_col
        metadata['grid_label'] = grid_label
        metadata['one_hot_label'] = one_hot_encoding(grid_label, grid_size * grid_size)
        metadata['lat'] = lat
        metadata['lng'] = lng
        metadata['filename'] = filename
        metadata['heading'] = heading
        return metadata
    else:
        return None

import pathlib

script_dir = pathlib.Path(__file__).parent.absolute()
os.makedirs(os.path.join(script_dir, PATH_FROM_SCRAPER), exist_ok=True)

metadata_file = os.path.join(script_dir, f'{PATH_FROM_SCRAPER}/metadata.json')

def scraper(grid_size, images_per_grid, keep_current_images=True):
    global counter
    global metadata_json
    global num_classes

    if os.path.isfile(metadata_file):
        with open(metadata_file, 'r') as f:
            metadata_json = json.load(f)
            counter = metadata_json['counter']
            num_classes = len(os.listdir(f'{PATH_FROM_ROOT}'))
            grid_label_counter = metadata_json['grid_label_counter']
    else:
        counter = 0
        metadata_json = {
            'counter': 0,
            'metadata': [],
            'lat_min': lat_min,
            'lat_max': lat_max,
            'lng_min': lng_min,
            'lng_max': lng_max,
            'grid_size': grid_size,
            'grid_label_counter': {i: 0 for i in range(grid_size * grid_size)}
        }
        grid_label_counter = metadata_json['grid_label_counter']

    if not keep_current_images:
        counter = 0
        metadata_json = {'counter': 0, 'metadata': []}
        num_classes = 0
        # clear the directory and the metadata file
        for root, dirs, files in os.walk(f'{PATH_FROM_ROOT}', topdown=False):
            for name in files:
                os.remove(os.path.join(root, name))
            for name in dirs:
                os.rmdir(os.path.join(root, name))
        
        if os.path.isfile(metadata_file):
            os.remove(metadata_file)

    headings = ['0', '90', '180']
    grid_label_counter = {}

    lat_step = (lat_max - lat_min) / grid_size
    lng_step = (lng_max - lng_min) / grid_size

    for grid_row in range(grid_size):
        for grid_col in range(grid_size):
            grid_label = grid_row * grid_size + grid_col
            grid_label_counter[grid_label] = 0

            while grid_label_counter[grid_label] < images_per_grid:
                lat = lat_min + grid_row * lat_step + random.uniform(0, lat_step)
                lng = lng_min + grid_col * lng_step + random.uniform(0, lng_step)

                for heading in headings:
                    metadata = get_image_and_metadata(lat, lng, heading, counter + 1, grid_row, grid_col, grid_size, grid_label)

                    if metadata is not None:
                        metadata_json['metadata'].append(metadata)
                        counter += 1
                        print(f'Saved image and metadata for location: {lat}, {lng}, heading: {heading} in grid: {grid_row}, {grid_col} with label: {grid_label}')

                grid_label_counter[grid_label] += 1

    # Save the metadata to a single JSON file
    metadata_json['counter'] = counter
    metadata_json['num_classes'] = len(os.listdir(f'{PATH_FROM_ROOT}'))
    metadata_json['grid_label_counter'] = grid_label_counter
    with open(metadata_file, 'w') as f:
        json.dump(metadata_json, f)

    return lat_min, lat_max, lng_min, lng_max, counter, num_classes