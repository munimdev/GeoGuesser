import os
import random
import requests
import json
from dotenv import load_dotenv
import google_streetview.api
from concurrent.futures import ThreadPoolExecutor
import threading
import pathlib
import time

PATH_FROM_SCRAPER = "../../data/scraped_images"
PATH_FROM_ROOT = "data/scraped_images"

load_dotenv()
apiKey = os.getenv('GOOGLE_API')

# def one_hot_encoding(n, size):
#     return [1 if i == n else 0 for i in range(size)]

def get_image_and_metadata(image_shape, lat, lng, heading, identifier, grid_row, grid_col, grid_size, grid_label):
    params = [{
        'size': f'{image_shape[0]}x{image_shape[1]}',
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
        metadata['grid_row'] = int(grid_row)
        metadata['grid_col'] = int(grid_col)
        metadata['grid_label'] = grid_label
        # metadata['one_hot_label'] = one_hot_encoding(grid_label, grid_size * grid_size)
        metadata['lat'] = lat
        metadata['lng'] = lng
        metadata['filename'] = filename
        metadata['heading'] = heading
        return metadata
    else:
        return None

metadata_lock = threading.Lock()
# available_grids_lock = threading.Lock()

def save_metadata(metadata):
    global metadata_json
    global counter

    with metadata_lock:
        counter += 1
        metadata_json['metadata'].append(metadata)
        metadata_json['counter'] += 1
        metadata_json['grid_image_counter'][metadata['grid_row']][metadata['grid_col']] += 1
        with open(metadata_file, 'w') as f:
            json.dump(metadata_json, f)


# def get_next_available_grid(available_grids):
#     with available_grids_lock:
#         if len(available_grids) > 0:
#             return available_grids.pop(0)
#         else:
#             return None

# def scrape_grid(grid_row, grid_col, grid_size, images_per_grid, grid_image_counter, end_time, available_grids):
def scrape_grid(image_shape, grid_row, grid_col, grid_size, images_per_grid, grid_image_counter, end_time):
    global counter
    headings = ['0', '90', '180']
    lat_step = (lat_max - lat_min) / grid_size
    lng_step = (lng_max - lng_min) / grid_size

    while time.time() < end_time:
        grid_label = grid_row * grid_size + grid_col
        # print(grid_image_counter)
        # print(grid_image_counter.keys())
        # while grid_image_counter[f'{grid_row}'][f'{grid_col}'] < images_per_grid*len(headings) and time.time() < end_time:
        while grid_image_counter[grid_row][grid_col] < images_per_grid*len(headings) and time.time() < end_time:
            lat = lat_min + grid_row * lat_step + random.uniform(0, lat_step)
            lng = lng_min + grid_col * lng_step + random.uniform(0, lng_step)

            for heading in headings:
                metadata = get_image_and_metadata(image_shape, lat, lng, heading, counter + 1, grid_row, grid_col, grid_size, grid_label)

                if metadata is not None:
                    save_metadata(metadata)
                    print(f'Saved image and metadata for location: {lat}, {lng}, heading: {heading} in grid: {grid_row}, {grid_col} with label: {grid_label}')

        # # Get the next available grid
        # next_grid = get_next_available_grid(available_grids)
        # if next_grid is None:
        #     break
        # else:
        #     print(f'Grid {grid_row}, {grid_col} finished. Next grid: {next_grid}')
        #     grid_row, grid_col = next_grid
        print(f'Grid {grid_row}, {grid_col} finished.')
        break


script_dir = pathlib.Path(__file__).parent.absolute()
os.makedirs(os.path.join(script_dir, PATH_FROM_SCRAPER), exist_ok=True)

metadata_file = os.path.join(script_dir, f'{PATH_FROM_SCRAPER}/metadata.json')

def scraper(grid_size, images_per_grid, image_shape, keep_current_images=True, bounding_box=None, location_name=None, timeout_minutes=30):
    global counter
    global metadata_json
    global num_classes
    global lat_min, lat_max, lng_min, lng_max

    if bounding_box is not None:
        lat_min, lat_max, lng_min, lng_max = bounding_box
    elif location_name is not None:
        geocoding_api_url = f'https://maps.googleapis.com/maps/api/geocode/json?address={location_name}&key={apiKey}'
        response = requests.get(geocoding_api_url)
        geocoding_data = response.json()
        bounds = geocoding_data['results'][0]['geometry']['bounds']

        lat_min = bounds['southwest']['lat']
        lat_max = bounds['northeast']['lat']
        lng_min = bounds['southwest']['lng']
        lng_max = bounds['northeast']['lng']
    else:
        raise ValueError('Either bounding_box or location_name must be specified')

    if not keep_current_images:
        counter = 0
        metadata_json = {'counter': 0, 'metadata': []}
        num_classes = 0
        grid_image_counter = {i: {j: 0 for j in range(grid_size)} for i in range(grid_size)}
        # clear the directory and the metadata file
        for root, dirs, files in os.walk(f'{PATH_FROM_ROOT}', topdown=False):
            for name in files:
                os.remove(os.path.join(root, name))
            for name in dirs:
                os.rmdir(os.path.join(root, name))

        if os.path.isfile(metadata_file):
            os.remove(metadata_file)
    elif os.path.isfile(metadata_file) and keep_current_images:
        print('Loading existing metadata')
        with open(metadata_file, 'r') as f:
            metadata_json = json.load(f)
            counter = metadata_json['counter']
            num_classes = len(os.listdir(f'{PATH_FROM_ROOT}'))-1
            # grid_image_counter = metadata_json['grid_image_counter']
            grid_image_counter = {row: {col: x for col, x in enumerate(row_dict.values())} for row, row_dict in enumerate(metadata_json['grid_image_counter'].values())}
            metadata_json['grid_image_counter'] = grid_image_counter
    else:
        counter = 0
        num_classes = 0
        grid_image_counter = {i: {j: 0 for j in range(grid_size)} for i in range(grid_size)}
        metadata_json = {
            'counter': 0,
            'metadata': [],
            'lat_min': lat_min,
            'lat_max': lat_max,
            'lng_min': lng_min,
            'lng_max': lng_max,
            'grid_size': grid_size,
            'grid_image_counter': grid_image_counter
        }
    end_time = time.time() + timeout_minutes * 60
    # available_grids = [(i, j) for i in range(grid_size) for j in range(grid_size)][1:]
    with ThreadPoolExecutor(max_workers=grid_size * grid_size) as executor:
        futures = []
        for grid_row in range(grid_size):
            for grid_col in range(grid_size):
                futures.append(executor.submit(scrape_grid, image_shape, grid_row, grid_col, grid_size, images_per_grid, grid_image_counter, end_time))
        # Wait for all threads to finish
        for future in futures:
            future.result()

    # Save the metadata to a single JSON file
    metadata_json['counter'] = counter
    metadata_json['num_classes'] = len(os.listdir(f'{PATH_FROM_ROOT}'))-1
    metadata_json['grid_image_counter'] = grid_image_counter
    with open(metadata_file, 'w') as f:
        json.dump(metadata_json, f)

    return lat_min, lat_max, lng_min, lng_max, counter, num_classes