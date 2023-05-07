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

grid_size = 30
lat_step = (lat_max - lat_min) / grid_size
lng_step = (lng_max - lng_min) / grid_size

def get_image_and_metadata(lat, lng, heading, identifier, grid_row, grid_col):
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
        os.makedirs(folder_path, exist_ok=True)

        filename = f'{heading}_{identifier}.png'
        with open(f'{folder_path}/{filename}', 'wb') as f:
            f.write(image_data)

        metadata['filepath'] = f'{grid_row}_{grid_col}/{lat:.6f}_{lng:.6f}/{filename}'
        metadata['grid_row'] = grid_row
        metadata['grid_col'] = grid_col
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

if os.path.isfile(metadata_file):
    with open(metadata_file, 'r') as f:
        metadata_json = json.load(f)
        counter = metadata_json['counter']
else:
    counter = 0
    metadata_json = {'counter': 0, 'metadata': []}

def scraper(total_images, keep_current_images=True):
    global counter
    global metadata_json

    if not keep_current_images:
        counter = 0
        metadata_json = {'counter': 0, 'metadata': []}

    headings = ['0', '90', '180']

    while counter < total_images:
        grid_row = random.randint(0, grid_size - 1)
        grid_col = random.randint(0, grid_size - 1)

        lat = lat_min + grid_row * lat_step + random.uniform(0, lat_step)
        lng = lng_min + grid_col * lng_step + random.uniform(0, lng_step)

        for heading in headings:
            metadata = get_image_and_metadata(lat, lng, heading, counter + 1, grid_row, grid_col)

            if metadata is not None:
                metadata_json['metadata'].append(metadata)
                counter += 1
                print(f'Saved image and metadata for location: {lat}, {lng}, heading: {heading}')

                if counter >= total_images:
                    break

        # Save the metadata to a single JSON file
        metadata_json['counter'] = counter
        with open(metadata_file, 'w') as f:
            json.dump(metadata_json, f)