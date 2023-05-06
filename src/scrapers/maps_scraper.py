import os
import random
import requests
import json
from dotenv import load_dotenv
import google_streetview.api

# Load the .env file
load_dotenv()

# Read the API key from the .env file
apiKey = os.getenv('GOOGLE_API')

# Get the coordinates of Islamabad using the Geocoding API
geocoding_api_url = f'https://maps.googleapis.com/maps/api/geocode/json?address=Islamabad,+Pakistan&key={apiKey}'
response = requests.get(geocoding_api_url)
geocoding_data = response.json()
bounds = geocoding_data['results'][0]['geometry']['bounds']

# Define the bounding box for Islamabad
lat_min = bounds['southwest']['lat']
lat_max = bounds['northeast']['lat']
lng_min = bounds['southwest']['lng']
lng_max = bounds['northeast']['lng']

# Divide the bounding box into a grid
grid_size = 10
lat_step = (lat_max - lat_min) / grid_size
lng_step = (lng_max - lng_min) / grid_size

def get_image_and_metadata(lat, lng, identifier):
    params = [{
        'size': '640x640',  # max 640x640 pixels
        'location': f'{lat},{lng}',
        'heading': '180',
        'pitch': '0',
        'key': apiKey}]

    # Check if a valid image exists at the location
    metadata_api_url = f'https://maps.googleapis.com/maps/api/streetview/metadata?size=640x640&location={lat},{lng}&heading=180&pitch=0&key={apiKey}&return_error_code=true'
    metadata_response = requests.get(metadata_api_url)
    metadata = metadata_response.json()

    if metadata['status'] == 'OK':
        # Save the image
        results = google_streetview.api.results(params)
        image_url = results.links[0]
        image_data = requests.get(image_url).content
        os.makedirs('./scraped_images', exist_ok=True)
        with open(f'./scraped_images/{identifier}.png', 'wb') as f:
            f.write(image_data)

        # Save the metadata
        metadata['filename'] = f'{identifier}.png'
        return metadata
    else:
        return None

# Read the highest counter value from the metadata file
metadata_file = './scraped_images/metadata.json'
if os.path.isfile(metadata_file):
    with open(metadata_file, 'r') as f:
        metadata_json = json.load(f)
        counter = metadata_json['counter']
else:
    counter = 0
    metadata_json = {'counter': 0, 'metadata': []}

# Download images and metadata
num_images = 100

def scraper():
    '''
    Downloads images and metadata for a given number of locations.
    '''
    while counter < num_images:
        grid_row = random.randint(0, grid_size - 1)
        grid_col = random.randint(0, grid_size - 1)

        lat = lat_min + grid_row * lat_step + random.uniform(0, lat_step)
        lng = lng_min + grid_col * lng_step + random.uniform(0, lng_step)
        metadata = get_image_and_metadata(lat, lng, counter + 1)

        if metadata is not None:
            metadata_json['metadata'].append(metadata)
            counter += 1
            print(f'Saved image and metadata for location: {lat}, {lng}')

            # Save the metadata to a single JSON file
            metadata_json['counter'] = counter
            with open(metadata_file, 'w') as f:
                json.dump(metadata_json, f)
