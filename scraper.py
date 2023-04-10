import os
import time
import requests
from flickrapi import FlickrAPI

# Replace with your own Flickr API key and secret
API_KEY = '995cefbf4866d62060072226c1d47147'
API_SECRET = 'fa036c20086fac72'

# Set your download folder path
DOWNLOAD_FOLDER = 'geotagged_images'

# Create the download folder if it doesn't exist
if not os.path.exists(DOWNLOAD_FOLDER):
    os.makedirs(DOWNLOAD_FOLDER)

# Initialize the Flickr API
flickr = FlickrAPI(API_KEY, API_SECRET, format='parsed-json')

def download_image(url, filename):
    response = requests.get(url, stream=True)
    with open(filename, 'wb') as f:
        for chunk in response.iter_content(chunk_size=8192):
            f.write(chunk)

# Download images from all across Pakistan without any tags
for page in range(1, 21):  # Download 20 pages of images
    search_params = {
        'bbox': '60.87,23.64,77.84,37.09',  # Bounding box for Pakistan
        'has_geo': 1,
        'extras': 'geo,url_o',
        'per_page': 100,
        'page': page
    }

    response = flickr.photos.search(**search_params)
    photos = response['photos']['photo']

    for photo in photos:
        if 'url_o' in photo:
            url = photo['url_o']
            filename = os.path.join(DOWNLOAD_FOLDER, f"{photo['id']}.jpg")
            download_image(url, filename)

            # Store geolocation data (e.g., in a dictionary or save to a file)
            latitude = photo['latitude']
            longitude = photo['longitude']
            print(f"Downloaded {filename}, Latitude: {latitude}, Longitude: {longitude}")

    time.sleep(5)  # Add a delay between requests to avoid overwhelming the API
