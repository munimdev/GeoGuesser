import os
from dotenv import load_dotenv
import google_streetview.api
import google_streetview.helpers

# Load the .env file
load_dotenv()

# Read the API key from the .env file
apiKey = os.getenv('GOOGLE_API')

params = [{
    'size': '640x640', # max 640x640 pixels
    'location': '43.08114194671841, -79.07801864925116',
    'heading': '180',
    'pitch': '0',
    'key': apiKey}]

results = google_streetview.api.results(params)
results.download_links('./scraped_images')