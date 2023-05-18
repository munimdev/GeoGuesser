import tkinter as tk
from tkinter import filedialog
import customtkinter as ctk
import requests
from PIL import Image, ImageTk
from io import BytesIO
import os
from dotenv import load_dotenv
# suppress tensorflow warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf
from tensorflow import keras
import numpy as np
from utils.geocoding import haversine_distance, mean_haversine_distance, median_haversine_distance


# get api key from .env
load_dotenv()
API_KEY = os.getenv('GOOGLE_API')
# google maps api configurations
center = 'London'
zoom = 7
displaySize = (400, 400)
imgSize = '400x400'
modelImgSize = (224, 224, 3)
scrapeRadius = 500 # 500m
# load models
custom_objects = {'haversine_distance': haversine_distance, 'mean_haversine_distance': mean_haversine_distance, 'median_haversine_distance': median_haversine_distance}
grid_classifier = keras.models.load_model('models\grid_classifier.h5')
location_regressor = keras.models.load_model('models\location_regressor.h5', custom_objects=custom_objects)
# ctk config
ctk.set_appearance_mode("System")
ctk.set_default_color_theme("blue")


class App(ctk.CTk):
    def __init__(self):
        super().__init__()
        self.geometry("900x600")
        self.title('GeoGuessr - AI Project - Bisma - Faizaan - Munim')

        # create grid layout
        self.grid_rowconfigure((0,1,2), weight=1)
        self.grid_columnconfigure((0,1), weight=1)

        # create frames
        self.upload_frame = ctk.CTkFrame(self)
        self.upload_frame.grid(row=0, column=0, rowspan=2, padx=40, pady=20, sticky="nsew")

        self.map_frame = ctk.CTkFrame(self)
        self.map_frame.grid(row=0, column=1, rowspan=2, padx=20, pady=20, sticky="nsew")

        # create labels
        self.map_label = ctk.CTkLabel(self.map_frame, text="")
        self.map_label.pack()

        self.upload_label = ctk.CTkLabel(self.upload_frame, text="")
        self.upload_label.pack()

        # create buttons in a subframe with 2 columns
        self.buttons_frame = ctk.CTkFrame(self, fg_color='transparent')
        self.buttons_frame.grid(row=2, column=0)
        self.buttons_frame.grid_rowconfigure(3, weight=1)

        self.upload_button = ctk.CTkButton(self.buttons_frame, command=self.upload_image, text="Upload Image")
        self.upload_button.grid(row=0, column=0, pady=10)

        self.scrape_image_button = ctk.CTkButton(self.buttons_frame, command=self.scrape_image, text="Scrape Image")
        self.scrape_image_button.grid(row=2, column=0, pady=10)


        # create subframe under map_frame to display distance
        self.distance_frame = ctk.CTkFrame(self, fg_color='transparent')
        self.distance_frame.grid(row=2, column=1)

        self.distance_label = ctk.CTkLabel(self.distance_frame, text="Distance from actual location: N/A", font=ctk.CTkFont(size=15, weight="bold"))
        self.distance_label.pack()

    
    # predict the location of image
    def extract_coordinates(self, image_path):
        # preprocess the image, grid-classifier takes image of size (224, 224, 3)
        image = tf.keras.preprocessing.image.load_img(image_path, target_size=modelImgSize)
        input_arr = keras.preprocessing.image.img_to_array(image)
        input_arr = np.array([input_arr])  # Convert single image to a batch.
        input_arr = tf.keras.applications.resnet50.preprocess_input(input_arr)
        # predict the grid
        grid = grid_classifier.predict(input_arr, verbose=0)
        # predict the location
        location = location_regressor.predict(grid, verbose=0)
        
        return location[0]

    
    # get single pin map image
    def get_map_image(self, latitude, longitude):
        url = f'https://maps.googleapis.com/maps/api/staticmap?center={center}&zoom={zoom}&size={imgSize}&markers=color:red%7C{latitude},{longitude}&key={API_KEY}'
        # url = f'https://maps.googleapis.com/maps/api/staticmap?center={latitude},{longitude}&zoom={10}&size={imgSize}&markers=color:red%7C{latitude},{longitude}&key={API_KEY}'
        response = requests.get(url)
        
        return Image.open(BytesIO(response.content))


    # handle the "Upload Image" button click
    def upload_image(self):
        file_path = filedialog.askopenfilename(filetypes=[("Image files", "*.jpg *.jpeg *.png")])
        if file_path:
            # Display the uploaded image on the GUI
            uploaded_image = ctk.CTkImage(dark_image=Image.open(file_path), size=displaySize)
            self.upload_label.configure(image=uploaded_image)
            self.upload_frame.configure(bg_color='transparent', fg_color='transparent')

            coordinates = self.extract_coordinates(file_path)
            if coordinates.any():
                latitude, longitude = coordinates
                # map_image = get_map_image(latitude, longitude)
                map_image = ctk.CTkImage(dark_image=self.get_map_image(latitude, longitude), size=displaySize)
                self.map_label.configure(image=map_image)
                self.map_frame.configure(bg_color='transparent', fg_color='transparent')

                # update the distance label
                self.distance_label.configure(text="Distance from actual location: N/A")
            else:
                print("Unable to extract coordinates from the image.")

    # handle the "Scrape Image" button click
    def scrape_image(self):
        # Set the coordinates of the boundaries for the city of London
        bounds = (51.384940, 51.672343, -0.351468, 0.148271)

        # Generate random coordinates within the boundaries
        latitude = np.random.uniform(bounds[0], bounds[1])
        longitude = np.random.uniform(bounds[2], bounds[3])

        # Get the map image from Google Maps API
        url = f"https://maps.googleapis.com/maps/api/streetview?size={imgSize}&location={latitude},{longitude}&radius={scrapeRadius}&key={API_KEY}"

        response = requests.get(url)
        if response.status_code != 200:
            print("Unable to get the map image from Google Maps API.")
            return
        im = Image.open(BytesIO(response.content))
        # Save the map image to the disk, data\test_images folder and get the prediction
        path = f"data\\test_images\\{latitude}_{longitude}.png"
        im.save(path)

        # Display the map image on the GUI on the upload image label
        map_image = ctk.CTkImage(dark_image=Image.open(path), size=displaySize)
        self.upload_label.configure(image=map_image)
        self.upload_frame.configure(bg_color='transparent', fg_color='transparent')

        coordinates = self.extract_coordinates(path)
        if coordinates.any():
            predicted_latitude, predicted_longitude = coordinates
            
            # url = f'https://maps.googleapis.com/maps/api/staticmap?center={center}&zoom={zoom}&size={imgSize}&markers=color:blue%7C{latitude},{longitude}&markers=color:red%7C{predicted_latitude},{predicted_longitude}&key={API_KEY}'
            lat = (latitude + predicted_latitude) / 2
            long = (longitude + predicted_longitude) / 2

            url = f'https://maps.googleapis.com/maps/api/staticmap?center={lat},{long}&size={imgSize}&markers=color:blue%7C{latitude},{longitude}&markers=color:red%7C{predicted_latitude},{predicted_longitude}&key={API_KEY}'
            r = requests.get(url)
            if r.status_code == 200:
                map_image = ctk.CTkImage(dark_image=Image.open(BytesIO(r.content)), size=displaySize)
                self.map_label.configure(image=map_image)
                self.map_frame.configure(bg_color='transparent', fg_color='transparent')

                # update the distance label
                distance = haversine_distance((latitude, longitude), (predicted_latitude, predicted_longitude))
                self.distance_label.configure(text=f"Distance from actual location: {distance[0]:.2f} km")
        else:
            print("Unable to extract coordinates from the image.")    


if __name__ == "__main__":
    app = App()
    app.mainloop()