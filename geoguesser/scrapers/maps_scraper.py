import json
import os
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import requests
from google_streetview.api import results as google_results  # type: ignore
from loguru import logger
from pydantic import BaseModel

from geoguesser.config import settings

IMAGES_PATH_FROM_SCRAPER = "../../data/scraped_images"
IMAGES_PATH_FROM_ROOT = "data/scraped_images"


# Create the directory for the scraped images
script_dir = Path(__file__).parent.absolute()
Path.mkdir(Path(script_dir) / IMAGES_PATH_FROM_SCRAPER, exist_ok=True)
metadata_file = Path(script_dir) / f"{IMAGES_PATH_FROM_SCRAPER}/metadata.json"


class BoundingBox(BaseModel):
    lat_min: float
    lat_max: float
    lng_min: float
    lng_max: float


class Metadata(BaseModel):
    filepath: str
    grid_row: int
    grid_col: int
    grid_label: int
    lat: float
    lng: float
    filename: str
    heading: str


class MetadataManager:
    def __init__(self, metadata_file: Path):
        self.metadata_file = metadata_file
        self.metadata_lock = threading.Lock()
        self.load_metadata()

    def load_metadata(self):
        if Path.is_file(self.metadata_file):
            with Path.open(self.metadata_file) as f:
                self.metadata_json = json.load(f)
                self.counter = self.metadata_json["counter"]
                self.grid_image_counter = {
                    row: dict(enumerate(row_dict.values()))
                    for row, row_dict in enumerate(self.metadata_json["grid_image_counter"].values())
                }
        else:
            self.metadata_json = {"counter": 0, "metadata": [], "grid_image_counter": {}}
            self.counter = 0
            self.grid_image_counter = {}

    def save_metadata(self, metadata: dict):
        with self.metadata_lock:
            self.counter += 1
            self.metadata_json["metadata"].append(metadata)
            self.metadata_json["counter"] = self.counter
            grid_row, grid_col = metadata["grid_row"], metadata["grid_col"]
            self.metadata_json["grid_image_counter"][grid_row][grid_col] += 1
            with Path.open(self.metadata_file, "w") as f:
                json.dump(self.metadata_json, f)

    def reset_metadata(self, grid_size: int):
        self.counter = 0
        self.metadata_json = {
            "counter": 0,
            "metadata": [],
            "grid_image_counter": {i: dict.fromkeys(range(grid_size), 0) for i in range(grid_size)},
        }


def get_image_and_metadata(
    image_shape: tuple[int, int],
    lat: float,
    lng: float,
    heading: str,
    identifier: int,
    grid_row: int,
    grid_col: int,
    grid_size: int,
    grid_label: int,
) -> dict | None:
    params = [
        {
            "size": f"{image_shape[0]}x{image_shape[1]}",
            "location": f"{lat},{lng}",
            "heading": heading,
            "pitch": "0",
            "key": settings.google_api_key,
        }
    ]

    metadata_api_url = f"https://maps.googleapis.com/maps/api/streetview/metadata?location={lat},{lng}&heading={heading}&key={settings.google_api_key}"
    metadata_response = requests.get(metadata_api_url, timeout=10)
    metadata = metadata_response.json()

    if metadata["status"] == "OK":
        results = google_results(params)
        image_url = results.links[0]
        image_data = requests.get(image_url, timeout=10).content

        folder_path = Path(IMAGES_PATH_FROM_ROOT) / f"{grid_row}_{grid_col}/{lat:.6f}_{lng:.6f}"
        Path.mkdir(folder_path, exist_ok=True)

        filename = f"{heading}_{identifier}.png"
        with Path.open(Path(folder_path) / filename, "wb") as f:
            f.write(image_data)

        metadata.update({
            "filepath": f"{grid_row}_{grid_col}/{lat:.6f}_{lng:.6f}/{filename}",
            "grid_row": grid_row,
            "grid_col": grid_col,
            "grid_label": grid_label,
            "lat": lat,
            "lng": lng,
            "filename": filename,
            "heading": heading,
        })

        return metadata

    return None


def scrape_grid(
    image_shape: tuple[int, int],
    grid_row: int,
    grid_col: int,
    grid_size: int,
    images_per_grid: int,
    end_time: float,
    lat_min: float,
    lat_max: float,
    lng_min: float,
    lng_max: float,
    metadata_manager: MetadataManager,
) -> None:
    headings = ["0", "90", "180"]
    lat_step = (lat_max - lat_min) / grid_size
    lng_step = (lng_max - lng_min) / grid_size

    while time.time() < end_time:
        grid_label = grid_row * grid_size + grid_col
        while (
            metadata_manager.grid_image_counter[grid_row][grid_col] < images_per_grid * len(headings)
            and time.time() < end_time
        ):
            lat = lat_min + grid_row * lat_step + np.random.uniform(0, lat_step)
            lng = lng_min + grid_col * lng_step + np.random.uniform(0, lng_step)

            for heading in headings:
                metadata = get_image_and_metadata(
                    image_shape,
                    lat,
                    lng,
                    heading,
                    metadata_manager.counter + 1,
                    grid_row,
                    grid_col,
                    grid_size,
                    grid_label,
                )

                if metadata is not None:
                    metadata_manager.save_metadata(metadata)
                    logger.info(
                        f"""Saved image and metadata for location: {lat}, {lng}, heading: {heading}
                        in grid: {grid_row},{grid_col} with label: {grid_label}"""
                    )
        logger.info(f"Grid {grid_row}, {grid_col} finished.")
        break


def scrape_images(
    grid_size: int,
    images_per_grid: int,
    image_shape: tuple[int, int],
    bounding_box: BoundingBox | None = None,
    location_name: str | None = None,
    timeout_minutes: int = 30,
    *,
    keep_current_images: bool = True,
) -> tuple[BoundingBox, int, int]:
    metadata_manager = MetadataManager(metadata_file)

    if not keep_current_images:
        metadata_manager.reset_metadata(grid_size)

    if bounding_box is not None:
        lat_min, lat_max, lng_min, lng_max = (
            bounding_box.lat_min,
            bounding_box.lat_max,
            bounding_box.lng_min,
            bounding_box.lng_max,
        )
    elif location_name is not None:
        geocoding_api_url = (
            f"https://maps.googleapis.com/maps/api/geocode/json?address={location_name}&key={settings.google_api_key}"
        )
        response = requests.get(geocoding_api_url, timeout=10)
        geocoding_data = response.json()
        bounds = geocoding_data["results"][0]["geometry"]["bounds"]

        lat_min = bounds["southwest"]["lat"]
        lat_max = bounds["northeast"]["lat"]
        lng_min = bounds["southwest"]["lng"]
        lng_max = bounds["northeast"]["lng"]
    else:
        raise ValueError("Either bounding_box or location_name must be specified")

    if not keep_current_images:
        metadata_manager.reset_metadata(grid_size)
        for root, dirs, files in os.walk(IMAGES_PATH_FROM_ROOT, topdown=False):
            for name in files:
                Path.unlink(Path(root) / name)
            for name in dirs:
                Path.rmdir(Path(root) / name)
    else:
        logger.info("Loading existing metadata")
        metadata_manager.load_metadata()

    end_time = time.time() + timeout_minutes * 60
    with ThreadPoolExecutor(max_workers=grid_size * grid_size) as executor:
        futures = [
            executor.submit(
                scrape_grid,
                image_shape,
                grid_row,
                grid_col,
                grid_size,
                images_per_grid,
                end_time,
                lat_min,
                lat_max,
                lng_min,
                lng_max,
                metadata_manager,
            )
            for grid_row in range(grid_size)
            for grid_col in range(grid_size)
        ]
        # Wait for all threads to finish
        for future in futures:
            future.result()

    # Save the metadata to a single JSON file
    with Path.open(metadata_file, "w") as f:
        json.dump(metadata_manager.metadata_json, f)

    return (
        BoundingBox(lat_min=lat_min, lat_max=lat_max, lng_min=lng_min, lng_max=lng_max),
        metadata_manager.counter,
        len(metadata_manager.metadata_json["grid_image_counter"]),
    )
