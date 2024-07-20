import json
import shutil
from pathlib import Path

from loguru import logger

metadata_file = Path("./scraped_images/metadata.json")


def update_metadata_and_filenames():
    """
    Searches for missing files and
    updates the metadata JSON file and
    the filenames of the images in the scraped_images folder.

    e.g. 1.png, 2.png, 4.png -> 1.png, 2.png, 3.png
    in the case that 3.png was deleted/missing.
    """
    # Load the metadata JSON file
    with Path.open(metadata_file, "r") as f:
        metadata_json = json.load(f)

    new_metadata_list = []
    new_counter = 0
    filename_map = {}

    # Iterate through the metadata and check if the image files exist
    for metadata in metadata_json["metadata"]:
        old_filename = metadata["filename"]
        old_filepath = Path(f"./scraped_images/{old_filename}")

        if Path.is_file(old_filepath):
            new_counter += 1
            new_filename = f"{new_counter}.png"
            new_filepath = f"./scraped_images/{new_filename}"
            shutil.move(old_filepath, new_filepath)
            metadata["filename"] = new_filename
            new_metadata_list.append(metadata)
            filename_map[old_filename] = new_filename

    # Update the metadata JSON file
    metadata_json["metadata"] = new_metadata_list
    metadata_json["counter"] = new_counter

    with Path.open(metadata_file, "w") as f:
        json.dump(metadata_json, f)

    logger.info("Updated metadata and filenames.")
    return filename_map


filename_map = update_metadata_and_filenames()
logger.info("Filename changes:")
for old_filename, new_filename in filename_map.items():
    logger.info(f"{old_filename} -> {new_filename}")
