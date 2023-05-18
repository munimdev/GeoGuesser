import os
import json
import shutil

metadata_file = './scraped_images/metadata.json'

def update_metadata_and_filenames():
    '''
    Searches for missing files and
    updates the metadata JSON file and 
    the filenames of the images in the scraped_images folder.

    e.g. 1.png, 2.png, 4.png -> 1.png, 2.png, 3.png
    in the case that 3.png was deleted/missing.
    '''
    # Load the metadata JSON file
    with open(metadata_file, 'r') as f:
        metadata_json = json.load(f)

    new_metadata_list = []
    new_counter = 0
    filename_map = {}

    # Iterate through the metadata and check if the image files exist
    for metadata in metadata_json['metadata']:
        old_filename = metadata['filename']
        old_filepath = f'./scraped_images/{old_filename}'

        if os.path.isfile(old_filepath):
            new_counter += 1
            new_filename = f'{new_counter}.png'
            new_filepath = f'./scraped_images/{new_filename}'
            shutil.move(old_filepath, new_filepath)
            metadata['filename'] = new_filename
            new_metadata_list.append(metadata)
            filename_map[old_filename] = new_filename

    # Update the metadata JSON file
    metadata_json['metadata'] = new_metadata_list
    metadata_json['counter'] = new_counter

    with open(metadata_file, 'w') as f:
        json.dump(metadata_json, f)

    print("Updated metadata and filenames.")
    return filename_map

filename_map = update_metadata_and_filenames()
print("Filename changes:")
for old_filename, new_filename in filename_map.items():
    print(f"{old_filename} -> {new_filename}")
