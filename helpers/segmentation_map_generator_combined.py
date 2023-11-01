"""
Code to generate segmentation maps from the contour points
"""


import json
import os
from PIL import Image, ImageDraw
import numpy as np

def generate_segmentation_masks(json_file_path, output_folder):
    # Load JSON data from the file
    with open(json_file_path, 'r') as json_file:
        data = json.load(json_file)

    # Get image dimensions
    image_height = data['imageHeight']
    image_width = data['imageWidth']

    # Create a dictionary to store masks for each class
    class_masks = {}

    
    mask_images = [] 
    # Iterate through the shapes in the JSON data
    for shape in data['shapes']:
        label = shape['label']

        
        
        # Check if the label is in the list of classes you want to create masks for
        if label in ['SchwannomaCanal', 'SchwannomaBrain']:
            # Get the polygon points
            polygon_points = shape['points']

            if len(polygon_points) < 3:
                # Skip this shape if it doesn't have at least 3 points
                continue

            # Convert polygon points to integer tuples
            polygon_points = [(int(x), int(y)) for x, y in polygon_points]

            # Create a blank black image with the same dimensions
            mask_image = Image.new('L', (image_width, image_height), 0)

            # Draw the polygon on the mask image with a white color (255)
            draw = ImageDraw.Draw(mask_image)
            draw.polygon(polygon_points, fill=255)

          
            #convert mask to np array

            mask_image_np = np.array(mask_image) 
            mask_images.append(mask_image_np)


    # creata a larger mask image with all the masks superimposed

    mask_image = np.zeros((image_height, image_width), dtype=np.uint8) 
    for mask in mask_images:

        mask_image = mask_image + mask

    # remove all values greater than 255
    mask_image[mask_image > 255] = 255 

    mask_image = Image.fromarray(mask_image)

    # Save the mask image in the class folder with the same name as the original image

    image_name = os.path.basename(data['imagePath'])
    #create directory if it doesn't exist
    os.makedirs(os.path.join(output_folder,'combined'), exist_ok=True) 
    mask_image_path = os.path.join(output_folder,'combined', image_name)
    mask_image.save(mask_image_path) 

    
    return class_masks



def process_json_files_in_directory(input_directory, output_directory):
    # Iterate through all files in the input directory
    for filename in os.listdir(input_directory):
        if filename.endswith(".json"):
            # Construct the full path to the JSON file
            json_file_path = os.path.join(input_directory, filename)

            # Generate segmentation masks and obtain class masks
            class_masks = generate_segmentation_masks(json_file_path, output_directory)

            # Print information about the processed JSON file
            print(f"Processed JSON file: {json_file_path}")

# Example usage:
input_directory = '/media/debayan/c7b64c90-ca4e-4192-8ed9-8fea1d005196/SchwanommaDS/UKE/labelled_ds/T2/'
output_directory = '/media/debayan/c7b64c90-ca4e-4192-8ed9-8fea1d005196/SchwanommaDS/UKE/labelled_ds/T2_masks/'

process_json_files_in_directory(input_directory, output_directory)