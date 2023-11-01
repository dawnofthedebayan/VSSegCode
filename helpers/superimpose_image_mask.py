from PIL import Image, ImageDraw

def superimpose_masks_on_image(original_image_path, masks, output_directory, opacity=0.5):
    # Open the original image
    original_image = Image.open(original_image_path)

    # Create a copy of the original image
    result_image = original_image.copy()

    # Define colors for masks (red and green) with alpha (transparency)
    colors_with_alpha = [(255, 0, 0, int(255 * opacity)), (0, 255, 0, int(255 * opacity))]

    # Create a drawing context on the result image
    draw = ImageDraw.Draw(result_image)

    # Iterate through masks
    for i, mask_path in enumerate(masks):
        # Load the mask image
        mask = Image.open(mask_path).convert("L")  # Convert to grayscale
        
        # Apply opacity to the mask
        #mask.putalpha(int(255 * opacity))

        # Create an empty RGBA image with the same size as the mask
        colored_mask = Image.new("RGBA", mask.size, (0, 0, 0, 0))

        # Create a drawing context on the colored mask
        draw_colored = ImageDraw.Draw(colored_mask)

        # Iterate through each pixel in the mask
        for x in range(mask.width):
            for y in range(mask.height):
                # Get the pixel value at this location in the mask
                pixel_value = mask.getpixel((x, y))

                
                

                # Check if the pixel is not fully transparent
                if pixel_value > 0:
                    # Get the corresponding color with alpha
                    color = colors_with_alpha[i]

                    # Draw a colored pixel on the colored mask
                    draw_colored.point((x, y), fill=color)

        # Paste the colored mask onto the result image
        result_image.paste(colored_mask, (0, 0), colored_mask)

    # Create the output directory if it doesn't exist
    os.makedirs(output_directory, exist_ok=True)

    # Construct the output image path
    output_image_path = os.path.join(output_directory, os.path.basename(original_image_path))

    # Save the resulting image
    result_image.save(output_image_path)

    return output_image_path



import os

def process_images_in_directory(image_directory, mask_directory1, mask_directory2, output_directory, opacity=0.5):
    # Get a list of all image files in the image directory
    image_files = [f for f in os.listdir(image_directory) if f.endswith('.png')]

    # Iterate through each image file
    for image_file in image_files:
        # Construct the full path to the image file
        image_path = os.path.join(image_directory, image_file)

        # Construct the full path to the corresponding masks
        mask_file_base = os.path.splitext(image_file)[0]
        mask1_path = os.path.join(mask_directory1, mask_file_base + '.png')
        mask2_path = os.path.join(mask_directory2, mask_file_base + '.png')

        masks = []

        # Check if mask1 exists and add it to the list
        if os.path.exists(mask1_path):
            masks.append(mask1_path)

        # Check if mask2 exists and add it to the list
        if os.path.exists(mask2_path):
            masks.append(mask2_path)

        # Check if at least one mask exists
        if masks:
            # Superimpose the masks on the image and save the resulting image
            superimposed_image_path = superimpose_masks_on_image(image_path, masks, output_directory, opacity)

            # Print information about the processed image
            print(f"Processed image: {image_path}")
            print(f"Superimposed image saved to: {superimposed_image_path}")
        else:
            print(f"No masks found for image: {image_path}")

# Example usage:
image_directory = '/media/debayan/c7b64c90-ca4e-4192-8ed9-8fea1d005196/SchwanommaDS/UKE/labelled_ds/T2/'
mask_directory1 = '/media/debayan/c7b64c90-ca4e-4192-8ed9-8fea1d005196/SchwanommaDS/UKE/labelled_ds/T2_masks/SchwannomaBrain/'
mask_directory2 = '/media/debayan/c7b64c90-ca4e-4192-8ed9-8fea1d005196/SchwanommaDS/UKE/labelled_ds/T2_masks/SchwannomaCanal/'
output_directory = '/media/debayan/c7b64c90-ca4e-4192-8ed9-8fea1d005196/SchwanommaDS/UKE/labelled_ds/superimposed/T2/'
opacity = 0.5  # Opacity level (0.0 - 1.0)

process_images_in_directory(image_directory, mask_directory1, mask_directory2, output_directory, opacity)
