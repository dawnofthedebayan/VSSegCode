"""
Code to generate segmentation maps from the contour points
"""


import json
import os
from PIL import Image, ImageDraw
import numpy as np
import cv2
import matplotlib.pyplot as plt

def generate_segmentation_masks(json_file_path, output_folder,crop = False,width = -1,height = -1,input_image_path = None):
    # Load JSON data from the file
    with open(json_file_path, 'r') as json_file:
        data = json.load(json_file)

    # Get image dimensions
    image_height = data['imageHeight']
    image_width = data['imageWidth']

    # Create a dictionary to store masks for each class
    class_masks = {}

    
    mask_images = [] 
    mask_label = []
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

            if label == 'SchwannomaCanal':
                mask_label.append(1)
            else:
                mask_label.append(2)

          
            #convert mask to np array

            mask_image_np = np.array(mask_image) 
            mask_images.append(mask_image_np)


    # creata a larger mask image with all the masks superimposed

    mask_image = np.zeros((image_height, image_width), dtype=np.uint8) 
    mask_image_class = np.zeros((image_height, image_width), dtype=np.uint8)
    for mask,label in zip(mask_images,mask_label):

        mask_image = mask_image + mask
       
        mask_image_class[mask == 255] = label 

    
    # remove all values greater than 255
    mask_image[mask_image > 255] = 255 

    mask_image = Image.fromarray(mask_image)
    mask_image_class = Image.fromarray(mask_image_class)

    # Save the mask image in the class folder with the same name as the original image


    # divide the mask image into two parts by splitting it along the width with a 20 % overlap
    mask_classes_ = []
    # Special cases where the masks go beyond the midline of the image and hence cannot be split into two parts therefore we split them not along the midline but along the midline + 10% of the image width
    if "_16_" in data['imagePath'] or "_58_" in data['imagePath'] or "_58_" in data['imagePath']:
        if crop == True:    
            image = cv2.imread(input_image_path+os.path.basename(data['imagePath'])) 

            # split the image into two parts

            image = Image.fromarray(image)

            original_image1 = image.crop((0,0, image_width//2 + image_width//10, image_height)) 
            original_image2 = image.crop((image_width//2 + image_width//10,0, image_width, image_height)) 

        mask_image1 = mask_image.crop((0,0, image_width//2 + image_width//10, image_height))
        mask_image2 = mask_image.crop((image_width//2 + image_width//10,0, image_width, image_height))

        mask_image_class1 = mask_image_class.crop((0,0, image_width//2 + image_width//10, image_height))
        mask_image_class2 = mask_image_class.crop((image_width//2 + image_width//10,0, image_width, image_height))


    elif "_39_" in data['imagePath'] or "_80_" in data['imagePath']:
        

        if crop == True: 
            image = cv2.imread(input_image_path+os.path.basename(data['imagePath'])) 

            # split the image into two parts

            image = Image.fromarray(image)

            original_image1 = image.crop((0,0, image_width//2 - image_width//10, image_height))
            original_image2 = image.crop((image_width//2 - image_width//10,0, image_width, image_height))


        mask_image1 = mask_image.crop((0,0, image_width//2 - image_width//10, image_height))
        mask_image2 = mask_image.crop((image_width//2 - image_width//10,0, image_width, image_height))

        mask_image_class1 = mask_image_class.crop((0,0, image_width//2 - image_width//10, image_height))
        mask_image_class2 = mask_image_class.crop((image_width//2 - image_width//10,0, image_width, image_height))

        """
        for mask in mask_images:

            mask_image_1 = mask.crop((0,0, image_width//2 - image_width//10, image_height)) 
            mask_image_2 = mask.crop((image_width//2 - image_width//10,0, image_width, image_height))

            mask_image_1 = np.array(mask_image_1)
            mask_image_2 = np.array(mask_image_2)

            mask_classes_.append((mask_image_1,mask_image_2))
        """

    else:

        if crop == True:
            image = cv2.imread(input_image_path+os.path.basename(data['imagePath'])) 

            # split the image into two parts

            image = Image.fromarray(image)

            
            original_image1 = image.crop((0,0, image_width//2 , image_height))
            original_image2 = image.crop((image_width//2,0, image_width, image_height))


        mask_image1 = mask_image.crop((0,0, image_width//2 , image_height))
        mask_image2 = mask_image.crop((image_width//2,0, image_width, image_height))

        mask_image_class1 = mask_image_class.crop((0,0, image_width//2 , image_height))
        mask_image_class2 = mask_image_class.crop((image_width//2,0, image_width, image_height))


    
    # find rectangles around the masks
    mask_image1 = np.array(mask_image1)
    mask_image2 = np.array(mask_image2)
    mask_image_class1 = np.array(mask_image_class1)
    mask_image_class2 = np.array(mask_image_class2)

    # dilate the mask image
    kernel = np.ones((5,5),np.uint8)
    mask_image1 = cv2.dilate(mask_image1,kernel,iterations = 1)
    mask_image2 = cv2.dilate(mask_image2,kernel,iterations = 1)


    # fit a rectangle around the mask

    # find the contours
    contours1, hierarchy1 = cv2.findContours(mask_image1,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE) 
    contours2, hierarchy2 = cv2.findContours(mask_image2,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
    
    # check if countours are found
    areas = [] 
    rectangles = []

    if len(contours1) != 0:

        
        x,y,w,h = cv2.boundingRect(contours1[0])
        areas.append(w*h)
        rectangles.append((x,y,w,h))

        # crop the mask image and original image
        if crop == True:

            image = cv2.imread(input_image_path+os.path.basename(data['imagePath']))
            # find center of the rectangle
            center_x = x + w//2
            center_y = y + h//2

            #mask image width and height
            image_width = mask_image1.shape[1]
            image_height = mask_image1.shape[0]

            # crop the image around the center and make sure the cropped image is within the image boundaries
            if center_x - width//2 < 0:
                center_x = width//2

            if center_x + width//2 > image_width:
                center_x = image_width - width//2

            if center_y - height//2 < 0:
                center_y = height//2

            if center_y + height//2 > image_height:
                center_y = image_height - height//2

            original_image1 = np.array(original_image1)


            top_y = center_y - height//2 if center_y - height//2 > 0 else 0 
            bottom_y = center_y + height//2 if center_y + height//2 < image_height else image_height
            left_x = center_x - width//2 if center_x - width//2 > 0 else 0
            right_x = center_x + width//2 if center_x + width//2 < image_width else image_width

            cropped_image = original_image1[top_y:bottom_y, left_x:right_x]
            cropped_mask = mask_image1[top_y:bottom_y, left_x:right_x]
            cropped_mask_class = mask_image_class1[top_y:bottom_y, left_x:right_x]

            # save the cropped image and mask   
            image_name = os.path.basename(data['imagePath'])

            #print(cropped_image.shape, cropped_mask.shape, cropped_mask_class.shape, mask_image1.shape, image_name)
            # save the cropped image and mask
            cropped_image = Image.fromarray(cropped_image)
            cropped_mask = Image.fromarray(cropped_mask)
            cropped_mask_class = Image.fromarray(cropped_mask_class)

        
            
            #create directory if it doesn't exist
            os.makedirs(os.path.join(output_folder,'image'), exist_ok=True) 
            os.makedirs(os.path.join(output_folder,'mask'), exist_ok=True)
            os.makedirs(os.path.join(output_folder,'mask_class'), exist_ok=True)

            cropped_image_path = os.path.join(output_folder,'image',"1_" + image_name)
            cropped_mask_path = os.path.join(output_folder,'mask',"1_" + image_name)
            cropped_mask_class_path = os.path.join(output_folder,'mask_class',"1_" + image_name)

            cropped_image.save(cropped_image_path)
            cropped_mask.save(cropped_mask_path)
            cropped_mask_class.save(cropped_mask_class_path)


            


            

        # draw the rectangles on the mask image
        #mask_image1 = cv2.rectangle(mask_image1,(x,y),(x+w,y+h),(255,255,255),2)
        
    if len(contours2) != 0:

        x1,y1,w1,h1 = cv2.boundingRect(contours2[0])
        areas.append(w1*h1)
        rectangles.append((x1,y1,w1,h1))

        #mask_image2 = cv2.rectangle(mask_image2,(x1,y1),(x1+w1,y1+h1),(255,255,255),2)
        # crop the mask image and original image
        if crop == True:

            image = cv2.imread(input_image_path+os.path.basename(data['imagePath']))

            # find center of the rectangle
            center_x = x1 + w1//2
            center_y = y1 + h1//2


            image_width = mask_image2.shape[1]
            image_height = mask_image2.shape[0]
            # crop the image around the center and make sure the cropped image is within the image boundaries
            if center_x - width//2 < 0:
                center_x = width//2

            if center_x + width//2 > image_width:
                center_x = image_width - width//2

            if center_y - height//2 < 0:
                center_y = height//2

            if center_y + height//2 > image_height:
                center_y = image_height - height//2

            original_image2 = np.array(original_image2)

            top_y = center_y - height//2 if center_y - height//2 > 0 else 0 
            bottom_y = center_y + height//2 if center_y + height//2 < image_height else image_height
            left_x = center_x - width//2 if center_x - width//2 > 0 else 0
            right_x = center_x + width//2 if center_x + width//2 < image_width else image_width
            
            cropped_image = original_image2[top_y:bottom_y, left_x:right_x]
            cropped_mask = mask_image2[top_y:bottom_y, left_x:right_x]
            cropped_mask_class = mask_image_class2[top_y:bottom_y, left_x:right_x]

            
            # save the cropped image and mask   
            image_name = os.path.basename(data['imagePath'])
            #print(cropped_image.shape, cropped_mask.shape,cropped_mask_class.shape, mask_image2.shape ,image_name)
            #create directory if it doesn't exist
            os.makedirs(os.path.join(output_folder,'image'), exist_ok=True) 
            os.makedirs(os.path.join(output_folder,'mask'), exist_ok=True)
            os.makedirs(os.path.join(output_folder,'mask_class'), exist_ok=True)

            cropped_image_path = os.path.join(output_folder,'image',"2_" + image_name)
            cropped_mask_path = os.path.join(output_folder,'mask',"2_" + image_name)
            cropped_mask_class_path = os.path.join(output_folder,'mask_class',"2_" + image_name)

            
            cropped_image = Image.fromarray(cropped_image)
            cropped_mask = Image.fromarray(cropped_mask)
            cropped_mask_class = Image.fromarray(cropped_mask_class)

            cropped_image.save(cropped_image_path)
            cropped_mask.save(cropped_mask_path)
            cropped_mask_class.save(cropped_mask_class_path)


    """
    # save the mask images
    mask_image1 = Image.fromarray(mask_image1)
    mask_image2 = Image.fromarray(mask_image2)
    # Save the mask image in the class folder with the same name as the original image

    image_name = os.path.basename(data['imagePath'])
    #create directory if it doesn't exist
    os.makedirs(os.path.join(output_folder,'demo1'), exist_ok=True)
    os.makedirs(os.path.join(output_folder,'demo2'), exist_ok=True)

    mask_image_path1 = os.path.join(output_folder,'demo1', image_name)
    mask_image_path2 = os.path.join(output_folder,'demo2', image_name)

    mask_image1.save(mask_image_path1)
    mask_image2.save(mask_image_path2)
    """
    

    # find the mask with the largest area

    if len(areas) == 0:

        return 0,0,0
    
    else:

        # return max area and the corresponding rectangle

        max_area = max(areas)
        max_area_index = areas.index(max_area)
        max_area_rectangle = rectangles[max_area_index]


        return max_area,max_area_rectangle,os.path.basename(data['imagePath'])
        



def process_json_files_in_directory(input_directory, output_directory):
    # Iterate through all files in the input directory

    max_area_rectangles = []
    areas = [] 
    image_names = []
    for filename in os.listdir(input_directory):
        if filename.endswith(".json"):
            # Construct the full path to the JSON file
            json_file_path = os.path.join(input_directory, filename)

            # Generate segmentation masks and obtain class masks
            area, max_area_rectangle,image_name = generate_segmentation_masks(json_file_path, output_directory)


            
            max_area_rectangles.append(max_area_rectangle)
            areas.append(area)
            image_names.append(image_name)



            # Print information about the processed JSON file
            #print(f"Processed JSON file: {json_file_path}")

    max_area = max(areas)
    max_area_index = areas.index(max_area)
    max_area_rectangle = max_area_rectangles[max_area_index]

    x,y,w,h = max_area_rectangle

    print(f"Max area: {max_area}")
    print(f"Max area rectangle: {max_area_rectangle}")
    print(f"Image name: {image_names[max_area_index]}")

    for filename in os.listdir(input_directory):
        if filename.endswith(".json"):
            # Construct the full path to the JSON file
            json_file_path = os.path.join(input_directory, filename)

            # Generate segmentation masks and obtain class masks
            area, max_area_rectangle,image_name = generate_segmentation_masks(json_file_path, output_directory, crop = True, width = 187, height = 152, input_image_path = input_directory)


            
            max_area_rectangles.append(max_area_rectangle)
            areas.append(area)
            image_names.append(image_name)



            # Print information about the processed JSON file
            #print(f"Processed JSON file: {json_file_path}")


    # find the length and width of the largest rectangle

    


    

# Example usage:
input_directory = '/media/debayan/c7b64c90-ca4e-4192-8ed9-8fea1d005196/SchwanommaDS/UKE/labelled_ds/T2/'
output_directory = '/media/debayan/c7b64c90-ca4e-4192-8ed9-8fea1d005196/SchwanommaDS/UKE/labelled_ds/T2_cropped/'

process_json_files_in_directory(input_directory, output_directory)