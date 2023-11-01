import SimpleITK as sitk
import numpy as np
import cv2

def rigid_registration_and_transform(fixed_image_path, moving_image_path, segmentation_mask_path, output_image_path, output_mask_path):

    print("Fixed image path: ", fixed_image_path)
    print("Moving image path: ", moving_image_path)
    print("Segmentation mask path: ", segmentation_mask_path)
    print("Output image path: ", output_image_path)
    print("Output mask path: ", output_mask_path)

    # Load the fixed and moving images (in PNG format)
    fixed_image = sitk.ReadImage(fixed_image_path, sitk.sitkFloat32)
    moving_image = sitk.ReadImage(moving_image_path, sitk.sitkFloat32)

    # Load the segmentation mask (in PNG format)
    segmentation_mask = cv2.imread(segmentation_mask_path, cv2.IMREAD_GRAYSCALE)
    segmentation_mask = sitk.GetImageFromArray(segmentation_mask.astype(np.float32))

    # Create a registration method
    registration = sitk.ImageRegistrationMethod()

    # Set the metric (mean squared error is commonly used)
    registration.SetMetricAsMeanSquares()

    # Set the optimizer (gradient descent is commonly used)
    registration.SetOptimizerAsRegularStepGradientDescent(learningRate=1.0, minStep=0.01, numberOfIterations=100)

    # Set the interpolator for image sampling
    registration.SetInterpolator(sitk.sitkLinear)

    # Set an initial transform (identity transform in this case)
    initial_transform = sitk.Euler2DTransform()
    registration.SetInitialTransform(initial_transform)

    # Perform rigid registration to align the moving image with the fixed image
    final_transform = registration.Execute(fixed_image, moving_image)

    # Apply the same transform to the segmentation mask
    transformed_segmentation_mask = sitk.Resample(segmentation_mask, fixed_image, final_transform, sitk.sitkNearestNeighbor, 0.0, sitk.sitkUInt8)

    # Save the registered image and transformed segmentation mask (in PNG format)
    final_transformed_image = sitk.Resample(moving_image, fixed_image, final_transform, sitk.sitkLinear, 0.0,sitk.sitkUInt8)

    # Save the registered image and transformed segmentation mask (in PNG format)
    sitk.WriteImage(final_transformed_image, output_image_path)
    sitk.WriteImage(transformed_segmentation_mask, output_mask_path)

import os
from tqdm import tqdm
image_directory = '/media/debayan/c7b64c90-ca4e-4192-8ed9-8fea1d005196/SchwanommaDS/UKE/labelled_ds/T1/'
mask_directory1 = '/media/debayan/c7b64c90-ca4e-4192-8ed9-8fea1d005196/SchwanommaDS/UKE/labelled_ds/T1_masks/SchwannomaBrain/'
mask_directory2 = '/media/debayan/c7b64c90-ca4e-4192-8ed9-8fea1d005196/SchwanommaDS/UKE/labelled_ds/T1_masks/SchwannomaCanal/'

output_directory = '/media/debayan/c7b64c90-ca4e-4192-8ed9-8fea1d005196/SchwanommaDS/UKE/registered_labelled_ds_CASE_32/'
output_image_directory = os.path.join(output_directory, 'T1')
output_mask_directory1 = os.path.join(output_directory + "/T1_masks", 'SchwannomaBrain')
output_mask_directory2 = os.path.join(output_directory + "/T1_masks", 'SchwannomaCanal')


#creat the output directory if it doesn't exist
os.makedirs(output_image_directory, exist_ok=True)
os.makedirs(output_mask_directory1, exist_ok=True)
os.makedirs(output_mask_directory2, exist_ok=True)

fixed_image_path = "/media/debayan/c7b64c90-ca4e-4192-8ed9-8fea1d005196/SchwanommaDS/UKE/labelled_ds/T1/CASE_32_IMG_1_T1.png"

images = os.listdir(image_directory)

# only keep png files
images = [image for image in images if image.endswith('.png')] 


# parse through the directory and find the images
for filename in tqdm(images,total=len(images)):

    #image path 
    image_path = os.path.join(image_directory, filename)

    # Construct the full path to the corresponding masks
    mask_file_base = os.path.splitext(filename)[0]
    mask1_path = os.path.join(mask_directory1, mask_file_base + '.png')
    mask2_path = os.path.join(mask_directory2, mask_file_base + '.png') 

    # rigid registration and transform 

    # Construct the output image path
    output_image_path = os.path.join(output_image_directory, filename) 

    # Construct the output mask path
    output_mask_path1 = os.path.join(output_mask_directory1, filename)
    output_mask_path2 = os.path.join(output_mask_directory2, filename)


    # rigid registration and transform if mask exists
    
    if os.path.exists(mask1_path) :

        rigid_registration_and_transform(fixed_image_path, image_path, mask1_path, output_image_path, output_mask_path1)
        print(f"Processed image: {image_path}")
        print(f"Processed mask: {mask1_path}")
        print(f"Registered image saved to: {output_image_path}")
        print(f"Registered mask saved to: {output_mask_path1}")

    if os.path.exists(mask2_path) :

        rigid_registration_and_transform(fixed_image_path, image_path, mask2_path, output_image_path, output_mask_path2)
        print(f"Processed image: {image_path}")
        print(f"Processed mask: {mask2_path}")
        print(f"Registered image saved to: {output_image_path}")
        print(f"Registered mask saved to: {output_mask_path2}")
