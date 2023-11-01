# this script reads nii.gz image and nii.gz mask, finds the largest connected component in the mask and crops the image and mask to the bounding box of the largest connected component.



import nibabel as nib
import numpy as np
import os
import matplotlib.pyplot as plt
from skimage import measure
from scipy import ndimage
import argparse
from scipy.ndimage import label
from tqdm import tqdm

root_dir = '/media/debayan/c7b64c90-ca4e-4192-8ed9-8fea1d005196/SchwanommaDS/VS-Seg/'
save_dir = '/media/debayan/c7b64c90-ca4e-4192-8ed9-8fea1d005196/SchwanommaDS/VS-Seg-cropped_100x100x50/'

# function to calculate largest connected component in a mask
def crop_around(mask,t1_image,t2_image):

    labels, num_labels = label(mask)
    label_indices = np.arange(1, num_labels + 1)

    # find bounding boxes for label 1 
    label_boxes = np.array([ndimage.find_objects(labels == i)[0] for i in label_indices])
    # find the center of the bounding box
    centers = np.array([(int((x.start + x.stop) / 2), int((y.start + y.stop) / 2), int((z.start + z.stop) / 2)) for x, y, z in label_boxes])

    

    new_box = np.array([centers[0] - [50, 50, 25], centers[0] + [50, 50, 25]])

    # crop the image and mask to the bounding box if the bounding box is within the image, adjust the bounding box otherwise
    for i in range(3):

        mask_shape = mask.shape[i]
        new_box_shape = new_box[1][i] - new_box[0][i]

        if new_box[0][i] < 0:
            new_box[0][i] = 0
            new_box[1][i] = new_box_shape

        if new_box[1][i] > mask_shape:
            new_box[1][i] = mask_shape
            new_box[0][i] = mask_shape - new_box_shape

    # crop the image and mask to the bounding box
    mask = mask[new_box[0][0]:new_box[1][0], new_box[0][1]:new_box[1][1], new_box[0][2]:new_box[1][2]]
    t1_image = t1_image[new_box[0][0]:new_box[1][0], new_box[0][1]:new_box[1][1], new_box[0][2]:new_box[1][2]]
    t2_image = t2_image[new_box[0][0]:new_box[1][0], new_box[0][1]:new_box[1][1], new_box[0][2]:new_box[1][2]]

    return t1_image, t2_image, mask





# gather folders containing images and masks

folders = os.listdir(root_dir)

for folder in tqdm(folders, total=len(folders)):

    # read files in the folder
    files = os.listdir(os.path.join(root_dir, folder))

    # if seg in the file name, then it is a mask
    mask_files = [f for f in files if 'seg' in f]
    # if _t1_ in the file name, then it is a t1 image
    t1_files = [f for f in files if '_t1_' in f]
    # if _t2_ in the file name, then it is a t2 image
    t2_files = [f for f in files if '_t2_' in f]

    if len(mask_files) == 0:
        print('No mask file found in the folder')
        continue

    if len(t1_files) == 0:

        print('No t1 file found in the folder')
        continue

    if len(t2_files) == 0:

        print('No t2 file found in the folder')
        continue


    # read the mask file
    mask_file = mask_files[0]
    mask = nib.load(os.path.join(root_dir, folder, mask_file))


    # read the t1 file
    t1_file = t1_files[0]
    t1 = nib.load(os.path.join(root_dir, folder, t1_file))

    # read the t2 file
    t2_file = t2_files[0]
    t2 = nib.load(os.path.join(root_dir, folder, t2_file))


    # crop the image and mask to the bounding box of the largest connected component in the mask
    mask_data = mask.get_fdata()
    t1_data = t1.get_fdata()
    t2_data = t2.get_fdata()

    t1_data, t2_data, mask_data = crop_around(mask_data, t1_data, t2_data)

    # save the cropped image and mask
    mask_data = nib.Nifti1Image(mask_data, mask.affine, mask.header)
    t1_data = nib.Nifti1Image(t1_data, t1.affine, t1.header)
    t2_data = nib.Nifti1Image(t2_data, t2.affine, t2.header)

    # create folder if it doesn't exist
    os.makedirs(os.path.join(save_dir, folder), exist_ok=True)

    # save the cropped image and mask
    nib.save(mask_data, os.path.join(save_dir, folder, mask_file))
    nib.save(t1_data, os.path.join(save_dir, folder, t1_file))
    nib.save(t2_data, os.path.join(save_dir, folder, t2_file))

    print('Saved cropped image and mask for folder: ', folder)




    
