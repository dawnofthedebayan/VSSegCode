# Generates image vs ground truth vs prediction plots for a given model



import os
import numpy as np
import matplotlib.pyplot as plt
import nibabel as nib

from skimage import measure
import cv2
from tqdm import tqdm


our_model_predictions = '/media/debayan/c7b64c90-ca4e-4192-8ed9-8fea1d005196/SchwanommaDS/UKE/code/inferrence/t2_2D_public_double_deep_sup_True_num_classes_1_loss_combined_concat_False_encoder_attention_2_dataset_DS4/test/predicted_mask/'

comparison_model_predictions = '/media/debayan/c7b64c90-ca4e-4192-8ed9-8fea1d005196/SchwanommaDS/UKE/code/inferrence/t2_2D_public_deep_sup_True_num_classes_1_loss_combined_dataset_DS4/test/predicted_mask/'

gt_dir = '/media/debayan/c7b64c90-ca4e-4192-8ed9-8fea1d005196/SchwanommaDS/UKE/code/inferrence/t2_2D_public_double_deep_sup_True_num_classes_1_loss_combined_concat_False_encoder_attention_2_dataset_DS4/test/gt_mask/'
org_imgs = '/media/debayan/c7b64c90-ca4e-4192-8ed9-8fea1d005196/SchwanommaDS/UKE/code/inferrence/t2_2D_public_double_deep_sup_True_num_classes_1_loss_combined_concat_False_encoder_attention_2_dataset_DS4/test/org_img/'
save_dir = '/media/debayan/c7b64c90-ca4e-4192-8ed9-8fea1d005196/SchwanommaDS/UKE/code/results/plot/2Dpublic_t2_DS4/'

#create dir if it does not exist
if not os.path.exists(save_dir):
    os.makedirs(save_dir)


# get all the files in the gt_dir
gt_files = os.listdir(gt_dir)
gt_file_filtered = []

for file in gt_files:

    new_file_name = file.split('_')[0] + '_' + file.split('_')[1] + '_' + file.split('_')[3]
    gt_file_filtered.append(new_file_name)
    

org_imgs_files = os.listdir(org_imgs)
org_imgs_files_filtered = []

for file in org_imgs_files:

    
    new_file_name = file.split('_')[0] + '_' + file.split('_')[1] + '_' + file.split('_')[2]
    org_imgs_files_filtered.append(new_file_name)

# find intersection of the files in gt_dir and our_model_predictions
intersection = list(set(gt_file_filtered) & set(org_imgs_files_filtered))


our_model_predictions_files = os.listdir(our_model_predictions)
our_model_predictions_files_filtered = []
for file in our_model_predictions_files:

    new_file_name = file.split('_')[0] + '_' + file.split('_')[1] + '_' + file.split('_')[2]
    our_model_predictions_files_filtered.append(new_file_name)


comparison_model_predictions_files = os.listdir(comparison_model_predictions)
comparison_model_predictions_files_filtered = []
for file in comparison_model_predictions_files:
    
        new_file_name = file.split('_')[0] + '_' + file.split('_')[1] + '_' + file.split('_')[2]
        comparison_model_predictions_files_filtered.append(new_file_name)

#find index in gt_files and our_model_predictions_files for the intersection
gt_files_index = []
our_model_predictions_files_index = []
comparison_model_predictions_files_index = []
org_imgs_files_index = []

for file in intersection:

    gt_files_index.append(gt_file_filtered.index(file))
    our_model_predictions_files_index.append(our_model_predictions_files_filtered.index(file))
    comparison_model_predictions_files_index.append(comparison_model_predictions_files_filtered.index(file))
    org_imgs_files_index.append(org_imgs_files_filtered.index(file))


# print the gt_files and our_model_predictions_files for the intersection
gt_files = np.array(gt_files)
our_model_predictions_files = np.array(our_model_predictions_files)
comparison_model_predictions_files = np.array(comparison_model_predictions_files)
org_imgs_files = np.array(org_imgs_files)

gt_files = gt_files[gt_files_index]
our_model_predictions_files = our_model_predictions_files[our_model_predictions_files_index]
comparison_model_predictions_files = comparison_model_predictions_files[comparison_model_predictions_files_index]
org_imgs_files = org_imgs_files[org_imgs_files_index]



for i in tqdm(range(len(gt_files))):
     
    # read nib files
    gt = nib.load(os.path.join(gt_dir, gt_files[i]))
    cascade_net_pred = nib.load(os.path.join(our_model_predictions, our_model_predictions_files[i]))
    cnn_pred = nib.load(os.path.join(comparison_model_predictions, comparison_model_predictions_files[i]))
    org = nib.load(os.path.join(org_imgs, org_imgs_files[i]))


    # take individual slices from the nib files
    gt = gt.get_fdata()
    cascade_net_pred = cascade_net_pred.get_fdata()
    cnn_pred = cnn_pred.get_fdata()
    org = org.get_fdata()

    # take the middle slice
    gt = gt[:,:,int(gt.shape[2]/2)]
    cascade_net_pred = cascade_net_pred[:,:,int(cascade_net_pred.shape[2]/2)]
    cnn_pred = cnn_pred[:,:,int(cnn_pred.shape[2]/2)]
    org = org[:,:,int(org.shape[2]/2)]

    # create an overlay of the gt and our_model_predictions and comparison_model_predictions in white, green and red respectively
    overlay = np.zeros((gt.shape[0], gt.shape[1], 3))
    overlay[:,:,0] = cnn_pred 
    overlay[:,:,1] = cascade_net_pred
    overlay[:,:,2] = gt

   

    # create a side by side plot of the original image, ground truth and the overlay
    fig, ax = plt.subplots(1,3, figsize=(15,15))
    ax[0].imshow(org, cmap='gray')
    #ax[0].set_title('Original Image')
    ax[1].imshow(gt, cmap='gray')
    #ax[1].set_title('Ground Truth')
    ax[2].imshow(overlay)

    #remove the axis
    ax[0].axis('off')
    ax[1].axis('off')
    ax[2].axis('off')

   

    # save the plot
    plt.savefig(os.path.join(save_dir, gt_files[i].split('.')[0] + '.png'))












