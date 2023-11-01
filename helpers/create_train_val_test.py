# Create training, validation, and test data loaders
import os
import random
import numpy as np
import torch
import torch.nn as nn
import pandas as pd

import cv2

t1_image_directory = '/media/debayan/c7b64c90-ca4e-4192-8ed9-8fea1d005196/SchwanommaDS/UKE/labelled_ds/T1_cropped/image/'
t2_image_directory = '/media/debayan/c7b64c90-ca4e-4192-8ed9-8fea1d005196/SchwanommaDS/UKE/labelled_ds/T2_cropped/image/' 
image_directory = '/media/debayan/c7b64c90-ca4e-4192-8ed9-8fea1d005196/SchwanommaDS/UKE/labelled_ds/{}_cropped/image/'
mask_image_directory = '/media/debayan/c7b64c90-ca4e-4192-8ed9-8fea1d005196/SchwanommaDS/UKE/labelled_ds/{}_cropped/mask_class/'

seed = 500
random.seed(seed)

csv_directory = f'/media/debayan/c7b64c90-ca4e-4192-8ed9-8fea1d005196/SchwanommaDS/UKE/labelled_ds/csv/seed_{seed}/'

# create the csv directory if it doesn't exist
os.makedirs(csv_directory, exist_ok=True)

# get all the images in the directory 
t1_images = os.listdir(t1_image_directory)
t2_images = os.listdir(t2_image_directory)

# replace _T1 with empty string
t1_images = [image.replace('_T1','') for image in t1_images]

# replace _T2 with empty string
t2_images = [image.replace('_T2','') for image in t2_images]

# get the common images in the two directories
common_images = list(set(t1_images).intersection(t2_images))

class_0_1_images = []
class_0_2_images = []
class_0_1_2_images = []

class_0_1_patients = []
class_0_2_patients = []
class_0_1_2_patients = []

for image in common_images:

    # get the mask 
    image_file = image.replace('.png','_T1.png')
    mask_image_path = os.path.join(mask_image_directory.format("T1"), image_file)

    mask = cv2.imread(mask_image_path,0)

    # get the unique values in the mask 
    unique_values = np.unique(mask)

    # check if the mask has only 0 and 1
    if len(unique_values) == 2:

        # check if the mask has only 0 and 1
        if 0 in unique_values and 1 in unique_values:
            class_0_1_images.append(image)
            patient = image.split('_')[2]
            class_0_1_patients.append(patient)



        # check if the mask has only 0 and 2
        if 0 in unique_values and 2 in unique_values:
            class_0_2_images.append(image)
            patient = image.split('_')[2]
            class_0_2_patients.append(patient)


    # check if the mask has 0, 1 and 2
    if len(unique_values) == 3:
        class_0_1_2_images.append(image)
        patient = image.split('_')[2]

        class_0_1_2_patients.append(patient)



print('Number of images with class 0 and 1: ', len(class_0_1_images))
print('Number of images with class 0 and 2: ', len(class_0_2_images))
print('Number of images with class 0, 1 and 2: ', len(class_0_1_2_images))
print('Total number of images: ', len(common_images))

print('Number of patients with class 0 and 1: ', len(set(class_0_1_patients)))
print('Number of patients with class 0 and 2: ', len(set(class_0_2_patients)))
print('Number of patients with class 0, 1 and 2: ', len(set(class_0_1_2_patients)))
print('Total number of patients: ', len(set(class_0_1_patients + class_0_2_patients + class_0_1_2_patients)))

# is there any overlap between the patients with class 0 and 1 and class 0 and 2
class_0_1_patients = set(class_0_1_patients)
class_0_2_patients = set(class_0_2_patients)
class_0_1_2_patients = set(class_0_1_2_patients)


# remove common patients that are in class 0 and 1 and class 0 and 2 
class_0_1_patients = class_0_1_patients.difference(class_0_2_patients) 
# remove common patients that are in class 0 and 1 and class 0 1 and 2
class_0_1_patients = class_0_1_patients.difference(class_0_1_2_patients) 

# remove common patients that are in class 0 and 2 and class 0 1 and 2

class_0_2_patients = class_0_2_patients.difference(class_0_1_patients)
class_0_2_patients = class_0_2_patients.difference(class_0_1_2_patients)

# is there any overlap between the patients with class 0 and 1 and class 0, 1 and 2
class_0_1_2_patients = set(class_0_1_2_patients)

print('Number of patients with class 0 and 1 and class 0 and 2: ', len(class_0_1_patients.intersection(class_0_2_patients)))


print('Number of patients with class 0 and 1 and class 0, 1 and 2: ', len(class_0_1_patients.intersection(class_0_1_2_patients)))

# is there any overlap between the patients with class 0 and 2 and class 0, 1 and 2
print('Number of patients with class 0 and 2 and class 0, 1 and 2: ', len(class_0_2_patients.intersection(class_0_1_2_patients)))


print('Number of patients with class 0 and 1: ', len(class_0_1_patients))
print('Number of patients with class 0 and 2: ', len(class_0_2_patients))
print('Number of patients with class 0, 1 and 2: ', len(class_0_1_2_patients))
print('Total number of patients: ', len(class_0_1_patients) + len(class_0_2_patients) + len(class_0_1_2_patients))




# create a 5 fold train, val and test split such that the patients are not repeated in the train, val and test splits



# shuffle the patients

test_set_images = []

# get 10 % of the patients for the test set

class_0_1_patients_test = random.sample(class_0_1_patients, int(0.1 * len(class_0_1_patients))) 
class_0_2_patients_test = random.sample(class_0_2_patients, int(0.1 * len(class_0_2_patients)))
class_0_1_2_patients_test = random.sample(class_0_1_2_patients, int(0.1 * len(class_0_1_2_patients)))

# remove the test set patients from the train and val set
class_0_1_patients = class_0_1_patients.difference(class_0_1_patients_test)
class_0_2_patients = class_0_2_patients.difference(class_0_2_patients_test)
class_0_1_2_patients = class_0_1_2_patients.difference(class_0_1_2_patients_test)

class_0_1_patients = np.array(list(class_0_1_patients))
class_0_2_patients = np.array(list(class_0_2_patients))
class_0_1_2_patients = np.array(list(class_0_1_2_patients))

random.shuffle(class_0_1_patients)
random.shuffle(class_0_2_patients)
random.shuffle(class_0_1_2_patients)



# split the patients into 5 folds

class_0_1_patients_folds = np.array_split(list(class_0_1_patients), 5) 
class_0_2_patients_folds = np.array_split(list(class_0_2_patients), 5)
class_0_1_2_patients_folds = np.array_split(list(class_0_1_2_patients), 5)


# get the train and val splits for each fold
# get the test set images

test_set_images.extend([image for image in class_0_1_images if image.split('_')[2] in class_0_1_patients_test])
test_set_images.extend([image for image in class_0_2_images if image.split('_')[2] in class_0_2_patients_test])
test_set_images.extend([image for image in class_0_1_2_images if image.split('_')[2] in class_0_1_2_patients_test])

for i in range(5):

    train_set_images = []
    val_set_images = []

    # get the  val set patients
    val_set_patients_01 = class_0_1_patients_folds[i]
    val_set_patients_02 = class_0_2_patients_folds[i]
    val_set_patients_012 = class_0_1_2_patients_folds[i]

    # get the train set patients

    train_set_patients_01 = np.concatenate(class_0_1_patients_folds[:i] + class_0_1_patients_folds[i+1:])
    train_set_patients_02 = np.concatenate(class_0_2_patients_folds[:i] + class_0_2_patients_folds[i+1:])
    train_set_patients_012 = np.concatenate(class_0_1_2_patients_folds[:i] + class_0_1_2_patients_folds[i+1:])



    # get the train set images
    for patient in train_set_patients_01:
        train_set_images.extend([image for image in class_0_1_images if image.split('_')[2] == patient])

    for patient in train_set_patients_02:
        train_set_images.extend([image for image in class_0_2_images if image.split('_')[2] == patient])

    for patient in train_set_patients_012:
        train_set_images.extend([image for image in class_0_1_2_images if image.split('_')[2] == patient])

    # get the val set images
    for patient in val_set_patients_01:
        val_set_images.extend([image for image in class_0_1_images if image.split('_')[2] == patient])

    for patient in val_set_patients_02:
        val_set_images.extend([image for image in class_0_2_images if image.split('_')[2] == patient])

    for patient in val_set_patients_012:
        val_set_images.extend([image for image in class_0_1_2_images if image.split('_')[2] == patient])

    # total train, validation and test set patients
    print('Number of train patients: ', len(train_set_patients_01) + len(train_set_patients_02) + len(train_set_patients_012))
    print('Number of val patients: ', len(val_set_patients_01) + len(val_set_patients_02) + len(val_set_patients_012))
    print('Number of test patients: ', len(class_0_1_patients_test) + len(class_0_2_patients_test) + len(class_0_1_2_patients_test))

    # number of images per patient in train, val and test set with mean and std

    train_set_images_per_patient = [len([image for image in train_set_images if image.split('_')[2] == patient]) for patient in train_set_patients_01] + [len([image for image in train_set_images if image.split('_')[2] == patient]) for patient in train_set_patients_02] + [len([image for image in train_set_images if image.split('_')[2] == patient]) for patient in train_set_patients_012]
    val_set_images_per_patient = [len([image for image in val_set_images if image.split('_')[2] == patient]) for patient in val_set_patients_01] + [len([image for image in val_set_images if image.split('_')[2] == patient]) for patient in val_set_patients_02] + [len([image for image in val_set_images if image.split('_')[2] == patient]) for patient in val_set_patients_012]
    test_set_images_per_patient = [len([image for image in test_set_images if image.split('_')[2] == patient]) for patient in class_0_1_patients_test] + [len([image for image in test_set_images if image.split('_')[2] == patient]) for patient in class_0_2_patients_test] + [len([image for image in test_set_images if image.split('_')[2] == patient]) for patient in class_0_1_2_patients_test]

    print('Number of images per patient in train set: ', np.mean(train_set_images_per_patient), np.std(train_set_images_per_patient))
    print('Number of images per patient in val set: ', np.mean(val_set_images_per_patient), np.std(val_set_images_per_patient))
    print('Number of images per patient in test set: ', np.mean(test_set_images_per_patient), np.std(test_set_images_per_patient))


    # number of images per patient in whole dataset with mean and std

    whole_set_images_per_patient = train_set_images_per_patient + val_set_images_per_patient + test_set_images_per_patient

    print('Number of images per patient in whole dataset: ', np.mean(whole_set_images_per_patient), np.std(whole_set_images_per_patient))


    
    
    print('Number of train images: ', len(train_set_images))
    print('Number of val images: ', len(val_set_images))
    print('Number of test images: ', len(test_set_images))


    # create the train, val and test csv files with T1, T1 mask, T2 and T2 mask paths 

    train_set = pd.DataFrame(columns=['T1', 'T1_mask', 'T2', 'T2_mask'])
    val_set = pd.DataFrame(columns=['T1', 'T1_mask', 'T2', 'T2_mask'])
    test_set = pd.DataFrame(columns=['T1', 'T1_mask', 'T2', 'T2_mask'])

    for image in train_set_images:

        image_file_t1 = image.replace('.png','_T1.png')
        image_file_t2 = image.replace('.png','_T2.png')

        image_path_t1 = os.path.join(image_directory.format('T1'), image_file_t1) 
        image_path_t2 = os.path.join(image_directory.format('T2'), image_file_t2)

        mask_image_path_t1 = os.path.join(mask_image_directory.format('T1'), image_file_t1)
        mask_image_path_t2 = os.path.join(mask_image_directory.format('T2'), image_file_t2)

        train_set.loc[len(train_set)] = [image_path_t1, mask_image_path_t1, image_path_t2, mask_image_path_t2]
    for image in val_set_images:

        image_file_t1 = image.replace('.png','_T1.png')
        image_file_t2 = image.replace('.png','_T2.png')

        image_path_t1 = os.path.join(image_directory.format('T1'), image_file_t1) 
        image_path_t2 = os.path.join(image_directory.format('T2'), image_file_t2)

        mask_image_path_t1 = os.path.join(mask_image_directory.format('T1'), image_file_t1)
        mask_image_path_t2 = os.path.join(mask_image_directory.format('T2'), image_file_t2)

        val_set.loc[len(val_set)] = [image_path_t1, mask_image_path_t1, image_path_t2, mask_image_path_t2]

    for image in test_set_images:

        image_file_t1 = image.replace('.png','_T1.png')
        image_file_t2 = image.replace('.png','_T2.png')

        image_path_t1 = os.path.join(image_directory.format('T1'), image_file_t1) 
        image_path_t2 = os.path.join(image_directory.format('T2'), image_file_t2)

        mask_image_path_t1 = os.path.join(mask_image_directory.format('T1'), image_file_t1)
        mask_image_path_t2 = os.path.join(mask_image_directory.format('T2'), image_file_t2)

        test_set.loc[len(test_set)] = [image_path_t1, mask_image_path_t1, image_path_t2, mask_image_path_t2]

    # save the train, val and test csv files

    #train_set.to_csv(csv_directory + 'train_set_{}.csv'.format(i), index=False)
    #val_set.to_csv(csv_directory + 'val_set_{}.csv'.format(i), index=False)
    #test_set.to_csv(csv_directory + 'test_set_{}.csv'.format(i), index=False)






    


    


   

    
    






    




    






    







# search for the common images in the two directories
    