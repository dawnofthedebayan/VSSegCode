# Data module class responsible for loading and preprocessing data 

import os
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import pytorch_lightning as pl

import cv2
import albumentations as A


from albumentations.pytorch import ToTensorV2

import re


class UKEDataloader(Dataset):
    def __init__(self, csv_file, transform=None, num_classes=2):
        self.data = pd.read_csv(csv_file)
        self.transform = transform
        self.num_classes = num_classes

    def __len__(self):
        return len(self.data)
    

    def extract_number(self,filename):
        match = re.search(r'(\d+)\.png', filename)
        if match:
            return int(match.group(1))
        return -1
    
    def seed_everything(self, seed=42):
        random.seed(seed)
        os.environ['PYTHONHASHSEED'] = str(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.use_deterministic_algorithms(mode=True,warn_only=True)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    def __getitem__(self, idx):

        image_path_t1 = self.data.iloc[idx, 0]
        image_path_t2 = self.data.iloc[idx, 1]
        mask_path = self.data.iloc[idx, 2]


        # get all images in image_path_t1

        # get names of all files inside the directory
        image_path_t1_files = os.listdir(image_path_t1)

        # file is of shape xx.png. Sort the files in increasing order of xx
        
        image_path_t1_files = sorted(image_path_t1_files, key=self.extract_number)
       
        

        # create image paths for all files
        images_p_t1 = [os.path.join(image_path_t1, f) for f in image_path_t1_files]


        # get all images in image_path_t2

        # get names of all files inside the directory
        image_path_t2_files = os.listdir(image_path_t2)

        image_path_t2_files = sorted(image_path_t2_files, key=self.extract_number)


        # create image paths for all files
        images_p_t2 = [os.path.join(image_path_t2, f) for f in image_path_t2_files]

        # get all images in mask_path

        # get names of all files inside the directory
        mask_path_files = os.listdir(mask_path)

        mask_path_files = sorted(mask_path_files, key=self.extract_number)

        # create image paths for all files


        masks_p = [os.path.join(mask_path, f) for f in mask_path_files]

        

        # sort in numerical order
        
        
        # read image and concatenate them along the channel axis
        image_t1 = [cv2.imread(f, cv2.IMREAD_GRAYSCALE) for f in images_p_t1]
        image_t2 = [cv2.imread(f, cv2.IMREAD_GRAYSCALE) for f in images_p_t2]
        mask_t1 = [cv2.imread(f, cv2.IMREAD_GRAYSCALE) for f in masks_p]

        # convert to numpy arrays
        image_t1 = np.array(image_t1)
        image_t2 = np.array(image_t2)
        mask_t1 = np.array(mask_t1)

        
        
        
        # concatanate the images t1 and t2

        image_t1t2 = np.concatenate((image_t1, image_t2), axis=0)

        # permute the axes to get the shape in the form of ( height, width, channels)

        image_t1t2 = np.transpose(image_t1t2, (1, 2, 0))

        image_t1 = np.transpose(image_t1, (1, 2, 0))

        image_t2 = np.transpose(image_t2, (1, 2, 0))

        mask_t1 = np.transpose(mask_t1, (1, 2, 0))

        
        

        # get random integer between 0 and 100000

        random_int = random.randint(0, 100000)


        # set the seed for the data augmentation transforms
        self.seed_everything(random_int)

    

       
        
        


        # transform t1 and t2 images separately
        transformed_t1 = self.transform(image=image_t1, mask=mask_t1)

        self.seed_everything(random_int)

        transformed_t2 = self.transform(image=image_t2, mask=mask_t1)


        

        # image t1 and t2
        image_t1 = transformed_t1['image']
        image_t2 = transformed_t2['image']
        mask_t1 = transformed_t1['mask']
        mask_t2 = transformed_t2['mask']

        # create t1t2 image
        image_t1t2 = torch.cat((image_t1, image_t2), dim=0)

        # permute mask to get the shape in the form of ( channels, height, width)
        mask_t1 = mask_t1.permute(2, 0, 1)

        # make mask float 
        mask_t1 = mask_t1.float()

        mask_t2 = mask_t2.permute(2, 0, 1)
        mask_t2 = mask_t2.float()

        


        
        return {'image_t1t2': image_t1t2, 'mask_t1t2': mask_t1, 'image_t2': image_t2, 'mask_t2': mask_t2, 'image_t1': image_t1, 'mask_t1': mask_t1}

# Define data augmentation transforms (you can adjust these as needed)
data_transforms = A.Compose([
    A.Resize(128, 128),
    A.HorizontalFlip(p=0.5),
    A.VerticalFlip(p=0.5),
    A.RandomRotate90(p=0.5),
    #A.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.1, rotate_limit=15, p=0.5),
    #A.RandomBrightnessContrast(p=0.5),
    A.GaussNoise(p=0.5),
    A.Normalize(mean= 0.5, std= 0.5, max_pixel_value=255, p=1.0),
    ToTensorV2()
])

test_transforms = A.Compose([
    A.Resize(128, 128),
    A.Normalize(mean=0.5, std= 0.5, max_pixel_value=255, p=1.0),
    ToTensorV2()
])



# Create a PyTorch Lightning DataLoader
class PublicDatamodule25D(pl.LightningDataModule):
    def __init__(self, csv_root, fold, batch_size=32, num_workers=8, num_classes=2,seed=42):
        super().__init__()
        self.batch_size = batch_size
        
        self.train_csv = csv_root + 'train_set_' + str(fold) + '.csv'
        self.val_csv = csv_root + 'val_set_' + str(fold) + '.csv'
        self.test_csv = csv_root + 'test_set_' + str(fold) + '.csv'
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.num_classes = num_classes
        self.seed = seed

    def setup(self, stage=None):
        self.train_dataset = UKEDataloader(self.train_csv, transform=data_transforms, num_classes=self.num_classes)
        self.val_dataset = UKEDataloader(self.val_csv, transform=test_transforms, num_classes=self.num_classes)
        self.test_dataset = UKEDataloader(self.test_csv, transform=test_transforms, num_classes=self.num_classes)



    def train_dataloader(self):

        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers,worker_init_fn=np.random.seed(self.seed))
    
    def val_dataloader(self):

        return DataLoader(self.val_dataset, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers,worker_init_fn=np.random.seed(self.seed))
    
    def test_dataloader(self):

        return DataLoader(self.test_dataset, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers,worker_init_fn=np.random.seed(self.seed))


