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




class UKEDataloader(Dataset):
    def __init__(self, csv_file, transform=None, num_classes=2):
        self.data = pd.read_csv(csv_file)
        self.transform = transform
        self.num_classes = num_classes

    def __len__(self):
        return len(self.data)
    
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
        mask_path_t1 = self.data.iloc[idx, 1]
        image_path_t2 = self.data.iloc[idx, 2]
        mask_path_t2 = self.data.iloc[idx, 3]

        # Read the image and mask from file
        
        image_t1 = cv2.imread(image_path_t1, cv2.IMREAD_GRAYSCALE)
        mask_t1 = cv2.imread(mask_path_t1, cv2.IMREAD_GRAYSCALE)
        image_t2 = cv2.imread(image_path_t2, cv2.IMREAD_GRAYSCALE)
        mask_t2 = cv2.imread(mask_path_t2, cv2.IMREAD_GRAYSCALE)



        # Convert the image and mask to numpy arrays
        image_t1 = np.array(image_t1)
        mask_t2 = np.array(mask_t2)
        image_t2 = np.array(image_t2)
        mask_t1 = np.array(mask_t1)

        

        

        # if dimensions do not match, resize the image and mask in the size of the larger image
        if image_t1.shape[0] != image_t2.shape[0] or image_t1.shape[1] != image_t2.shape[1]:

            image_t2 = cv2.resize(image_t2, (image_t1.shape[1], image_t1.shape[0]), interpolation=cv2.INTER_NEAREST)
            mask_t2 = cv2.resize(mask_t2, (image_t1.shape[1], image_t1.shape[0]), interpolation=cv2.INTER_NEAREST)
            mask_t1 = cv2.resize(mask_t1, (image_t1.shape[1], image_t1.shape[0]), interpolation=cv2.INTER_NEAREST)
            
        
                

        # concatanate the images t1 and t2
        #print(image_t1.shape, image_t2.shape, "image shapes")
        #print(mask_t1.shape, mask_t2.shape, "mask shapes")
        image_t1 = np.expand_dims(image_t1, axis=2)
        image_t2 = np.expand_dims(image_t2, axis=2)
        
        image_t1t2 = np.concatenate((image_t1, image_t2), axis=2)
        

        # get random integer between 0 and 100000

        random_int = random.randint(0, 100000)


        # set the seed for the data augmentation transforms
        self.seed_everything(random_int)

        # Apply the data augmentation transforms
        transformed = self.transform(image=image_t1t2, mask=mask_t1)
        image = transformed['image']
        mask = transformed['mask']

        # apply the data augmentation transforms on t1 image
        image_t1 = image[:1,:,:]
        



        # Apply the data augmentation transforms on t2 image
        # repeat image_t2 for 2 channels
        image_t2 = np.concatenate((image_t2, image_t2), axis=2)

        self.seed_everything(random_int)
        transformed = self.transform(image=image_t2, mask=mask_t2)
        image_t2 = transformed['image']
        image_t2 = image_t2[:1,:,:]
        mask_t2 = transformed['mask']

        

        if self.num_classes == 2:
            
            # Convert the segmentation mask torch tensor to binary (0s and 1s)
            mask = (mask > 0).float()
            mask_t2 = (mask_t2 > 0).float()

        
        # Convert the segmentation mask to binary (0s and 1s)
        #mask = (mask > 0).astype(np.uint8)
        
        return {'image_t1t2': image, 'mask_t1t2': mask, 'image_t2': image_t2, 'mask_t2': mask_t2, 'image_t1': image_t1, 'mask_t1': mask}

# Define data augmentation transforms (you can adjust these as needed)
data_transforms = A.Compose([
    A.Resize(256, 256),
    A.HorizontalFlip(p=0.5),
    A.VerticalFlip(p=0.5),
    A.RandomRotate90(p=0.5),
    #A.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.1, rotate_limit=15, p=0.5),
    #A.RandomBrightnessContrast(p=0.5),
    A.GaussNoise(p=0.5),
    A.Normalize(mean=[0.5, 0.5], std=[0.5, 0.5], max_pixel_value=255, p=1.0),
    ToTensorV2()
])

test_transforms = A.Compose([
    A.Resize(256, 256),
    A.Normalize(mean=[0.5, 0.5], std=[0.5, 0.5], max_pixel_value=255, p=1.0),
    ToTensorV2()
])



# Create a PyTorch Lightning DataLoader
class UKESegmentationDataModule(pl.LightningDataModule):
    def __init__(self, csv_root, fold, batch_size=32, num_workers=8, num_classes=2,seed=42,test_batch_size=None):
        super().__init__()
        self.batch_size = batch_size
        
        self.train_csv = csv_root + 'train_set_' + str(fold) + '.csv'
        self.val_csv = csv_root + 'val_set_' + str(fold) + '.csv'
        self.test_csv = csv_root + 'test_set_' + str(fold) + '.csv'
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.num_classes = num_classes
        self.seed = seed
        self.test_batch_size = test_batch_size

    def setup(self, stage=None):
        self.train_dataset = UKEDataloader(self.train_csv, transform=data_transforms, num_classes=self.num_classes)
        self.val_dataset = UKEDataloader(self.val_csv, transform=test_transforms, num_classes=self.num_classes)
        self.test_dataset = UKEDataloader(self.test_csv, transform=test_transforms, num_classes=self.num_classes)



    def train_dataloader(self):

        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers,worker_init_fn=np.random.seed(self.seed))
    
    def val_dataloader(self):
        if self.test_batch_size is not None:
            return DataLoader(self.val_dataset, batch_size=self.test_batch_size, shuffle=False, num_workers=self.num_workers,worker_init_fn=np.random.seed(self.seed))
        else:
            return DataLoader(self.val_dataset, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers,worker_init_fn=np.random.seed(self.seed))
    
    def test_dataloader(self):
        
        if self.test_batch_size is not None:
            return DataLoader(self.test_dataset, batch_size=self.test_batch_size, shuffle=False, num_workers=self.num_workers,worker_init_fn=np.random.seed(self.seed))
        else:
            return DataLoader(self.test_dataset, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers,worker_init_fn=np.random.seed(self.seed))


