import torch
import torchio
import pytorch_lightning as pl
import pandas as pd
import numpy as np
class PublicDataset(torch.utils.data.Dataset):
    def __init__(self, csv, transform=None):
        self.data = pd.read_csv(csv)
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        t1_path, t2_path, mask_path = row["T1"], row["T2"], row["mask"]
        
        # Load images using torchio
        t1 = torchio.Image(t1_path, type=torchio.INTENSITY)
        t2 = torchio.Image(t2_path, type=torchio.INTENSITY)
        mask = torchio.Image(mask_path, type=torchio.LABEL)


        
        subject = torchio.Subject({'T1': t1, 'T2': t2, 'mask': mask})
        
        if self.transform:
            subject = self.transform(subject)
        
        return subject['T1']['data'], subject['T2']['data'], subject['mask']['data']

class PublicDatamodule(pl.LightningDataModule):
    def __init__(self, csv_root, fold, batch_size=32, num_workers=8, num_classes=2, seed=42):
        super().__init__()
        self.csv_root = csv_root
        self.fold = fold

        self.train_csv = csv_root + 'train_set_' + str(fold) + '.csv'
        self.val_csv = csv_root + 'val_set_' + str(fold) + '.csv'
        self.test_csv = csv_root + 'test_set_' + str(fold) + '.csv'


        self.batch_size = batch_size
        self.num_workers = num_workers
        self.num_classes = num_classes
        self.seed = seed

    def setup(self, stage=None):
        
        # Create train, val, test datasets

        transforms = {
            torchio.RandomFlip(axes=('LR',)): 0.5,
            torchio.RandomAffine(scales=(0.9, 1.1), degrees=(0, 15)) : 0.5,
            #torchio.RandomGamma(log_gamma=(0.7, 1.3)) : 0.5,
            torchio.RandomNoise(std=(0, 0.1))   : 0.5,
           
        }
        
        train_transform = torchio.Compose([
            torchio.EnsureShapeMultiple(32),
            torchio.OneOf(transforms,p=0.5),
            torchio.RescaleIntensity(out_min_max=(0, 1), percentiles=(0.5, 99.5))
            ,
        ])  

        val_transform = torchio.Compose([
            torchio.EnsureShapeMultiple(32),
            torchio.RescaleIntensity(out_min_max=(0, 1), percentiles=(0.5, 99.5))
        ])

       



        self.train_dataset = PublicDataset(self.train_csv, transform=train_transform)
        self.val_dataset = PublicDataset(self.val_csv, transform=val_transform)
        self.test_dataset = PublicDataset(self.test_csv, transform=val_transform)






    def train_dataloader(self):
        
        return torch.utils.data.DataLoader(self.train_dataset, batch_size=self.batch_size, num_workers=self.num_workers, shuffle=True, worker_init_fn=np.random.seed(self.seed))

    def val_dataloader(self):
        return torch.utils.data.DataLoader(self.val_dataset, batch_size=self.batch_size, num_workers=self.num_workers, shuffle=False, worker_init_fn=np.random.seed(self.seed))
        

    def test_dataloader(self):
        
        return torch.utils.data.DataLoader(self.test_dataset, batch_size=self.batch_size, num_workers=self.num_workers, shuffle=False, worker_init_fn=np.random.seed(self.seed))
    



# dry run

if __name__ == "__main__":

    csv_root = '/media/debayan/c7b64c90-ca4e-4192-8ed9-8fea1d005196/SchwanommaDS/UKE/labelled_ds/csv_public_ds_2/DS1/'
    fold = 0
    batch_size = 32
    num_workers = 8
    num_classes = 2
    seed = 42

    datamodule = PublicDatamodule(csv_root, fold, batch_size, num_workers, num_classes, seed)
    datamodule.setup()
    train_loader = datamodule.train_dataloader()
    val_loader = datamodule.val_dataloader()
    test_loader = datamodule.test_dataloader()

    for batch in train_loader:

        print(batch[0].shape, batch[1].shape, batch[2].shape)

        # get unique values in the batch 2 
        print(np.unique(batch[2]))
        break
    