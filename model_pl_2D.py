import argparse
import torch
import torch.nn as nn
import pytorch_lightning as pl
import wandb
from model_2D import UNet
from losses import DiceBCELoss3D
from datamodule_public_ds import PublicDatamodule

from metrics import relative_volume_error

import monai
from PIL import Image
import os
import numpy as np
import pandas as pd
import nibabel as nib

# Define a PyTorch Lightning module for training
class SegmentationLightningModel2D_public(pl.LightningModule):
    def __init__(self, model, loss_type="dice", optimizer="adam", lr=1e-3, weight_decay=1e-5, scheduler="cosine", enable_deep_supervision= False , input_type= "t1", num_classes=3 , args =None, **kwargs):
        super().__init__()
        
        self.args  = args
        if args is not None:

            #create group name based on input type, model, deep supervision
            self.group_name = args.input_type + '_' + args.unet_model + '_deep_sup_' + str(args.enable_deep_supervision) + '_num_classes_' + str(args.num_classes) + '_loss_' + args.loss_type + '_dataset_' + args.dataset 
            #create run name based on fold number
            self.run_name = 'fold_' + str(args.fold)


        self.model = model
        if loss_type == "dice":
            self.loss = DiceBCELoss3D(weight_dice=1.0, weight_bce=0, smooth=1e-5)
        elif loss_type == "cross_entropy":
            self.loss = DiceBCELoss3D(weight_dice=0, weight_bce=1.0, smooth=1e-5)
        elif loss_type == "combined":
            self.loss = DiceBCELoss3D(weight_dice=1.0, weight_bce=1.0, smooth=1e-5)
        else:
            raise ValueError(f"Invalid loss type: {loss_type}")
        
        if optimizer == "adam":

            self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr, weight_decay=weight_decay)
        elif optimizer == "sgd":
            self.optimizer = torch.optim.SGD(self.model.parameters(), lr=lr, weight_decay=weight_decay)
        else:
            raise ValueError(f"Invalid optimizer: {optimizer}")
    
        if scheduler == "cosine":
            self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=10)
        elif scheduler == "step":
            self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=10, gamma=0.1)
        else:
            raise ValueError(f"Invalid scheduler: {scheduler}")
        

        # define metrics 
        self.enable_deep_supervision = enable_deep_supervision
        self.input_type = input_type
        self.num_classes = num_classes

        self.test_counter = 0

        self.train_metrics, self.val_metrics, self.test_metrics = self.return_metrics()
        


    # Define a function to return metrics
    def return_metrics(self):

        train_metrics = {'dice': [monai.metrics.DiceMetric(include_background=False, reduction="mean") for i in range(self.num_classes)],
                        'hausdorff': [monai.metrics.HausdorffDistanceMetric(include_background=False,percentile=95, reduction="mean") for i in range(self.num_classes)],
                        'confusionmatrix': monai.metrics.ConfusionMatrixMetric(include_background=False, metric_name = ["sensitivity", "specificity", "precision"], reduction="mean"),
                        'assd':monai.metrics.SurfaceDistanceMetric(include_background=False, symmetric=True, distance_metric='euclidean', reduction="mean", get_not_nans=False),
                        'rve': []
                        }
        
        val_metrics = {'dice': [monai.metrics.DiceMetric(include_background=False, reduction="mean") for i in range(self.num_classes)],
                        'hausdorff': [monai.metrics.HausdorffDistanceMetric(include_background=False,percentile=95,  reduction="mean") for i in range(self.num_classes)],
                        'confusionmatrix': monai.metrics.ConfusionMatrixMetric(include_background=False, metric_name = ["sensitivity", "specificity", "precision"], reduction="mean"),
                        'assd':monai.metrics.SurfaceDistanceMetric(include_background=False, symmetric=True, distance_metric='euclidean', reduction="mean", get_not_nans=False),
                        'rve': []
                        }
        
        test_metrics = {
                        'dice': [monai.metrics.DiceMetric(include_background=False, reduction="mean") for i in range(self.num_classes)],
                        'hausdorff': [monai.metrics.HausdorffDistanceMetric(include_background=False,percentile=95, reduction="mean") for i in range(self.num_classes)],
                        'confusionmatrix': monai.metrics.ConfusionMatrixMetric(include_background=False, metric_name = ["sensitivity", "specificity", "precision"], reduction="mean"),
                        'assd':monai.metrics.SurfaceDistanceMetric(include_background=False, symmetric=True, distance_metric='euclidean', reduction="mean", get_not_nans=False),
                        'rve': []
                        }
        

        return train_metrics, val_metrics, test_metrics
                        
    def make_one_hot(self, tensor_mat, num_classes=2, use_argmax=False):

        tensor_mat = tensor_mat.long()
        # make one hot for 3D tensor of shape B,C,H,W,D

        if use_argmax:
            tensor_mat = torch.argmax(tensor_mat, dim=1)


        # Create an empty one-hot tensor
        one_hot_tensor = np.zeros((tensor_mat.shape[0], num_classes + 1, tensor_mat.shape[2], tensor_mat.shape[3], tensor_mat.shape[4]))

        # Convert the numpy array to a tensor
        one_hot_tensor = torch.from_numpy(one_hot_tensor).float()
        # cuda tensor
        one_hot_tensor = one_hot_tensor.cuda()


        # Set the first channel to 1 where the original tensor has 0s
        one_hot_tensor[:, 0, :, :, :][tensor_mat[:,0,:,:,:] == 0] = 1

        # Set the second channel to 1 where the original tensor has 1s
        one_hot_tensor[:, 1, :, :, :][tensor_mat[:,0,:,:,:] == 1] = 1

        

        return one_hot_tensor
    

    def save_image(self, image, mask, predicted_mask,type="train", test_counter=0):


        # Uplaod the t1,t2 image, mask and predicted mask to wandb for visualisation
        
            # upload each image in the batch

        #print("inside save image", image.shape, mask.shape, predicted_mask.shape)

        # create save directory if it doesn't exist
        os.makedirs("/media/debayan/c7b64c90-ca4e-4192-8ed9-8fea1d005196/SchwanommaDS/UKE/code/inferrence/" + "/" + self.group_name + "/" + type +  "/org_img/" , exist_ok=True)
        os.makedirs("/media/debayan/c7b64c90-ca4e-4192-8ed9-8fea1d005196/SchwanommaDS/UKE/code/inferrence/" + "/" + self.group_name + "/" + type +  "/gt_mask/" , exist_ok=True)
        os.makedirs("/media/debayan/c7b64c90-ca4e-4192-8ed9-8fea1d005196/SchwanommaDS/UKE/code/inferrence/" + "/" + self.group_name + "/" + type +  "/predicted_mask/" , exist_ok=True)
        
        # covert to numpy array
        image = image.cpu().numpy()
        mask = mask.cpu().numpy()

        predicted_mask = torch.sigmoid(predicted_mask)
        predicted_mask = predicted_mask.cpu().numpy()

        mask = np.expand_dims(mask, axis=1)


        image = np.transpose(image, (1,2,3,0))
        mask = np.transpose(mask, (1,2,3,0))
        predicted_mask = np.transpose(predicted_mask, (1,2,3,0))

        # add C dimension 
        image = np.expand_dims(image, axis=1)
        mask = np.expand_dims(mask, axis=1)
        predicted_mask = np.expand_dims(predicted_mask, axis=1)


        
            
        #  image 
        image_ = image[0,:,:,:] 

        # change value range suitable for nibabel
        image_ = image_ * 255

        # squeeze the channel dimension
        image_ = image_.squeeze(0)

        # save nib image 
        image_ = nib.Nifti1Image(image_, np.eye(4)) 
        nib.save(image_, "/media/debayan/c7b64c90-ca4e-4192-8ed9-8fea1d005196/SchwanommaDS/UKE/code/inferrence/" + "/" + self.group_name + "/" + type +  "/org_img/" + self.run_name + "_" + str(test_counter)  + "_image.nii.gz")


        # mask
        mask_ = mask[0,:,:,:]
        # change value range suitable for nibabel
        mask_ = mask_ * 255

        # squeeze the channel dimension
        mask_ = mask_.squeeze(0)

        # save nib image
        mask_ = nib.Nifti1Image(mask_, np.eye(4))

        nib.save(mask_, "/media/debayan/c7b64c90-ca4e-4192-8ed9-8fea1d005196/SchwanommaDS/UKE/code/inferrence/" + "/" + self.group_name + "/" + type +  "/gt_mask/" + self.run_name + "_" + str(0) + "_"  + str(test_counter) + "_mask.nii.gz")

        predicted_mask 
        # predicted mask
        predicted_mask_ = predicted_mask[0,:,:,:]
        # change value range suitable for nibabel
        predicted_mask_ = predicted_mask_ * 255

        # squeeze the channel dimension

        predicted_mask_ = predicted_mask_.squeeze(0)

        # save nib image
        predicted_mask_ = nib.Nifti1Image(predicted_mask_, np.eye(4))
        
        nib.save(predicted_mask_, "/media/debayan/c7b64c90-ca4e-4192-8ed9-8fea1d005196/SchwanommaDS/UKE/code/inferrence/" + "/" + self.group_name + "/" + type +  "/predicted_mask/" + self.run_name + "_"  + str(test_counter)  + "_predicted_mask.nii.gz")
        
        test_counter += 1
            # save image to local folder with increasing index
            # create directory if it doesn't exist
        #os.makedirs("/media/debayan/c7b64c90-ca4e-4192-8ed9-8fea1d005196/SchwanommaDS/UKE/code/inferrence/" + "/" + self.group_name + "/" + type +  "/" , exist_ok=True) 
        # combined_image.save("/media/debayan/c7b64c90-ca4e-4192-8ed9-8fea1d005196/SchwanommaDS/UKE/code/inferrence/" + "/" + self.group_name + "/" +  type +  "/" +  self.run_name + "_" + str(i) + "_" +f"{self.global_step}" + ".png")
        self.test_counter = test_counter
    
    
    def forward(self, batch, type="train"):


       
        

        if self.input_type == "t1":

            image = batch["image_t1"]
            mask = batch["mask_t1"]

   
            

        elif self.input_type == "t2":

            image = batch["image_t2"]
            mask = batch["mask_t2"]



        elif self.input_type == "t1t2":
        
            
            image = batch["image_t1t2"]
            mask = batch["mask_t1t2"]




        else:
            raise ValueError(f"Invalid input type: {self.input_type}")

        
        

        
        predicted_mask, decoder_masks,_ = self.model(image)

        

        

        if type == "test" :


            if self.input_type == "t1":

                image = image
                # squeeze the batch dimension
                self.save_image(image, mask, predicted_mask, type, test_counter=self.test_counter)

            if self.input_type == "t2":

                image = image
                # squeeze the batch dimension
                self.save_image(image, mask, predicted_mask, type, test_counter=self.test_counter)

            if self.input_type == "t1t2":
                

                image = image
                self.save_image(image, mask, predicted_mask, type, test_counter=self.test_counter)


        return predicted_mask, decoder_masks, mask


    def compute_metrics(self, pred, target, metrics, type="train"):

        if type == "test" or type == "val":
            
            #print(pred.shape, "pred shape", target.shape, "target shape")
            # permute pred and target to shape B,H,W,D 
            pred = pred.permute(1,2,3,0)
            target = target.unsqueeze(1)
            target = target.permute(1,2,3,0)

            # add C dimension 
            pred = pred.unsqueeze(1)
            target = target.unsqueeze(1)




        
            for metric_name, metric in metrics.items():
                
                if metric_name == "confusionmatrix":
                    
                    pred_metric = torch.sigmoid(pred)
                    pred_metric = (pred_metric > 0.5).float() 
                    
                    
                    #make one hot
                    #pred = self.make_one_hot(pred, num_classes=self.num_classes, use_argmax=True) 
                    target = self.make_one_hot(target, num_classes=self.num_classes)
                    metric(pred_metric, target)


                elif metric_name == "assd":
                    
                    pred_metric = torch.sigmoid(pred)
                    pred_metric = (pred_metric > 0.5).float() 
                    
                    #make one hot
                    #pred = self.make_one_hot(pred, num_classes=self.num_classes, use_argmax=True) 
                    target = self.make_one_hot(target, num_classes=self.num_classes)

                    metric(pred_metric, target)

                elif metric_name == "rve":

                    pred_metric = torch.sigmoid(pred)
                    pred_metric = (pred_metric > 0.5).float() 
                    
                    #make one hot
                    #pred = self.make_one_hot(pred, num_classes=self.num_classes, use_argmax=True) 
                    metric = metric + relative_volume_error(pred_metric, target) 
                    metrics[metric_name] = metric

                else:

                    
                    for i in range(self.num_classes):

                        
                        # convert into one hot encoding
                        target_one_hot = self.make_one_hot(target, num_classes=self.num_classes) 
                        target_one_hot = target_one_hot[:,i+1,:,:,:].unsqueeze(1)

                        #print(target_one_hot.shape, "target one hot shape")
                        #binarize the prediction by  making the max value 1 and rest 0
                        pred_metric = torch.sigmoid(pred)
                        pred_metric = (pred_metric > 0.5).float() 
                        pred_metric = pred_metric[:,i,:,:,:].unsqueeze(1)
                    
                        #print(pred_one_hot.shape, "pred one hot shape", target_one_hot.shape, "target one hot shape")
                        metric[i](pred_metric, target_one_hot)




    def aggregate_metrics(self, metrics, prefix = "train"):

        if prefix == "val" or prefix == "test":
            # create a pandas dataframe with following columns: Group, Run, Fold, Precision, specificity, Sensitivity, Dice, Hausdorff
            dictionary_entry = {"Group": self.group_name, "Run": self.run_name, "Fold": self.args.fold, "prefix": prefix,"dataset": self.args.dataset}

            # Log the metrics
            for metric_name, metric in metrics.items():
                
                if metric_name == "confusionmatrix":
                    metric_value = metric.aggregate()
                    conf_mat_names = ["sensitivity", "specificity", "precision"]
                    for conf_mat_name in conf_mat_names:
                        self.log(f"{prefix}_{metric_name}_{conf_mat_name}", metric_value[i].item(), on_epoch=True, prog_bar=True, logger=True)

                        # add to dictionary entry
                        dictionary_entry[conf_mat_name] = [metric_value[i].item()]

                    metric.reset()

                
                elif metric_name == "assd":

                    metric_value = metric.aggregate()
                    self.log(f"{prefix}_{metric_name}", metric_value.item(), on_epoch=True, prog_bar=True, logger=True)
                    # add to dictionary entry
                    dictionary_entry[metric_name] = [metric_value.item()]
                    metric.reset()

                elif metric_name == "rve":
                        
                    metric_value = sum(metric) / len(metric)
                
                    self.log(f"{prefix}_{metric_name}", metric_value, on_epoch=True, prog_bar=True, logger=True)
                    # add to dictionary entry
                    dictionary_entry[metric_name] = [metric_value]
                    metric =  []

                    metrics[metric_name] = metric
                    

                else:
                    
                    values = []
                    
                    classes = ["Schwannoma"]
                    for i in range(self.num_classes):
                        metric_value = metric[i].aggregate().item()

                        values.append(metric_value)
                        self.log(f"{prefix}_{metric_name}_{i}", metric_value, on_epoch=True, prog_bar=True, logger=True)
                        #print(f"{prefix}_{metric_name}_{classes[i]}: {metric_value}")
                        # add to dictionary entry
                        dictionary_entry[metric_name + "_" + classes[i]] = [metric_value] 
                        metric[i].reset()



                    metric_value = sum(values) 

                    self.log(f"{prefix}_{metric_name}", metric_value, on_epoch=True, prog_bar=True, logger=True)

            if prefix == "val" :

                # print the dictionary entry
                print(dictionary_entry)
            if prefix == "test":
                # save the dataframe to a csv file and append to the existing csv file
                df = pd.DataFrame.from_dict(dictionary_entry)
                
                # if file doesn't exist write header

                if not os.path.isfile("/media/debayan/c7b64c90-ca4e-4192-8ed9-8fea1d005196/SchwanommaDS/UKE/code/results/" + f"{self.group_name}" + ".csv"):
                    df.to_csv("/media/debayan/c7b64c90-ca4e-4192-8ed9-8fea1d005196/SchwanommaDS/UKE/code/results/" + f"{self.group_name}" + ".csv", index=False)
                else:
                    df.to_csv("/media/debayan/c7b64c90-ca4e-4192-8ed9-8fea1d005196/SchwanommaDS/UKE/code/results/" + f"{self.group_name}" + ".csv", mode='a', header=False, index=False)

        


        
    def training_step(self, batch, batch_idx):
        # Training step logicpip i

        predicted_t1_mask, decoder_masks, target = self.forward(batch)

        # Compute the loss
        loss = self.loss(predicted_t1_mask, target)
        if self.enable_deep_supervision:
            for pred in decoder_masks:

                loss += self.loss(pred, target)
            
            loss = loss / (len(decoder_masks) + 1)

            
        # Compute the metrics
        self.compute_metrics(predicted_t1_mask, target, self.train_metrics, type="train")
        

        # Log the loss and metrics
        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        # log learning rate
        self.log("lr", self.optimizer.param_groups[0]['lr'], on_step=True, on_epoch=False, prog_bar=True, logger=True)
        


        return loss
    
    def on_train_epoch_end(self):

        # Log the metrics
        self.aggregate_metrics(self.train_metrics, prefix = "train")
        


    def validation_step(self, batch, batch_idx):
        # Validation step logic
        predicted_t1_mask, decoder_masks, target = self.forward(batch, type="val")

        
        # Compute the loss
        loss = self.loss(predicted_t1_mask, target)

        if self.enable_deep_supervision:
            for pred in decoder_masks:
                loss += self.loss(pred, target)
            
            loss = loss / (len(decoder_masks) + 1)

        # Compute the metrics

        self.compute_metrics(predicted_t1_mask, target, self.val_metrics, type="val")


        # Log the loss and metrics
        self.log("val_loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)

        return loss
    
    def on_validation_epoch_end(self):

        # Log the metrics
        self.aggregate_metrics(self.val_metrics, prefix = "val")
        

    def test_step(self, batch, batch_idx):
        # Test step logic
     


        predicted_t1_mask, decoder_masks, target = self.forward(batch, type="test")

        
        # Compute the loss

        loss = self.loss(predicted_t1_mask, target)

        if self.enable_deep_supervision:
            for pred in decoder_masks:
                loss += self.loss(pred, target)
            
            loss = loss / (len(decoder_masks) + 1)

        # Compute the metrics

        self.compute_metrics(predicted_t1_mask, target, self.test_metrics, type="test")


        

        # Log the loss and metrics
        self.log("test_loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)

        return loss
    
    def on_test_epoch_end(self):

        # Log the metrics
        self.aggregate_metrics(self.test_metrics, prefix = "test")
        

    def configure_optimizers(self):
        # Define your optimizer here

        return [self.optimizer], [self.scheduler]