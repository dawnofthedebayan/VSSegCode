import argparse
import torch
import torch.nn as nn
import pytorch_lightning as pl
import wandb
from model_2D import UNet
from losses import SegmentationLosses
from datamodule import UKESegmentationDataModule
import monai
from PIL import Image
import os
import numpy as np
import pandas as pd
# Define a PyTorch Lightning module for training
class DoubleSegmentationLightningModel(pl.LightningModule):
    def __init__(self, model, loss_type="dice", optimizer="adam", lr=1e-3, weight_decay=1e-5, scheduler="cosine", enable_deep_supervision= False , input_type= "t1", num_classes=3 , args =None, **kwargs):
        super().__init__()
        
        self.args  = args
        if args is not None:

            #create group name based on input type, model, deep supervision
            self.group_name = args.input_type + '_' + args.unet_model + '_deep_sup_' + str(args.enable_deep_supervision) + '_num_classes_' + str(args.num_classes) + '_loss_' + args.loss_type + '_concat_' + str(args.concatanate_features) + '_encoder_attention_' + str(args.encoder_attention) + '_dataset_' + args.dataset + "_new"
              #create run name based on fold number
            self.run_name = 'fold_' + str(args.fold)


        self.model = model
        if loss_type == "dice":
            self.loss = SegmentationLosses.dice_loss_multi_class
        elif loss_type == "cross_entropy":
            self.loss = SegmentationLosses.cross_entropy_loss_multi_class
        elif loss_type == "combined":
            self.loss = SegmentationLosses.combined_loss_multi_class
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
        self.encoder_attention = args.encoder_attention

        self.train_metrics, self.val_metrics, self.test_metrics = self.return_metrics()
        


    # Define a function to return metrics
    def return_metrics(self):

        train_metrics = {'dice': [monai.metrics.DiceMetric(include_background=False, reduction="mean") for i in range(self.num_classes)],
                        'hausdorff': [monai.metrics.HausdorffDistanceMetric(include_background=False,percentile=95, reduction="mean") for i in range(self.num_classes)],
                        'confusionmatrix': monai.metrics.ConfusionMatrixMetric(include_background=False, metric_name = ["sensitivity", "specificity", "precision"], reduction="mean"),
                        #'auroc': monai.metrics.ROCAUCMetric( average="mean"),
                        }
        
        val_metrics = {'dice': [monai.metrics.DiceMetric(include_background=False, reduction="mean") for i in range(self.num_classes)],
                        'hausdorff': [monai.metrics.HausdorffDistanceMetric(include_background=False,percentile=95,  reduction="mean") for i in range(self.num_classes)],
                        'confusionmatrix': monai.metrics.ConfusionMatrixMetric(include_background=False, metric_name = ["sensitivity", "specificity", "precision"], reduction="mean"),
                        #'auroc': monai.metrics.ROCAUCMetric( average="mean"),
                        }
        
        test_metrics = {'dice': [monai.metrics.DiceMetric(include_background=False, reduction="mean") for i in range(self.num_classes)],
                        'hausdorff': [monai.metrics.HausdorffDistanceMetric(include_background=False,percentile=95, reduction="mean") for i in range(self.num_classes)],
                        'confusionmatrix': monai.metrics.ConfusionMatrixMetric(include_background=False, metric_name = ["sensitivity", "specificity", "precision"], reduction="mean"),
                        #'auroc': monai.metrics.ROCAUCMetric( average="mean"),
                        }
        

        return train_metrics, val_metrics, test_metrics
                        
    def make_one_hot(self, tensor_mat, num_classes=2, use_argmax=False):

        tensor_mat = tensor_mat.long()
        if use_argmax:
            tensor_mat = torch.argmax(tensor_mat, dim=1)
            
            one_hot = torch.zeros(tensor_mat.size(0), num_classes, tensor_mat.size(1), tensor_mat.size(2)).to(tensor_mat.device)
            target = one_hot.scatter_(1, tensor_mat.unsqueeze(1), 1)
        else:
            one_hot = torch.zeros(tensor_mat.size(0), num_classes, tensor_mat.size(1), tensor_mat.size(2)).to(tensor_mat.device)
            target = one_hot.scatter_(1, tensor_mat.unsqueeze(1), 1)


        return target
    

    def save_image(self, image, mask, predicted_mask,type="train"):


        # Uplaod the t1,t2 image, mask and predicted mask to wandb for visualisation
        
            # upload each image in the batch

        
        for i in range(image.shape[0]):
            
            img_t1 = None
            img_t2 = None

            if len(image.shape) == 4:
                #t1 and t2 image
                img_t1 = image[i,0,:,:].unsqueeze(0)
                img_t1 = img_t1.permute(1,2,0)
                img_t2 = image[i,1,:,:].unsqueeze(0)
                img_t2 = img_t2.permute(1,2,0)
                img_t1 = img_t1.cpu().numpy()
                img_t2 = img_t2.cpu().numpy()
                img_t1 = img_t1.copy()
                img_t2 = img_t2.copy()

                #unnormlize the image from mean 0.5 and std 0.5

                

                # get range of values
                
                
                img_t1 = img_t1 * 0.5 + 0.5
                img_t2 = img_t2 * 0.5 + 0.5

                
                #numpy permute
                
                img_t1 = img_t1 * 255

                img_t1 = np.squeeze(img_t1)

                #convert to int
                img_t1 = img_t1.astype(np.uint8)
                
                img_t2 = img_t2 * 255

                img_t2 = np.squeeze(img_t2)

                #convert to int
                img_t2 = img_t2.astype(np.uint8)

            
                # segmask one hot
                seg_mask_one_hot = self.make_one_hot(mask.detach().clone(), num_classes=self.num_classes)

                
                seg_mask_one_hot = seg_mask_one_hot[i,:,:,:].detach().clone()
                seg_mask_one_hot = seg_mask_one_hot * 255

                seg_mask_one_hot = seg_mask_one_hot.cpu().numpy()
            

                # predicted mask
                pred_mask = predicted_mask[i,:,:,:].detach().clone()
                pred_mask = pred_mask.permute(1,2,0) 
                
            
                # convert into single channel class mask    
                pred_mask_class = torch.argmax(pred_mask, dim=2) 

                pred_mask_class = pred_mask_class.cpu().numpy()

                pred_mask = pred_mask * 255
                pred_mask = pred_mask.cpu().numpy()

                # convert into one hot encoding
                
                if self.num_classes == 3: 
                
                    # put all the images in a single image
                    combined_image = Image.new('RGB', (img_t1.shape[0] * 6, img_t1.shape[1])) 
                    combined_image.paste(Image.fromarray(img_t1), (0,0))
                    combined_image.paste(Image.fromarray(img_t2), (img_t1.shape[0],0))
                    combined_image.paste(Image.fromarray(seg_mask_one_hot[1,:,:]), (img_t1.shape[0] * 2,0))
                    combined_image.paste(Image.fromarray(seg_mask_one_hot[2,:,:]), (img_t1.shape[0] * 3,0))
                    combined_image.paste(Image.fromarray(pred_mask[:,:,1]), (img_t1.shape[0] * 4,0))
                    combined_image.paste(Image.fromarray(pred_mask[:,:,2]), (img_t1.shape[0] * 5,0))
                elif self.num_classes  ==  2 :

                    # put all the images in a single image
                    combined_image = Image.new('RGB', (img_t1.shape[0] * 4, img_t1.shape[1]))
                    combined_image.paste(Image.fromarray(img_t1), (0,0))
                    combined_image.paste(Image.fromarray(img_t2), (img_t1.shape[0],0))
                    combined_image.paste(Image.fromarray(seg_mask_one_hot[1,:,:]), (img_t1.shape[0] * 2,0))
                    combined_image.paste(Image.fromarray(pred_mask[:,:,1]), (img_t1.shape[0] * 3,0))
                                    

            else:
                img = image[i,:,:].unsqueeze(0)

                img = img.permute(1,2,0)
                img = img.cpu().numpy()
                img = img.copy()

                #unnormlize the image from mean 0.5 and std 0.5
                img = img * 0.5 + 0.5

                #squeeze the batch dimension
                img = np.squeeze(img)

                img = img * 255
                # convert to numpy


                # segmask one hot
                seg_mask_one_hot = self.make_one_hot(mask.detach().clone(), num_classes=self.num_classes)
                seg_mask_one_hot = seg_mask_one_hot[i,:,:,:].detach().clone()
                seg_mask_one_hot = seg_mask_one_hot * 255

               

                seg_mask_one_hot = seg_mask_one_hot.cpu().numpy()
            

                # predicted mask
                pred_mask = predicted_mask[i,:,:,:].detach().clone()
                pred_mask = pred_mask.permute(1,2,0) 
                
            
                # convert into single channel class mask    
                pred_mask_class = torch.argmax(pred_mask, dim=2) 

                pred_mask_class = pred_mask_class.cpu().numpy()

                pred_mask = pred_mask * 255
                pred_mask = pred_mask.cpu().numpy()

                

                if self.num_classes == 3:
                    # put all the images in a single image
                    combined_image = Image.new('RGB', (img.shape[0] * 5, img.shape[1])) 
                    combined_image.paste(Image.fromarray(img), (0,0))
                    combined_image.paste(Image.fromarray(seg_mask_one_hot[1,:,:]), (img.shape[0],0))
                    combined_image.paste(Image.fromarray(seg_mask_one_hot[2,:,:]), (img.shape[0] * 2,0))
                    combined_image.paste(Image.fromarray(pred_mask[:,:,1]), (img.shape[0] * 3,0))
                    combined_image.paste(Image.fromarray(pred_mask[:,:,2]), (img.shape[0] * 4,0))

                elif self.num_classes  == 2:

                    # put all the images in a single image
                    combined_image = Image.new('RGB', (img.shape[0] * 3, img.shape[1]))
                    combined_image.paste(Image.fromarray(img), (0,0))
                    combined_image.paste(Image.fromarray(seg_mask_one_hot[1,:,:]), (img.shape[0],0))
                    combined_image.paste(Image.fromarray(pred_mask[:,:,1]), (img.shape[0] * 2,0))


            # save image to local folder with increasing index
            # create directory if it doesn't exist
            os.makedirs("/media/debayan/c7b64c90-ca4e-4192-8ed9-8fea1d005196/SchwanommaDS/UKE/code/inferrence/" + "/" + self.group_name + "/" + type +  "/" , exist_ok=True) 
            combined_image.save("/media/debayan/c7b64c90-ca4e-4192-8ed9-8fea1d005196/SchwanommaDS/UKE/code/inferrence/" + "/" + self.group_name + "/" +  type +  "/" +  self.run_name + "_" + str(i) + "_" +f"{self.global_step}" + ".png")
            
    
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

        
        

        
        predicted_mask, decoder_masks = self.model(image)

        predicted_mask_refined, predicted_mask_course = predicted_mask

        # softmax the output
        predicted_mask_refined = nn.Softmax(dim=1)(predicted_mask_refined) 
        predicted_mask_course = nn.Softmax(dim=1)(predicted_mask_course)
        for i in range(len(decoder_masks)):
            decoder_masks[i] = nn.Softmax(dim=1)(decoder_masks[i])

        
        if type == "test" or type == "val":


            if self.input_type == "t1":

                image = image
                # squeeze the batch dimension
                image = image.squeeze(1)
                self.save_image(image, mask, predicted_mask_refined, type)

            if self.input_type == "t2":

                image = image
                # squeeze the batch dimension
                image = image.squeeze(1)
                self.save_image(image, mask, predicted_mask_refined, type)

            if self.input_type == "t1t2":
                

                image = image
                self.save_image(image, mask, predicted_mask_refined, type)


        return predicted_mask_refined, predicted_mask_course, decoder_masks, mask


    def compute_metrics(self, pred, target, metrics, type="train"):
        
        for metric_name, metric in metrics.items():
            
            if metric_name == "confusionmatrix":
                
                
                pred = (pred > 0.5).float() 
                
                
                #make one hot
                #pred = self.make_one_hot(pred, num_classes=self.num_classes, use_argmax=True) 
                target = self.make_one_hot(target, num_classes=self.num_classes)
                metric(pred, target)


            else:

                target_images = []
                pred_images = []
                for i in range(self.num_classes):
                    
                    # convert into one hot encoding
                    
                    target_one_hot = self.make_one_hot(target, num_classes=self.num_classes) 
                    target_one_hot = target_one_hot[:,i,:,:].unsqueeze(1)

                    #print(target_one_hot.shape, "target one hot shape")
                    """
                    # convert into one hot encoding into png image
                    target_one_hot_png = target_one_hot[0,:,:,:] * 255
                    target_one_hot_png = target_one_hot_png.squeeze(1)
                    target_one_hot_png = target_one_hot_png.permute(1,2,0)
                    target_one_hot_png = target_one_hot_png.cpu().numpy()
                    target_one_hot_png = target_one_hot_png.copy()
                    
                    # squeeze the batch dimension
                    target_one_hot_png = target_one_hot_png.squeeze(2)
                    
                    target_images.append(Image.fromarray(target_one_hot_png))
                    
                    """
                    #binarize the prediction by  making the max value 1 and rest 0
                    

                   


                    pred_one_hot = (pred > 0.5).float() 
                    pred_one_hot = pred_one_hot[:,i,:,:].unsqueeze(1)

                    """

                    # convert into one hot encoding into png image
                    pred_one_hot_png = pred_one_hot[0,:,:,:] * 255
                    pred_one_hot_png = pred_one_hot_png.squeeze(1)
                    pred_one_hot_png = pred_one_hot_png.permute(1,2,0)
                    pred_one_hot_png = pred_one_hot_png.cpu().numpy()
                    pred_one_hot_png = pred_one_hot_png.copy()

                    # squeeze the batch dimension
                    pred_one_hot_png = pred_one_hot_png.squeeze(2)
                    pred_images.append(Image.fromarray(pred_one_hot_png))
                    """
                    #print(pred_one_hot.shape, "pred one hot shape", target_one_hot.shape, "target one hot shape")
                    metric[i](pred_one_hot, target_one_hot)

                """
                # save the target and predicted images into a single image by concatenating them
                combined_image = Image.new('RGB', (target_images[0].width * self.num_classes, target_images[0].height * 2))
                
                for i in range(self.num_classes):
                    combined_image.paste(target_images[i], (target_images[0].width * i,0))
                    combined_image.paste(pred_images[i], (target_images[0].width * i,target_images[0].height))

                # save image to local folder with increasing index

                combined_image.save(f"/media/debayan/c7b64c90-ca4e-4192-8ed9-8fea1d005196/SchwanommaDS/UKE/code/check_images/target_pred_{type}_{self.input_type}_{self.current_epoch}_{self.global_step}.png")
                """


    def aggregate_metrics(self, metrics, prefix = "train"):

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


                

            else:
                
                values = []
                if self.num_classes == 2:
                    classes = ["Background", "SchwannomaCanal"]
                else:
                    classes = ["Background", "SchwannomaCanal", "SchwannomaBrain"]
                for i in range(self.num_classes):
                    metric_value = metric[i].aggregate().item()
                    values.append(metric_value)
                    self.log(f"{prefix}_{metric_name}_{i}", metric_value, on_epoch=True, prog_bar=True, logger=True)
                    #print(f"{prefix}_{metric_name}_{classes[i]}: {metric_value}")
                    # add to dictionary entry
                    dictionary_entry[metric_name + "_" + classes[i]] = [metric_value] 
                    metric[i].reset()



                #remove background class
                values.pop(0)
                metric_value = sum(values) / len(values)

                self.log(f"{prefix}_{metric_name}", metric_value, on_epoch=True, prog_bar=True, logger=True)

            
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

        predicted_t1_mask,predicted_t1_mask_course, decoder_masks, target = self.forward(batch, type="train")
        

        # Compute the loss
        loss_1 = self.loss(predicted_t1_mask, target, num_classes=self.num_classes)
        loss_2 = self.loss(predicted_t1_mask_course, target, num_classes=self.num_classes)
        loss    = loss_1 + loss_2
        if self.enable_deep_supervision:
            for pred in decoder_masks:
                loss += self.loss(pred, target, num_classes=self.num_classes)
            
            loss = loss / (len(decoder_masks) + 2)

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
        predicted_t1_mask,predicted_t1_mask_course, decoder_masks, target = self.forward(batch, type="val")

        
        # Compute the loss

        loss = self.loss(predicted_t1_mask, target, num_classes=self.num_classes)

        if self.enable_deep_supervision:
            for pred in decoder_masks:
                loss += self.loss(pred, target, num_classes=self.num_classes)
            
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
     


        predicted_t1_mask,predicted_t1_mask_course, decoder_masks, target = self.forward(batch, type="test")

        
        # Compute the loss

        loss = self.loss(predicted_t1_mask, target, num_classes=self.num_classes)

        if self.enable_deep_supervision:
            for pred in decoder_masks:
                loss += self.loss(pred, target, num_classes=self.num_classes)
            
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