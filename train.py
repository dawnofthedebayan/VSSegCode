


import argparse
import torch
import torch.nn as nn
import pytorch_lightning as pl
import wandb
from model_2D import UNet, DoubleUNet
from model_3D import UNet3D, DoubleUNet3D
from model_25D import UNet25D, DoubleUNet25D

from datamodule import UKESegmentationDataModule
from datamodule_public_ds import PublicDatamodule
from datamodule_25D import PublicDatamodule25D
from datamodule_public_2D import PublicDSSegmentationDataModule2D
import monai
from pytorch_lightning.loggers import WandbLogger 
from model_pl import SegmentationLightningModel
from model_pl_2 import DoubleSegmentationLightningModel
from model_pl_3D import SegmentationLightningModel3D
from model_pl_3D_2 import DoubleSegmentationLightningModel3D
from model_pl_25D import SegmentationLightningModel25D
from model_pl_25D_2 import DoubleSegmentationLightningModel25D
from model_pl_2D import SegmentationLightningModel2D_public
from model_pl_2D_2 import DoubleSegmentationLightningModel2D_public

from pytorch_lightning.callbacks  import ModelCheckpoint, EarlyStopping
from utils import seed_everything, find_latest_checkpoint


# Main training script
def train(args):
    
    # Set seed for reproducibility
    seed_everything(args.seed)
    

    if args.unet_model == '2D' and (args.input_type == 't1' or args.input_type == 't2'):

        # Initialize data loaders
        lightning_datamodule = UKESegmentationDataModule(args.csv_root + args.dataset + "/", args.fold, args.batch_size, args.num_workers, args.num_classes, args.seed)

        model = UNet(in_channels=1, out_channels=args.num_classes)
        # Initialize PyTorch Lightning model
        lightning_model = SegmentationLightningModel(model, loss_type=args.loss_type, num_classes=args.num_classes, optimizer=args.optimizer, lr=args.lr, weight_decay=args.weight_decay, scheduler=args.scheduler, enable_deep_supervision=args.enable_deep_supervision, input_type=args.input_type, args = args)


    elif args.unet_model == '2D' and args.input_type == 't1t2':

        # Initialize data loaders
        lightning_datamodule = UKESegmentationDataModule(args.csv_root + args.dataset + "/", args.fold, args.batch_size, args.num_workers, args.num_classes, args.seed)


        model = UNet(in_channels=2, out_channels=args.num_classes) 
        # Initialize PyTorch Lightning model
        lightning_model = SegmentationLightningModel(model, loss_type=args.loss_type, num_classes=args.num_classes, optimizer=args.optimizer, lr=args.lr, weight_decay=args.weight_decay, scheduler=args.scheduler, enable_deep_supervision=args.enable_deep_supervision, input_type=args.input_type, args = args)

    
    elif args.unet_model == '2D_double' and (args.input_type == 't1' or args.input_type == 't2'):

        # Initialize data loaders
        lightning_datamodule = UKESegmentationDataModule(args.csv_root + args.dataset + "/", args.fold, args.batch_size, args.num_workers, args.num_classes, args.seed)


        model_1 = UNet(in_channels=1, out_channels=args.num_classes)
        checkpoint = find_latest_checkpoint(args.checkpoint_path,args.fold)


        #load model weights in pytorch lightning model
        lightning_model = SegmentationLightningModel.load_from_checkpoint(checkpoint,model = model_1, loss_type=args.loss_type, num_classes=args.num_classes, optimizer=args.optimizer, lr=args.lr, weight_decay=args.weight_decay, scheduler=args.scheduler, enable_deep_supervision=args.enable_deep_supervision, input_type=args.input_type, args = args)

        print("Model loaded successfully from checkpoint: ", checkpoint)
        unet_model_pretrained = lightning_model.model
        
        # Initialize PyTorch Lightning model
        model = DoubleUNet(in_channels=2, out_channels=args.num_classes, concat_features_encoder= args.concatanate_features , encoder_attention=args.encoder_attention , model_1=unet_model_pretrained)
        lightning_model = DoubleSegmentationLightningModel(model, loss_type=args.loss_type, num_classes=args.num_classes, optimizer=args.optimizer, lr=args.lr, weight_decay=args.weight_decay, scheduler=args.scheduler, enable_deep_supervision=args.enable_deep_supervision, input_type=args.input_type, args = args)
        

    elif args.unet_model == '2D_double' and args.input_type == 't1t2':

        # Initialize data loaders
        lightning_datamodule = UKESegmentationDataModule(args.csv_root + args.dataset + "/", args.fold, args.batch_size, args.num_workers, args.num_classes, args.seed)

            
        model_1 = UNet(in_channels=2, out_channels=args.num_classes)
        checkpoint = find_latest_checkpoint(args.checkpoint_path, args.fold)

        print("Loading model from checkpoint: ", checkpoint)
        
        #load model weights in pytorch lightning model
        lightning_model = SegmentationLightningModel(model_1, loss_type=args.loss_type, num_classes=args.num_classes, optimizer=args.optimizer, lr=args.lr, weight_decay=args.weight_decay, scheduler=args.scheduler, enable_deep_supervision=args.enable_deep_supervision, input_type=args.input_type, args = args)
        lightning_model.load_from_checkpoint(checkpoint,model = model_1, loss_type=args.loss_type, num_classes=args.num_classes, optimizer=args.optimizer, lr=args.lr, weight_decay=args.weight_decay, scheduler=args.scheduler, enable_deep_supervision=args.enable_deep_supervision, input_type=args.input_type, args = args)
        print("Model loaded successfully from checkpoint: ", checkpoint)
        unet_model_pretrained = lightning_model.model
        model = DoubleUNet(in_channels=3, out_channels=args.num_classes,concat_features_encoder= args.concatanate_features, encoder_attention=args.encoder_attention, model_1=unet_model_pretrained)
        
        # Initialize PyTorch Lightning model
        lightning_model = DoubleSegmentationLightningModel(model, loss_type=args.loss_type, num_classes=args.num_classes, optimizer=args.optimizer, lr=args.lr, weight_decay=args.weight_decay, scheduler=args.scheduler, enable_deep_supervision=args.enable_deep_supervision, input_type=args.input_type, args = args)

    

    elif args.unet_model == '2D_public' and (args.input_type == 't1' or args.input_type == 't2'):

        # Initialize data loaders
        lightning_datamodule = PublicDSSegmentationDataModule2D(args.csv_root + args.dataset + "/", args.fold, args.batch_size, args.num_workers, args.num_classes, args.seed,test_batch_size = 50)

        model = UNet(in_channels=1, out_channels=args.num_classes)
        # Initialize PyTorch Lightning model
        lightning_model = SegmentationLightningModel2D_public(model, loss_type=args.loss_type, num_classes=args.num_classes, optimizer=args.optimizer, lr=args.lr, weight_decay=args.weight_decay, scheduler=args.scheduler, enable_deep_supervision=args.enable_deep_supervision, input_type=args.input_type, args = args)


    elif args.unet_model == '2D_public' and args.input_type == 't1t2':

        # Initialize data loaders
        lightning_datamodule = PublicDSSegmentationDataModule2D(args.csv_root + args.dataset + "/", args.fold, args.batch_size, args.num_workers, args.num_classes, args.seed,test_batch_size = 50)


        model = UNet(in_channels=2, out_channels=args.num_classes) 
        # Initialize PyTorch Lightning model
        lightning_model = SegmentationLightningModel2D_public(model, loss_type=args.loss_type, num_classes=args.num_classes, optimizer=args.optimizer, lr=args.lr, weight_decay=args.weight_decay, scheduler=args.scheduler, enable_deep_supervision=args.enable_deep_supervision, input_type=args.input_type, args = args)


    elif args.unet_model == '2D_public_double' and (args.input_type == 't1' or args.input_type == 't2'):

        # Initialize data loaders
        lightning_datamodule = PublicDSSegmentationDataModule2D(args.csv_root + args.dataset + "/", args.fold, args.batch_size, args.num_workers, args.num_classes, args.seed,test_batch_size = 50)


        model_1 = UNet(in_channels=1, out_channels=args.num_classes)
        checkpoint = find_latest_checkpoint(args.checkpoint_path,args.fold)


        #load model weights in pytorch lightning model
        lightning_model = SegmentationLightningModel2D_public.load_from_checkpoint(checkpoint,model = model_1, loss_type=args.loss_type, num_classes=args.num_classes, optimizer=args.optimizer, lr=args.lr, weight_decay=args.weight_decay, scheduler=args.scheduler, enable_deep_supervision=args.enable_deep_supervision, input_type=args.input_type, args = args)

        print("Model loaded successfully from checkpoint: ", checkpoint)
        unet_model_pretrained = lightning_model.model
        
        # Initialize PyTorch Lightning model
        model = DoubleUNet(in_channels=2, out_channels=args.num_classes, concat_features_encoder= args.concatanate_features , encoder_attention=args.encoder_attention , model_1=unet_model_pretrained)
        lightning_model = DoubleSegmentationLightningModel2D_public(model, loss_type=args.loss_type, num_classes=args.num_classes, optimizer=args.optimizer, lr=args.lr, weight_decay=args.weight_decay, scheduler=args.scheduler, enable_deep_supervision=args.enable_deep_supervision, input_type=args.input_type, args = args)
        

    elif args.unet_model == '2D_public_double' and args.input_type == 't1t2':

        # Initialize data loaders
        lightning_datamodule = PublicDSSegmentationDataModule2D(args.csv_root + args.dataset + "/", args.fold, args.batch_size, args.num_workers, args.num_classes, args.seed,test_batch_size = 50)

            
        model_1 = UNet(in_channels=2, out_channels=args.num_classes)
        checkpoint = find_latest_checkpoint(args.checkpoint_path, args.fold)

        print("Loading model from checkpoint: ", checkpoint)
        
        #load model weights in pytorch lightning model
        lightning_model = SegmentationLightningModel2D_public(model_1, loss_type=args.loss_type, num_classes=args.num_classes, optimizer=args.optimizer, lr=args.lr, weight_decay=args.weight_decay, scheduler=args.scheduler, enable_deep_supervision=args.enable_deep_supervision, input_type=args.input_type, args = args)
        lightning_model.load_from_checkpoint(checkpoint,model = model_1, loss_type=args.loss_type, num_classes=args.num_classes, optimizer=args.optimizer, lr=args.lr, weight_decay=args.weight_decay, scheduler=args.scheduler, enable_deep_supervision=args.enable_deep_supervision, input_type=args.input_type, args = args)
        print("Model loaded successfully from checkpoint: ", checkpoint)
        unet_model_pretrained = lightning_model.model
        model = DoubleUNet(in_channels=3, out_channels=args.num_classes,concat_features_encoder= args.concatanate_features, encoder_attention=args.encoder_attention, model_1=unet_model_pretrained)
        
        # Initialize PyTorch Lightning model
        lightning_model = DoubleSegmentationLightningModel2D_public(model, loss_type=args.loss_type, num_classes=args.num_classes, optimizer=args.optimizer, lr=args.lr, weight_decay=args.weight_decay, scheduler=args.scheduler, enable_deep_supervision=args.enable_deep_supervision, input_type=args.input_type, args = args)

    


    elif args.unet_model == '25D' and (args.input_type == 't1' or args.input_type == 't2'):

        print("25D model")
        # initialize data loaders
        lightning_datamodule = PublicDatamodule25D(args.csv_root + args.dataset + "/", args.fold, args.batch_size, args.num_workers, args.num_classes, args.seed)

        model = UNet25D(in_channels=50, out_channels=50)
        # Initialize PyTorch Lightning model

        lightning_model = SegmentationLightningModel25D(model, loss_type=args.loss_type, num_classes=args.num_classes, optimizer=args.optimizer, lr=args.lr, weight_decay=args.weight_decay, scheduler=args.scheduler, enable_deep_supervision=args.enable_deep_supervision, args = args, input_type=args.input_type)
    

    elif args.unet_model == '25D' and args.input_type == 't1t2':

        print("25D model")
        # initialize data loaders
        lightning_datamodule = PublicDatamodule25D(args.csv_root + args.dataset + "/", args.fold, args.batch_size, args.num_workers, args.num_classes, args.seed)

        model = UNet25D(in_channels=100, out_channels=50)
        # Initialize PyTorch Lightning model

        lightning_model = SegmentationLightningModel25D(model, loss_type=args.loss_type, num_classes=args.num_classes, optimizer=args.optimizer, lr=args.lr, weight_decay=args.weight_decay, scheduler=args.scheduler, enable_deep_supervision=args.enable_deep_supervision, args = args, input_type=args.input_type)
        

    elif args.unet_model == '25D_double' and (args.input_type == 't1' or args.input_type == 't2'):

        print("25D model")
        # initialize data loaders
        lightning_datamodule = PublicDatamodule25D(args.csv_root + args.dataset + "/", args.fold, args.batch_size, args.num_workers, args.num_classes, args.seed)

        model_1 = UNet25D(in_channels=50, out_channels=50)
        checkpoint = find_latest_checkpoint(args.checkpoint_path,args.fold)

        print("Loading model from checkpoint: ", checkpoint)

        #load model weights in pytorch lightning model
        lightning_model = SegmentationLightningModel25D.load_from_checkpoint(checkpoint, model = model_1, loss_type=args.loss_type, num_classes=args.num_classes, optimizer=args.optimizer, lr=args.lr, weight_decay=args.weight_decay, scheduler=args.scheduler, args = args, input_type=args.input_type)
        print("Model loaded successfully from checkpoint: ", checkpoint)
        unet_model_pretrained = lightning_model.model
        model = DoubleUNet25D(in_channels=100, out_channels=50, model_1=unet_model_pretrained)
        # Initialize PyTorch Lightning model
        lightning_model = DoubleSegmentationLightningModel25D(model, loss_type=args.loss_type, num_classes=args.num_classes, optimizer=args.optimizer, lr=args.lr, weight_decay=args.weight_decay, scheduler=args.scheduler,args = args, input_type=args.input_type)

    elif args.unet_model == '25D_double' and args.input_type == 't1t2':

        print("25D model")
        # initialize data loaders
        lightning_datamodule = PublicDatamodule25D(args.csv_root + args.dataset + "/", args.fold, args.batch_size, args.num_workers, args.num_classes, args.seed)

        model_1 = UNet25D(in_channels=100, out_channels=50)
        checkpoint = find_latest_checkpoint(args.checkpoint_path,args.fold)

        #load model weights in pytorch lightning model
        lightning_model = SegmentationLightningModel25D.load_from_checkpoint(checkpoint, model = model_1, loss_type=args.loss_type, num_classes=args.num_classes, optimizer=args.optimizer, lr=args.lr, weight_decay=args.weight_decay, scheduler=args.scheduler, args = args, input_type=args.input_type)
        print("Model loaded successfully from checkpoint: ", checkpoint)
        unet_model_pretrained = lightning_model.model
        model = DoubleUNet25D(in_channels=150, out_channels=50, model_1=unet_model_pretrained)
        # Initialize PyTorch Lightning model
        lightning_model = DoubleSegmentationLightningModel25D(model, loss_type=args.loss_type, num_classes=args.num_classes, optimizer=args.optimizer, lr=args.lr, weight_decay=args.weight_decay, scheduler=args.scheduler,args = args, input_type=args.input_type)

    elif args.unet_model == '3D' and (args.input_type == 't1' or args.input_type == 't2'):


        # Initialize data loaders
        lightning_datamodule = PublicDatamodule(args.csv_root + args.dataset + "/", args.fold, args.batch_size, args.num_workers, args.num_classes, args.seed)
        model = UNet3D(in_channels=1, out_channels=args.num_classes)
        # Initialize PyTorch Lightning model
        lightning_model = SegmentationLightningModel3D(model, loss_type=args.loss_type, num_classes=args.num_classes, optimizer=args.optimizer, lr=args.lr, weight_decay=args.weight_decay, scheduler=args.scheduler, enable_deep_supervision=args.enable_deep_supervision, input_type=args.input_type, args = args)

        
    elif args.unet_model == '3D_double' and (args.input_type == 't1' or args.input_type == 't2'):

        # Initialize data loaders
        lightning_datamodule = PublicDatamodule(args.csv_root + args.dataset + "/", args.fold, args.batch_size, args.num_workers, args.num_classes, args.seed)
        checkpoint = find_latest_checkpoint(args.checkpoint_path,args.fold)

        model = SegmentationLightningModel3D.load_from_checkpoint(checkpoint, model = UNet3D(in_channels=1, out_channels=args.num_classes), loss_type=args.loss_type, num_classes=args.num_classes, optimizer=args.optimizer, lr=args.lr, weight_decay=args.weight_decay, scheduler=args.scheduler, enable_deep_supervision=args.enable_deep_supervision, input_type=args.input_type, args = args)
        
        print("Model loaded successfully from checkpoint: ", checkpoint)
        unet_model_pretrained = model.model
        model = DoubleUNet3D(in_channels=2, out_channels=args.num_classes, model_1=unet_model_pretrained)
        # Initialize PyTorch Lightning model
        lightning_model = DoubleSegmentationLightningModel3D(model, loss_type=args.loss_type, num_classes=args.num_classes, optimizer=args.optimizer, lr=args.lr, weight_decay=args.weight_decay, scheduler=args.scheduler, enable_deep_supervision=args.enable_deep_supervision, input_type=args.input_type, args = args)






    
    wandb_logger = None

    if args.unet_model == '2D_double' or args.unet_model == '3D_double' or  args.unet_model == '25D_double' or args.unet_model == '2D_public_double':
        group_name = args.input_type + '_' + args.unet_model + '_deep_sup_' + str(args.enable_deep_supervision) + '_num_classes_' + str(args.num_classes) + '_loss_' + args.loss_type + '_concat_' + str(args.concatanate_features) + '_encoder_attention_' + str(args.encoder_attention) + '_dataset_' + args.dataset  + "_new"
    

    else:
        group_name = args.input_type + '_' + args.unet_model + '_deep_sup_' + str(args.enable_deep_supervision) + '_num_classes_' + str(args.num_classes) + '_loss_' + args.loss_type + '_dataset_' + args.dataset + "_new"
    run_name = 'fold_' + str(args.fold)




    
    if args.enable_wandb:
        # Initialize Wandb logger

        #create group name based on input type, model, deep supervision
        
        #create run name based on fold number
        
        wandb_logger = WandbLogger(project=args.wandb_project, group=group_name, name=run_name, log_model=True) 

    # Initialise callbacks
    checkpoint_callback = ModelCheckpoint(
        monitor='val_loss',
        dirpath=f'checkpoints/{group_name}',
        filename= f'model-{run_name}',
        save_top_k=1,
        mode='min',
    )

    early_stop_callback = EarlyStopping(
        monitor='val_loss',
        patience=2,
        verbose=True,
        mode='min'
    )

    # Training loop using PyTorch Lightning Trainer
    trainer = pl.Trainer(

        devices=args.gpus,
        accelerator='gpu',
        max_epochs=args.max_epochs,
        log_every_n_steps=args.log_every_n_steps,
        logger=wandb_logger,
        callbacks=[early_stop_callback, checkpoint_callback] #checkpoint_callback, 

    )


    trainer.fit(lightning_model, lightning_datamodule)

    # Test the best model on the test set
    
    trainer.test(lightning_model,  ckpt_path= "best", datamodule=lightning_datamodule)

    

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Segmentation Training Script")
    # Add argparse configurations for model, data directories, hyperparameters, etc.


    parser.add_argument('--optimizer', default='adam', choices=['adam', 'sgd'], help='Optimizer (adam or sgd)')
    parser.add_argument('--lr', default=1e-3, type=float, help='Learning rate')
    parser.add_argument('--weight_decay', default=1e-5, type=float, help='Weight decay')
    parser.add_argument('--scheduler', default='cosine', choices=['cosine', 'step'], help='Learning rate scheduler (cosine or step)')
    parser.add_argument('--loss_type', default='combined', choices=['dice', 'cross_entropy', 'combined'], help='Loss function (dice, cross_entropy, or combined)')
    parser.add_argument('--batch_size', default=8, type=int, help='Batch size')
    parser.add_argument('--max_epochs', default=100, type=int, help='Maximum number of epochs')
    parser.add_argument('--log_every_n_steps', default=1, type=int, help='Log every n steps')
    parser.add_argument('--gpus', default=1, type=int, help='Number of GPUs')
    parser.add_argument('--fold', default=0, type=int, help='Fold number') # 0, 1, 2, 3, 4
    parser.add_argument('--csv_root', default='/media/debayan/c7b64c90-ca4e-4192-8ed9-8fea1d005196/SchwanommaDS/UKE/labelled_ds/csv/', type=str, help='Path to csv files')
    parser.add_argument('--dataset', default='UKE', choices=['DS1', 'DS2', 'DS3','DS4','DS5','DS6'], type=str, help='Dataset name')
    parser.add_argument('--enable_wandb', action='store_true', help='Enable wandb logging') 
    parser.add_argument('--input_type', default='t1t2', type=str, help='Input modality (t1, t2, t1t2)')
    parser.add_argument('--num_workers', default=8, type=int, help='Number of workers for dataloader') 
    parser.add_argument('--unet_model', default= '2D', choices=['2D', '3D', '2D_double','3D_double','25D','25D_double', '2D_public', '2D_public_double' ], type=str, help='Unet model (2D, 3D, 2D_double)')
    parser.add_argument('--concatanate_features',action='store_true', help='Concatanate features in 2D_double model')
    parser.add_argument('--enable_deep_supervision', action='store_true', help='Enable deep supervision')
    parser.add_argument('--wandb_project', default='VS', type=str, help='Wandb project name')
    parser.add_argument('--num_classes', default=2, type=int, help='Number of classes')
    parser.add_argument('--seed', default=42, type=int, help='Seed')
    parser.add_argument('--checkpoint_path', default=None, type=str, help='Path to checkpoint')
    parser.add_argument('--encoder_attention', default=2, type=int, choices=[1,2], help='Apply attention to encoder of second UNET')
    
    args = parser.parse_args()
    train(args)