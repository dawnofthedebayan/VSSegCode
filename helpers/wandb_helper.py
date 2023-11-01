import wandb 


def upload_image(image,pred,gt,name):
    
    wandb.log({name:[wandb.Image(image, caption="Input"), wandb.Image(pred, caption="Prediction"), wandb.Image(gt, caption="Ground Truth")]})