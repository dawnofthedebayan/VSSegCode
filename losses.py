import torch
import torch.nn as nn
import torch.nn.functional as F

class SegmentationLosses:
    @staticmethod
    def dice_loss_multi_class(predicted, target, num_classes, smooth=1.0):
        loss = 0
        #check if softmax is applied
        for class_index in range(num_classes):
            predicted_class = predicted[:, class_index, ...]
           
            target_class = (target == class_index).float()

            intersection = (predicted_class * target_class).sum()
            union = predicted_class.sum() + target_class.sum()

            class_dice = (2.0 * intersection + smooth) / (union + smooth)
            loss -= torch.log(class_dice)


        return loss / num_classes

    @staticmethod
    def cross_entropy_loss_multi_class(predicted, target):
        loss = F.cross_entropy(predicted, target, reduction='mean')
        return loss

    @staticmethod
    def combined_loss_multi_class(predicted, target, num_classes, alpha=0.5):
        dice = SegmentationLosses.dice_loss_multi_class(predicted, target, num_classes)
        ce = SegmentationLosses.cross_entropy_loss_multi_class(predicted, target.long())
        loss = alpha * ce + (1 - alpha) * dice

        #print(f"Dice Loss: {dice.item()}")
        return loss



class DiceBCELoss3D(nn.Module):
    def __init__(self, weight_dice=1.0, weight_bce=1.0, smooth=1e-5):
        super(DiceBCELoss3D, self).__init__()
        self.weight_dice = weight_dice
        self.weight_bce = weight_bce
        self.smooth = smooth

    def forward(self, input, target):
        # Flatten both the input and target tensors

        #print("inside dicebceloss3d", input.shape, target.shape)
        #print("get unique values in target", torch.unique(target), "get unique values in input", torch.unique(input))
        # sigmoid the input
        #print(input.shape, target.shape, "input and target shapes")
        #print("range of input", torch.min(input), torch.max(input))
        input = torch.sigmoid(input)

        #print("range of input after sigmoid", torch.min(input), torch.max(input))
        input = input.view(-1)
        target = target.view(-1)



        # Compute Dice Loss
        intersection = (input * target).sum()
        dice_loss = (2.0 * intersection + self.smooth) / (input.sum() + target.sum() + self.smooth)

        #input = torch.sigmoid(input)
        # Compute BCE Loss
        bce_loss = nn.BCELoss()(input, target)

        # Combine Dice and BCE Loss
        combined_loss = self.weight_dice * (1 - dice_loss) + self.weight_bce * bce_loss

        return combined_loss

"""
# Example usage:
if __name__ == "__main__":
    num_classes = 5  # Change this to the number of classes in your segmentation task

    # Example tensors (replace with your data)
    predicted_logits = torch.randn(4, num_classes, 256, 256)  # Batch size of 4, 256x256 prediction map
    target_labels = torch.randint(0, num_classes, (4, 256, 256))  # Ground truth class labels

    # Create an instance of SegmentationLosses
    losses = SegmentationLosses()

    # Calculate the loss using different loss functions
    dice_loss = losses.dice_loss_multi_class(predicted_logits, target_labels, num_classes)
    ce_loss = losses.cross_entropy_loss_multi_class(predicted_logits, target_labels)
    combined_loss = losses.combined_loss_multi_class(predicted_logits, target_labels, num_classes)

    # Print the losses
    print(f"Dice Loss: {dice_loss.item()}")
    print(f"Cross-Entropy Loss: {ce_loss.item()}")
    print(f"Combined Loss: {combined_loss.item()}")
"""