import torch


def relative_volume_error(pred, target, class_index=1):
    """
    Calculate the Relative Volume Error (RVE) per image for 3D segmentation tasks with PyTorch tensors.

    Args:
        pred (torch.Tensor): Predicted segmentation tensor of shape (B, 1, H, W, D).
        target (torch.Tensor): Ground truth segmentation tensor of shape (B, 1, H, W, D).
        class_index (int): Index of the class for which to calculate RVE (default is 1 for binary segmentation).

    Returns:
        torch.Tensor: Tensor of RVE values, one per image in the batch.
    """
    # Ensure that the input tensors are binary (0 or 1)
    assert torch.all(torch.logical_or(pred == 0, pred == 1)), "Predicted segmentation should be binary (0 or 1)."
    assert torch.all(torch.logical_or(target == 0, target == 1)), "Ground truth segmentation should be binary (0 or 1)."

    rve = []


    for i in range(pred.shape[0]):

        Vp = torch.sum(pred[i, 0, ...])
        Vg = torch.sum(target[i, 0, ...])

        rve_value = torch.abs(Vp - Vg) / Vg
        rve_value_percentage = rve_value * 100
        # percentage error


        # convert to numpy array
        rve.append(rve_value_percentage.cpu().numpy().item())



        
    

    return rve