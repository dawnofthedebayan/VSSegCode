
import os
import random
import numpy as np
import torch


def seed_everything( seed=42):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(mode=True,warn_only=True)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def find_latest_checkpoint(checkpoint_folder,fold):
    if not os.path.exists(checkpoint_folder):
        raise Exception(f"Checkpoint folder {checkpoint_folder} does not exist")
    # find the latest checkpoint
    checkpoints = os.listdir(checkpoint_folder)
    checkpoints = [checkpoint for checkpoint in checkpoints if checkpoint.endswith('.ckpt')]
    checkpoints = [checkpoint for checkpoint in checkpoints if f'fold_{fold}' in checkpoint] 
    checkpoints = sorted(checkpoints)
    latest_checkpoint = checkpoints[-1]
    latest_checkpoint = os.path.join(checkpoint_folder, latest_checkpoint)

    return latest_checkpoint