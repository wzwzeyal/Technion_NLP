import torch
import random
import numpy as np


def set_seed(seed):
    """
    Set random seeds for reproducibility
    """
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)


def check_args(training_args):
    """
    This is where you can check the validity of the configuration and set extra attributes that can't be incorporated
    in the YAML file
    """
    return training_args
