import logging
import os
import random

import numpy as np
import torch

import wandb


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


def load_vectors(fname):
    fin = open(fname)
    data = {}
    for line in fin:
        tokens = line.split()
        data[tokens[0]] = np.array([float(value) for value in tokens[1:]])
    return data


class CheckpointSaver:
    def __init__(self, dirpath, decreasing=True, top_n=5):
        """
        dirpath: Directory path where to store all model weights
        decreasing: If decreasing is `True`, then lower metric is better
        top_n: Total number of models to track based on validation metric value
        """
        if not os.path.exists(dirpath): os.makedirs(dirpath)
        self.dirpath = dirpath
        self.top_n = top_n
        self.decreasing = decreasing
        self.top_model_paths = []
        self.best_metric_val = np.Inf if decreasing else -np.Inf

    def __call__(self, model, epoch, metric_val):
        model_path = os.path.join(self.dirpath,
                                  model.__class__.__name__ + f'_{metric_val * 100:.2f}' + f'_epoch{epoch}.pt')
        save = metric_val < self.best_metric_val if self.decreasing else metric_val > self.best_metric_val
        if save:
            logging.info(
                f"Current metric value better than {metric_val} better than best {self.best_metric_val}, saving model at {model_path}, & logging model weights to W&B.")
            self.best_metric_val = metric_val
            torch.save(model.state_dict(), model_path)
            self.log_artifact(f'model-ckpt-epoch-{metric_val * 100:.2f}-{epoch}.pt', model_path, metric_val)
            self.top_model_paths.append({'path': model_path, 'score': metric_val})
            self.top_model_paths = sorted(self.top_model_paths, key=lambda o: o['score'], reverse=not self.decreasing)
        if len(self.top_model_paths) > self.top_n:
            self.cleanup()

    @staticmethod
    def log_artifact(filename, model_path, metric_val):
        artifact = wandb.Artifact(filename, type='model', metadata={'Validation score': metric_val})
        artifact.add_file(model_path)
        wandb.run.log_artifact(artifact)

    def cleanup(self):
        to_remove = self.top_model_paths[self.top_n:]
        logging.info(f"Removing extra models.. {to_remove}")
        for o in to_remove:
            if os.path.exists(o['path']):
                pass
                os.remove(o['path'])
        self.top_model_paths = self.top_model_paths[:self.top_n]
