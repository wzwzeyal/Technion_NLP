import argparse
import os
import yaml
from box import Box
from tqdm import tqdm
from pprint import pprint
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

import wandb

from dataloading import TweetDataset
from modeling import TweetNetBase, TweetLSTM, TweetRNN, TweetGRU
from utils import *
from consts import *

import seaborn as sns


def train(training_args):
    # Setting up logging
    wandb.init(project=PROJECT_NAME, name=training_args.name, config=training_args, entity="wzeyal")

    pprint(training_args)

    # Check args and unpack them
    check_args(training_args)
    data_args, model_args = training_args.data_args, training_args.model_args

    # Set seed for reproducibility
    set_seed(training_args.seed)

    print("Loading datasets")
    train_dataset = TweetDataset(data_args, DATA_DIR / (TRAIN + CSV))
    fig = plt.figure(1)
    sns.countplot(x=LABEL, data=train_dataset.df)
    wandb.log({"train_dataset balance ": fig})

    # sns.countplot(x='sex', data=train_dataset)

    train_dataloader = DataLoader(train_dataset, data_args.batch_size, shuffle=data_args.shuffle)
    if True:  # training_args.do_eval:
        dev_dataset = TweetDataset(data_args, DATA_DIR / (DEV + CSV), train_dataset.vocab)
        fig = plt.figure(2)
        sns.countplot(x=LABEL, data=dev_dataset.df)
        wandb.log({"dev_dataset balance ": fig})

        dev_dataloader = DataLoader(dev_dataset, data_args.eval_batch_size)
    if training_args.do_test:
        test_dataset = TweetDataset(data_args, DATA_DIR / (TEST + CSV), train_dataset.vocab)
        test_dataloader = DataLoader(test_dataset, data_args.eval_batch_size)

    print("Initializing model")
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model_dict = {
        "LSTM": TweetLSTM(model_args, train_dataset.vocab_size),
        "RNN": TweetRNN(model_args, train_dataset.vocab_size),
        "GRU": TweetGRU(model_args, train_dataset.vocab_size),
    }
    model = model_dict[training_args.seq_model_name].to(device)
    wandb.watch(model)
    optimizer = torch.optim.Adam(model.parameters(), lr=training_args.learning_rate)
    loss_fn = nn.CrossEntropyLoss()

    for epoch in range(training_args.num_epochs):
        print(f"\n\n-------- Epoch: {epoch} --------\n")
        if training_args.do_train:
            train_loop(train_dataloader, model, loss_fn, optimizer, device, epoch)
        if training_args.do_eval_on_train:
            eval_loop(train_dataloader, model, loss_fn, device, TRAIN, epoch)
        if training_args.do_eval:
            eval_loop(dev_dataloader, model, loss_fn, device, DEV, epoch)

    if training_args.do_test:
        eval_loop(test_dataloader, model, loss_fn, device, TEST, epoch)

    return


def train_loop(dataloader, model, loss_fn, optimizer, device, epoch):
    model.train()

    for iter_num, (input_ids, labels) in enumerate(tqdm(dataloader, desc="Train Loop")):
        input_ids, labels = input_ids.to(device), labels.to(device)

        # Compute prediction and loss
        logits = model(input_ids)
        loss = loss_fn(logits, labels)

        # Backpropagation
        # optimizer.zero_grad()
        loss.backward()

        # accumulate gradients: preform a optimization step every training_args.accumulate_grad_batches iterations, or when you reach the end of the epoch
        # Remember: iter_num starts at 0. If you set training_args.accumulate_grad_batches to 3, you want to preform your first optimization at the third iteration.

        if ((iter_num + 1) % training_args.accumulation_steps == 0) or (iter_num + 1 == len(dataloader)):
            optimizer.step()
            optimizer.zero_grad()

        # Log loss
        wandb.log({"train_loop_loss": loss, EPOCH: epoch, ITERATION: iter_num})


def eval_loop(dataloader, model, loss_fn, device, split, epoch):
    # Change model to eval mode
    model.eval()

    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    average_loss, correct = 0, 0

    with torch.no_grad():
        for iter_num, (input_ids, labels) in enumerate(tqdm(dataloader, desc=f"Eval loop on {split}")):
            input_ids, labels = input_ids.to(device), labels.to(device)

            logits = model(input_ids)

            # Compute metrics
            average_loss += loss_fn(logits, labels).item()
            pred = logits.argmax(dim=1)
            correct += (pred == labels).float().sum().item()

    # Aggregate metrics
    average_loss /= num_batches
    accuracy = correct / size

    # Log metrics, report everything twice for cross-model comparison too
    wandb.log({f"{split}_average_loss": average_loss, EPOCH: epoch})
    wandb.log({f"{split}_accuracy": accuracy, EPOCH: epoch})


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train an LSTM model on the IMDB dataset.')
    parser.add_argument('--config', default='config.yaml', type=str,
                        help='Path to YAML config file. Defualt: config.yaml')

    # with open('config.yaml', 'r') as stream:
    #     config_vars = yaml.safe_load(stream)
    #     pprint(config_vars)
    #     pass

    parser.add_argument('--learning_rate', default=1, type=float,
                        help='learning_rate')

    parser.add_argument('--accumulation_steps', default=1, type=int,
                        help='accumulation_steps')

    parser.add_argument('--hidden_size', default=1, type=int,
                        help='hidden_size')

    parser.add_argument('--num_layers', default=1, type=int,
                        help='num_layers')

    parser.add_argument('--seq_model_name', default="LSTM", type=str,
                        help='seq_model_name: LSTM/RNN/GRU')

    parser.add_argument('--model_args.seq_args.bidirectional', default=True, type=bool,
                        help='bidirectional')

    parser.add_argument('--model_args.dropout', default=True, type=bool,
                        help='model_args.dropout')

    parser.add_argument('--model_args.seq_args.dropout', default=True, type=bool,
                        help='model_args.seq_args.dropout')

    args = parser.parse_args()

    with open(args.config) as f:
        training_args = Box(yaml.load(f, Loader=yaml.FullLoader))

    train(training_args)

# https://jovian.ai/aakanksha-ns/lstm-multiclass-text-classification
# https://towardsdatascience.com/multiclass-text-classification-using-lstm-in-pytorch-eac56baed8df
# https://www.kaggle.com/code/mlwhiz/multiclass-text-classification-pytorch
