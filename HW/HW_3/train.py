import argparse
import uuid
from pprint import pprint

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import torch.nn as nn
import yaml
from box import Box
from sklearn.metrics import confusion_matrix
from torch.utils.data import DataLoader
from tqdm import tqdm
from consts import *
from dataloading import TweetDataset
from modeling import TweetNet
from utils import *


def do_infer(dataloader, model, device):
    model.eval()

    infer_results = []
    with torch.no_grad():
        for iter_num, (input_ids, lengths, labels) in enumerate(tqdm(dataloader, desc=f"infer")):
            input_ids, labels = input_ids.to(device), labels.to(device)

            logits = model(input_ids, lengths)

            pred = logits.argmax(dim=1)
            pred_list = pred.cpu().detach().numpy()
            infer_results.extend(pred_list)

    test_df = dataloader.dataset.df

    test_df[LABEL] = infer_results

    output_file = SUBMISSION + CSV
    test_df[[TEXT, LABEL]].to_csv(output_file, index=False)
    artifact = wandb.Artifact(output_file, type='result')
    artifact.add_file(f"./{output_file}")
    wandb.run.log_artifact(artifact)


def train(training_args):
    # Setting up logging
    wandb.login(key='9bce89baa843332823d504e2b87e2821826c9390')
    wandb.init(project=PROJECT_NAME, name=training_args.name, config=training_args, entity="nlpcourse")

    pprint(training_args)

    test_dataloader = None
    epoch = 0

    # Check args and unpack them
    check_args(training_args)
    data_args, model_args = training_args.data_args, training_args.model_args

    # Set seed for reproducibility
    set_seed(training_args.seed)

    print("Loading datasets")
    print("Loading datasets")
    train_dataset = TweetDataset(data_args, DATA_DIR / (TRAIN + CSV))
    train_dataloader = DataLoader(train_dataset, data_args.batch_size, shuffle=data_args.shuffle)
    if training_args.perform_eda:
        fig = plt.figure(1)
        sns.countplot(x=LABEL, data=train_dataset.df)
        wandb.log({"train_dataset balance ": fig})
    # if training_args.do_eval:
    dev_dataset = TweetDataset(data_args, DATA_DIR / (DEV + CSV), train_dataset.vocab)
    if training_args.perform_eda:
        fig = plt.figure(2)
        sns.countplot(x=LABEL, data=dev_dataset.df)
        wandb.log({"dev_dataset balance ": fig})
    dev_dataloader = DataLoader(dev_dataset, data_args.eval_batch_size)

    if training_args.do_test:
        test_dataset = TweetDataset(data_args, DATA_DIR / (TEST + CSV), train_dataset.vocab)
        test_dataloader = DataLoader(test_dataset, data_args.eval_batch_size)
    if training_args.do_infer:
        test_dataset = TweetDataset(data_args, DATA_DIR / (TEST + CSV), train_dataset.vocab)
        test_dataloader = DataLoader(test_dataset, data_args.eval_batch_size)

    print("Initializing model")
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = TweetNet(model_args, train_dataset.vocab).to(device)
    wandb.watch(model)

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=training_args.learning_rate,
    )
    loss_fn = nn.CrossEntropyLoss()

    checkpoint_saver = CheckpointSaver(dirpath=f'./{MODEL_WEIGHTS}', decreasing=False, top_n=1)

    for epoch in range(training_args.num_epochs):
        print(f"\n\n-------- Epoch: {epoch} --------\n")
        if training_args.do_train:
            train_loop(train_dataloader, model, loss_fn, optimizer, device, epoch)
        if training_args.do_eval_on_train:
            eval_loop(train_dataloader, model, loss_fn, device, TRAIN, epoch)
        if training_args.do_eval:
            if dev_dataloader is not None:
                eval_loop(dev_dataloader, model, loss_fn, device, DEV, epoch, checkpoint_saver)

    if training_args.do_test:
        if test_dataloader is not None:
            eval_loop(test_dataloader, model, loss_fn, device, TEST, epoch)

    if training_args.do_infer:
        if test_dataloader is not None:
            do_infer(test_dataloader, model, device)
    return


def train_loop(dataloader, model, loss_fn, optimizer, device, epoch):
    model.train()

    for iter_num, (input_ids, lengths, labels) in enumerate(tqdm(dataloader, desc="Train Loop")):
        input_ids, labels = input_ids.to(device), labels.to(device)

        # Compute prediction and loss
        logits = model(input_ids, lengths)
        loss = loss_fn(logits, labels)

        # Backpropagation
        loss.backward()

        # accumulate gradients: preform an optimization step every training_args.accumulate_grad_batches iterations, or when you reach the end of the epoch
        # Remember: iter_num starts at 0. If you set training_args.accumulate_grad_batches to 3, you want to preform your first optimization at the third iteration.

        if ((iter_num + 1) % training_args.accumulation_steps == 0) or (iter_num + 1 == len(dataloader)):
            optimizer.step()
            optimizer.zero_grad()

        # Log loss
        wandb.log({"train_loop_loss": loss, EPOCH: epoch, ITERATION: iter_num})


def eval_loop(dataloader, model, loss_fn, device, split, epoch, checkpoint_saver=None):
    # Change model to eval mode
    model.eval()

    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    average_loss, correct = 0, 0

    y_pred = []

    with torch.no_grad():
        for iter_num, (input_ids, lengths, labels) in enumerate(tqdm(dataloader, desc=f"Eval loop on {split}")):
            input_ids, labels = input_ids.to(device), labels.to(device)

            logits = model(input_ids, lengths)

            # Compute metrics
            average_loss += loss_fn(logits, labels).item()
            preds = torch.argmax(logits, dim=1)
            correct += (preds == labels).float().sum().item()
            pred_list = preds.cpu().detach().numpy()
            y_pred.extend(pred_list)

    # Aggregate metrics
    average_loss /= num_batches
    accuracy = correct / size

    if checkpoint_saver is not None:
        checkpoint_saver(model, epoch, accuracy)
        if accuracy > ACC_THRESHOLD:
            wandb.alert(
                title="High Accuracy",
                text=f"Run {training_args.name} achieved {accuracy * 100:.2f}% !"
            )

    # Log metrics, report everything twice for cross-model comparison too
    wandb.log({f"{split}_average_loss": average_loss, EPOCH: epoch})
    wandb.log({f"{split}_accuracy": accuracy, EPOCH: epoch})


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train an LSTM model on the IMDB dataset.')
    parser.add_argument('--config', default='config.yaml', type=str,
                        help='Path to YAML config file. Default: config.yaml')

    parser.add_argument('--learning_rate', default=0.0001, type=float,
                        help='learning_rate')

    parser.add_argument('--accumulation_steps', default=1, type=int,
                        help='accumulation_steps')

    parser.add_argument('--hidden_size', default=512, type=int,
                        help='hidden_size')

    parser.add_argument('--num_layers', default=3, type=int,
                        help='num_layers')

    parser.add_argument('--dropout', default=0.2, type=float,
                        help='dropout')

    parser.add_argument('--backbone_model', default="GRU", type=str,
                        help='backbone_model')

    parser.add_argument('--bidirectional', default="False", type=str,
                        help='bidirectional')

    parser.add_argument('--input_size', default=100, type=int,
                        help='input_size')

    parser.add_argument('--minimum_vocab_freq_threshold', default=1, type=int,
                        help='minimum_vocab_freq_threshold')

    parser.add_argument('--embedding', default="glove-wiki-gigaword", type=str,
                        help='https://github.com/RaRe-Technologies/gensim-data')

    parser.add_argument('--embedding_weight_requires_grad', default="True", type=str,
                        help='embedding_weight_requires_grad')

    parser.add_argument('--batch_size', default=16, type=int,
                        help='batch_size')

    parser.add_argument('--cat_max_and_mean', default="False", type=str,
                        help='cat_max_and_mean')

    args = parser.parse_args()

    with open(args.config) as config_file:
        training_args = Box(yaml.load(config_file, Loader=yaml.FullLoader))

    training_args.learning_rate = args.learning_rate
    training_args.accumulation_steps = args.accumulation_steps

    training_args.data_args.batch_size = args.batch_size
    training_args.data_args.minimum_vocab_freq_threshold = args.minimum_vocab_freq_threshold

    training_args.model_args.dropout = args.dropout
    training_args.model_args.embedding_weight_requires_grad = args.embedding_weight_requires_grad == "True"
    training_args.model_args.cat_max_and_mean = args.cat_max_and_mean == "True"
    training_args.model_args.embedding = args.embedding
    training_args.model_args.seq_args.hidden_size = args.hidden_size
    training_args.model_args.seq_args.bidirectional = args.bidirectional == "True"
    training_args.model_args.seq_args.input_size = args.input_size

    training_args.name = f"{training_args.name}_{str(uuid.uuid4())[:8]}"

    train(Box(training_args))
