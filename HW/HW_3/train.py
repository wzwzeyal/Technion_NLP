import argparse
import uuid
from pprint import pprint

import matplotlib.pyplot as plt
import seaborn as sns
import torch.nn as nn
import yaml
from box import Box
from torch.optim.lr_scheduler import OneCycleLR
from torch.utils.data import DataLoader
from tqdm import tqdm

import wandb
from consts import *
from dataloading import TweetDataset
from modeling import TweetNet
from utils import *
from wandb_utils import CheckpointSaver


def do_infer(dataloader, model, device):
    model.eval()

    infer_results = []
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
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
    wandb.init(project=PROJECT_NAME, name=training_args.name, config=training_args, entity="wzeyal")

    pprint(training_args)

    dev_dataloader = None
    test_dataloader = None
    epoch = 0

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

    train_dataloader = DataLoader(train_dataset, data_args.batch_size, shuffle=True)
    if training_args.do_eval:
        dev_dataset = TweetDataset(data_args, DATA_DIR / (DEV + CSV), train_dataset.vocab)
        fig = plt.figure(2)
        sns.countplot(x=LABEL, data=dev_dataset.df)
        wandb.log({"dev_dataset balance": fig})

        dev_dataloader = DataLoader(dev_dataset, data_args.eval_batch_size)
    if training_args.do_infer:
        test_dataset = TweetDataset(data_args, DATA_DIR / (TEST + CSV), train_dataset.vocab)
        test_dataloader = DataLoader(test_dataset, data_args.eval_batch_size)

    print("Initializing model")
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = TweetNet(model_args, train_dataset.vocab).to(device)
    wandb.watch(model)

    n_iter = training_args.num_epochs * len(train_dataloader)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=training_args.learning_rate,
        betas=(0.9, 0.99),
        weight_decay=1e-2
    )
    scheduler = OneCycleLR(optimizer, max_lr=training_args.learning_rate, total_steps=n_iter)
    loss_fn = nn.CrossEntropyLoss()

    checkpoint_saver = CheckpointSaver(dirpath=f'./{MODEL_WEIGHTS}', decreasing=False, top_n=1)

    for epoch in range(training_args.num_epochs):
        print(f"\n\n-------- Epoch: {epoch} --------\n")
        if training_args.do_train:
            train_loop(train_dataloader, model, loss_fn, optimizer, scheduler, device, epoch)
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


def train_loop(dataloader, model, loss_fn, optimizer, scheduler, device, epoch):
    model.train()

    for iter_num, (input_ids, lengths, labels) in enumerate(tqdm(dataloader, desc="Train Loop")):
        input_ids, labels = input_ids.to(device), labels.to(device)

        # Compute prediction and loss
        logits = model(input_ids, lengths)
        loss = loss_fn(logits, labels)

        # Backpropagation
        # optimizer.zero_grad()
        loss.backward()

        # accumulate gradients: preform a optimization step every training_args.accumulate_grad_batches iterations, or when you reach the end of the epoch
        # Remember: iter_num starts at 0. If you set training_args.accumulate_grad_batches to 3, you want to preform your first optimization at the third iteration.

        if ((iter_num + 1) % training_args.accumulation_steps == 0) or (iter_num + 1 == len(dataloader)):
            optimizer.step()
            optimizer.zero_grad()
            last_lr = scheduler.get_last_lr()[0]
            wandb.log({"lr": last_lr, EPOCH: epoch, ITERATION: iter_num})
            scheduler.step()

        # Log loss
        wandb.log({"train_loop_loss": loss, EPOCH: epoch, ITERATION: iter_num})


def eval_loop(dataloader, model, loss_fn, device, split, epoch, checkpoint_saver=None):
    # Change model to eval mode
    model.eval()

    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    average_loss, correct = 0, 0

    with torch.no_grad():
        for iter_num, (input_ids, lengths, labels) in enumerate(tqdm(dataloader, desc=f"Eval loop on {split}")):
            input_ids, labels = input_ids.to(device), labels.to(device)

            logits = model(input_ids, lengths)

            # Compute metrics
            average_loss += loss_fn(logits, labels).item()
            preds = torch.argmax(logits, dim=1)
            # pred = logits.argmax(dim=1)
            correct += (preds == labels).float().sum().item()

    # Aggregate metrics
    average_loss /= num_batches
    accuracy = correct / size

    if checkpoint_saver is not None:
        checkpoint_saver(model, epoch, accuracy)
        if accuracy > ACC_THRESHOLD:
            wandb.alert(
                title="High Accuracy",
                text=f"Run {training_args.name} achieved {accuracy*100:.2f}% !"
            )

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

    parser.add_argument('--learning_rate', default=0.001, type=float,
                        help='learning_rate')

    parser.add_argument('--accumulation_steps', default=1, type=int,
                        help='accumulation_steps')

    parser.add_argument('--data_args.batch_size', default=64, type=int,
                        help='batch_size')

    parser.add_argument('--data_args.minimum_vocab_freq_threshold', default=1, type=int,
                        help='minimum_vocab_freq_threshold')

    parser.add_argument('--model_args.dropout', default=0.2, type=float,
                        help='dropout')

    parser.add_argument('--model_args.weight_requires_grad', default=False, type=bool,
                        help='weight_requires_grad')

    parser.add_argument('--model_args.cat_max_and_mean', default=False, type=bool,
                        help='cat_max_and_mean')

    parser.add_argument('--model_args.embedding_model', default="glove-wiki-gigaword", type=str,
                        help='embedding_model')

    parser.add_argument('--model_args.seq_args.hidden_size', default=256, type=int,
                        help='hidden_size')

    parser.add_argument('--model_args.seq_args.bidirectional', default=True, type=bool,
                        help='bidirectional')

    parser.add_argument('--model_args.seq_args.input_size', default=50, type=int,
                        help='input_size')

    parser.add_argument('--model_args.seq_args.num_layers:', default=2, type=int,
                        help='num_layers')

    args = parser.parse_args()

    args_box = Box(vars(args))

    with open(args.config) as config_file:
        training_args = Box(yaml.load(config_file, Loader=yaml.FullLoader))
        # training_args = yaml.safe_load(config_file)

    training_args.learning_rate = args_box.learning_rate
    training_args.accumulation_steps = args_box.accumulation_steps

    training_args.data_args.batch_size = args_box.data_args_batch_size
    training_args.data_args.minimum_vocab_freq_threshold = args_box.data_args_minimum_vocab_freq_threshold

    training_args.model_args.dropout = args_box.model_args_dropout
    training_args.model_args.weight_requires_grad = args_box.model_args_weight_requires_grad
    training_args.model_args.cat_max_and_mean = args_box.model_args_cat_max_and_mean
    training_args.model_args.embedding_model = args_box.model_args_embedding_model
    training_args.model_args.seq_args.hidden_size = args_box.model_args_seq_args_hidden_size
    training_args.model_args.seq_args.bidirectional = args_box.model_args_seq_args_bidirectional
    training_args.model_args.seq_args.input_size = args_box.model_args_seq_args_input_size
    training_args.model_args.seq_args.num_layers = args_box.model_args_seq_args_num_layers

    training_args.name = f"{training_args.name}_{str(uuid.uuid4())[:8]}"

    # merged_args = dict(training_args, **args_dict)
    train(Box(training_args))

# https://jovian.ai/aakanksha-ns/lstm-multiclass-text-classification
# https://towardsdatascience.com/multiclass-text-classification-using-lstm-in-pytorch-eac56baed8df
# https://www.kaggle.com/code/mlwhiz/multiclass-text-classification-pytorch
