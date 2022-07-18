import gensim
import pandas as pd
import torch
from torch.utils.data import Dataset
from sklearn.preprocessing import LabelEncoder
from sklearn.utils import compute_class_weight

from consts import *
import numpy as np




class TweetDataset(Dataset):
    def __init__(self, data_args, file_path, vocab=None):
        self.data_args = data_args
        self.file_path = file_path

        # Load data to dataframe
        self.df = pd.read_csv(self.file_path).head(1000)

        label_encoder = LabelEncoder().fit(self.df[LABEL])
        classes = label_encoder.classes_
        self.class_weight = compute_class_weight(class_weight="balanced", classes = classes, y=self.df[LABEL])

        self.unk_token = UNK_TOKEN
        self.pad_token = PAD_TOKEN

        # assert self.df.columns.tolist() == [TEXT, LABEL]

        # Get vocab
        if vocab is None:
            # Tokenize all of the text using gensim.utils.tokenize(text, lowercase=True)
            # https://stackoverflow.com/questions/51776314/how-to-concatenate-all-rows-of-a-column-of-a-data-frame-in-pandas-without-group
            tokenized_text = gensim.utils.tokenize(" ".join(self.df[TEXT].tolist()), lowercase=True)



            # # Create a set of all the unique tokens in the text
            # self.vocab = set(tokenized_text)

            # # Add the UNK token to the vocab
            # self.vocab.add(self.unk_token)

            # limit vocab by number of appearance
            counts = pd.Series(tokenized_text).value_counts()
            counts = counts[counts > self.data_args.minimum_vocab_freq_threshold]
            self.vocab = counts.keys().tolist()
            self.vocab.append(self.unk_token)
            self.vocab.append(self.pad_token)

        else:
            self.vocab = vocab



        # Set the vocab size
        self.vocab_size = len(self.vocab)

        # Create a dictionary mapping tokens to indices
        self.token2id = {i: v for v, i in enumerate(self.vocab)}
        self.id2token = {v: k for k, v in self.token2id.items()}

        assert self.token2id[self.id2token[0]] == 0

        # Tokenize data using the tokenize function
        self.df[INPUT_IDS] = self.df.apply(lambda row: self.tokenize(row[TEXT]), axis=1)

        pad_id = self.token2id[self.pad_token]
        self.df[TEXT_LEN] = self.df.apply(
            lambda row: len(row[INPUT_IDS]) - row[INPUT_IDS].count(pad_id), axis=1)
        pass
        # print(self.df.iloc[0])

    def __len__(self):
        # Return the length of the dataset
        return len(self.df)

    def __getitem__(self, idx):

        # return the input_ids and the label as tensors, make sure to convert the label type to a long

        input_ids = self.df.iloc[idx][INPUT_IDS]
        length = self.df.iloc[idx][TEXT_LEN]
        label = self.df.iloc[idx][LABEL]

        return (
            torch.LongTensor(input_ids),
            length,
            label
        )

    def tokenize(self, text):
        input_ids = []
        for i, token in enumerate(gensim.utils.tokenize(text, lowercase=True)):
            # Trim sequences to max_seq_length
            if i >= self.data_args.max_seq_length:
                break
            # Gets the token id, if unknown returns self.unk_token

            input_ids.append(
                self.token2id[token]
                if token in self.token2id.keys()
                else self.token2id[self.unk_token]
            )

        # Pad
        for i in range(self.data_args.max_seq_length - len(input_ids)):
            input_ids.append(self.token2id[self.pad_token])

        # return torch.LongTensor(input_ids)
        return input_ids