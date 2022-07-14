import gensim
import pandas as pd
import torch
from torch.utils.data import Dataset

from consts import *


class TweetDataset(Dataset):
    def __init__(self, data_args, file_path, vocab=None):
        self.data_args = data_args
        self.file_path = file_path

        # Load data to dataframe
        self.df = pd.read_csv(self.file_path).head(100)

        # assert self.df.columns.tolist() == [TEXT, LABEL]

        # Get vocab
        if vocab is None:
            # Tokenize all of the text using gensim.utils.tokenize(text, lowercase=True)
            # https://stackoverflow.com/questions/51776314/how-to-concatenate-all-rows-of-a-column-of-a-data-frame-in-pandas-without-group
            tokenized_text = gensim.utils.tokenize(" ".join(self.df[TEXT].tolist()), lowercase=True)
            # Create a set of all the unique tokens in the text
            self.vocab = set(tokenized_text)
        else:
            self.vocab = vocab

        # Add the UNK token to the vocab
        self.unk_token = UNK_TOKEN
        self.vocab.add(self.unk_token)

        # Set the vocab size
        self.vocab_size = len(self.vocab)

        # Create a dictionary mapping tokens to indices
        self.token2id = {i: v for v, i in enumerate(self.vocab)}
        self.id2token = {v: k for k, v in self.token2id.items()}

        assert self.token2id[self.id2token[400]] == 400

        # Tokenize data using the tokenize function
        self.df[INPUT_IDS] = self.df.apply(lambda row: self.tokenize(row[TEXT]), axis=1)
        # print(self.df.iloc[0])

    def __len__(self):
        # Return the length of the dataset
        return len(self.df)

    def __getitem__(self, idx):

        # return the input_ids and the label as tensors, make sure to convert the label type to a long

        input_ids = self.df.iloc[idx][INPUT_IDS]
        label = self.df.iloc[idx][LABEL]

        return torch.tensor(input_ids), torch.tensor(label, dtype=torch.long)

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

        return input_ids