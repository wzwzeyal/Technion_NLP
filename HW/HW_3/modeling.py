import torch
import torch.nn as nn
import numpy as np
from gensim import downloader


class TweetNet(nn.Module):
    def __init__(self, model_args, vocab):
        super(TweetNet, self).__init__()
        self.model_args = model_args
        if self.model_args.seq_args.bidirectional:
            self.hidden_size = self.model_args.seq_args.hidden_size * 2
        else:
            self.hidden_size = self.model_args.seq_args.hidden_size
        self.output_size = model_args.output_size
        self.dropout = model_args.dropout

        if model_args.seq_model_name == "RNN":
            self.seq_model = nn.LSTM(**self.model_args.seq_args)
        elif model_args.seq_model_name == "GRU":
            self.seq_model = nn.GRU(**self.model_args.seq_args)
        elif model_args.seq_model_name == "LSTM":
            self.seq_model = nn.LSTM(**self.model_args.seq_args)
        else:
            assert KeyError(), f"illegal seq model: {self.seq_model_name}"

        glove_model = f"{model_args.embedding_model}-{model_args.seq_args.input_size}"
        print(f"downloading {glove_model}")
        embedding_model = downloader.load(glove_model)
        embedding_matrix = self.create_embedding_matrix(vocab, embedding_model, model_args.seq_args.input_size)

        self.embedding = nn.Embedding(len(vocab), model_args.seq_args.input_size, padding_idx=len(vocab)-1)
        self.embedding.weight = nn.Parameter(torch.tensor(embedding_matrix, dtype=torch.float32))
        # we do not want to train the pretrained embeddings
        self.embedding.weight.requires_grad = model_args.weight_requires_grad
        self.cat_max_and_mean = model_args.cat_max_and_mean

        hidden_factor = 2 if self.cat_max_and_mean else 1
        # Classifier containing dropout, linear layer and sigmoid
        self.classifier = nn.Sequential(
            nn.Dropout(self.dropout),
            nn.Linear(hidden_factor * self.hidden_size, self.output_size),
            nn.Sigmoid()
        )


    def create_embedding_matrix(self, word_index, embedding_model, embedding_size):
        """
        This function creates the embedding matrix
        :param word_index: a dictionary of word: index_value
        :param embedding_dict:
        :return a numpy array with embedding vectors for all known words
        """
        # intialize the embedding matrix
        embedding_matrix = np.zeros((len(word_index), embedding_size))
        for i, word in enumerate(word_index):
            if word in embedding_model:
                embedding_matrix[i] = embedding_model.vectors[embedding_model.key_to_index[word]]
        return embedding_matrix

    def forward(self, input_ids, lengths):
        # Embed
        embeds = self.embedding(input_ids)  # (B, max_seq_length, input_size)

        # Pack and run through LSTM
        packed_embeds = torch.nn.utils.rnn.pack_padded_sequence(embeds, lengths=lengths, batch_first=True, enforce_sorted=False)
        lstm_packed_out, _ = self.seq_model(packed_embeds)  # (B, max_seq_length, hidden_size)
        lstm_out, _ = torch.nn.utils.rnn.pad_packed_sequence(lstm_packed_out, batch_first=True)

        # Take the mean of the output vectors of every sequence (with consideration of their length and padding)
        classifier_input = (lstm_out.sum(dim=1).t() / lengths.to(lstm_out.device)).t()  # (B, hidden_size)

        if self.cat_max_and_mean:
            max_embeddings, _ = torch.max(lstm_out, 1)
            classifier_input = torch.cat((classifier_input, max_embeddings), 1)

        # Classifier
        logits = self.classifier(classifier_input)  # (B, n_classes)

        logits = logits.float()
        return logits
