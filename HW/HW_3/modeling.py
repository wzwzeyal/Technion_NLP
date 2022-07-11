import torch
import torch.nn as nn


class TweetNet(nn.Module):
    def __init__(self, model_args, vocab_size):
        super(TweetNet, self).__init__()
        self.lstm_args = model_args.lstm_args
        self.hidden_size = self.lstm_args.hidden_size if not self.lstm_args.bidirectional else self.lstm_args.hidden_size * 2
        self.output_size = model_args.output_size
        self.dropout = model_args.dropout

        # Embedding of dim vocab_size x model_args.lstm_args.input_size
        self.embedding = nn.Embedding(vocab_size, model_args.lstm_args.input_size)
        # LSTM
        self.lstm = nn.LSTM(**self.lstm_args)
        # Classifier containing dropout, linear layer and sigmoid
        self.classifier =  nn.Sequential(
            nn.Dropout(self.dropout),
            nn.Linear(self.hidden_size, self.output_size),
            nn.Sigmoid()
        )

    def forward(self, input_ids):
        # Embed
        embeds =  self.embedding(input_ids) # (1, seq_length) -> (1, seq_length, input_size)

        # Run through LSTM and take the final layer's output
          # (1, seq_length, input_size) -> (1, max_seq_length, hidden_size)

        # Take the mean of all the output vectors
        seq_embeddings = None  # (1, max_seq_length, hidden_size) -> (1, hidden_size)

        # Classifier
        logits =  None # (1, hidden_size) -> (1, n_classes)
        logits = logits.float()
        return logits
