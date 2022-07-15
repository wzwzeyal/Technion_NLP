import torch
import torch.nn as nn
import wandb


class TweetNetBase(nn.Module):
    def __init__(self, model_args, vocab_size):
        super(TweetNetBase, self).__init__()
        wandb.log({"model_args": model_args})
        wandb.log({"vocab_size": vocab_size})
        self.model_args = model_args
        if self.model_args.seq_args.bidirectional:
            self.hidden_size = self.model_args.seq_args.hidden_size * 2
        else:
            self.hidden_size = self.model_args.seq_args.hidden_size
        self.output_size = model_args.output_size
        self.dropout = model_args.dropout

        self.embedding = nn.Embedding(vocab_size, model_args.seq_args.input_size)

        # Classifier containing dropout, linear layer and sigmoid
        self.classifier = nn.Sequential(
            nn.Dropout(self.dropout),
            nn.Linear(self.hidden_size, self.output_size),
            nn.Sigmoid()
        )

    def forward_hidden_seq_model(self, embedding):
        raise NotImplementedError("Subclass should implement this")

    def forward(self, input_ids):
        # Embed
        embedding = self.embedding(input_ids)  # (1, seq_length) -> (1, seq_length, input_size)

        # Run through LSTM and take the final layer's output
        # (1, seq_length, input_size) -> (1, max_seq_length, hidden_size)
        # output_1_4_512, (hidden_6_1_256, cell_6_1_256) = self.lstm(embeds_1_4_50)
        # output_1_4_512, hidden_6_1_256 = self.lstm(embeds_1_4_50)
        hidden = self.forward_hidden_seq_model(embedding)

        # output_4_512 = torch.squeeze(output_1_4_512)
        # Take the mean of all the output vectors
        mean_vector = hidden.mean(dim=0)  # (1, max_seq_length, hidden_size) -> (1, hidden_size)

        # # Classifier
        logits = self.classifier(mean_vector)
        logits = logits.float()
        return logits


class TweetLSTM(TweetNetBase):
    def __init__(self, model_args, vocab_size):
        super(TweetLSTM, self).__init__(model_args, vocab_size)
        self.seq_model = nn.LSTM(**self.model_args.seq_args)

    def forward_hidden_seq_model(self, embedding):
        _, (hidden, _) = self.seq_model(embedding)
        return hidden


class TweetRNN(TweetNetBase):
    def __init__(self, model_args, vocab_size):
        super(TweetRNN, self).__init__(model_args, vocab_size)
        self.seq_model = nn.RNN(**self.model_args.seq_args)

    def forward_hidden_seq_model(self, embedding):
        _, hidden = self.seq_model(embedding)
        return hidden


class TweetGRU(TweetNetBase):
    def __init__(self, model_args, vocab_size):
        super(TweetGRU, self).__init__(model_args, vocab_size)
        self.seq_model = nn.GRU(**self.model_args.seq_args)

    def forward_hidden_seq_model(self, embedding):
        _, hidden = self.seq_model(embedding)
        return hidden
