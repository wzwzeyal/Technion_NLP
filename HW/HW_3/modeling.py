import torch
import torch.nn as nn
import wandb


class TweetNetBase(nn.Module):
    def __init__(self, model_args, vocab_size):
        super(TweetNetBase, self).__init__()
        self.model_args = model_args
        if self.model_args.seq_args.bidirectional:
            self.hidden_size = self.model_args.seq_args.hidden_size * 2
        else:
            self.hidden_size = self.model_args.seq_args.hidden_size
        self.output_size = model_args.output_size
        self.dropout = model_args.dropout

        self.embedding = nn.Embedding(vocab_size, model_args.seq_args.input_size, padding_idx=vocab_size-1)

        # Classifier containing dropout, linear layer and sigmoid
        self.classifier = nn.Sequential(
            nn.Dropout(self.dropout),
            nn.Linear(self.hidden_size, self.output_size),
            nn.Sigmoid()
        )

    def forward_hidden_seq_model(self, embedding):
        raise NotImplementedError("Subclass should implement this")

    def forward(self, input_ids, lengths):
        # Embed
        embeds = self.embedding(input_ids)  # (B, max_seq_length, input_size)

        # Pack and run through LSTM
        packed_embeds = torch.nn.utils.rnn.pack_padded_sequence(embeds, lengths=lengths, batch_first=True, enforce_sorted=False)
        lstm_packed_out, _ = self.seq_model(packed_embeds)  # (B, max_seq_length, hidden_size)
        lstm_out, _ = torch.nn.utils.rnn.pad_packed_sequence(lstm_packed_out, batch_first=True)

        # Take the mean of the output vectors of every sequence (with consideration of their length and padding)
        seq_embeddings = (lstm_out.sum(dim=1).t() / lengths.to(lstm_out.device)).t()  # (B, hidden_size)

        # Classifier
        logits = self.classifier(seq_embeddings)  # (B, n_classes)
        # logits = logits[:, 1]  # Take only the logits corresponding to 1
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
