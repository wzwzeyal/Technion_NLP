import torch
import torch.nn as nn


class TweetNet(nn.Module):
    def __init__(self, model_args, vocab_size):
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
            assert KeyError(f"illeagal seq model: {self.seq_model}")

        self.embedding = nn.Embedding(vocab_size, model_args.seq_args.input_size, padding_idx=vocab_size - 1)

        # Classifier containing dropout, linear layer and sigmoid
        self.classifier = nn.Sequential(
            nn.Dropout(self.dropout),
            nn.Linear(self.hidden_size, self.output_size),
            nn.Sigmoid()
        )

    def forward(self, input_ids, lengths):
        # Embed
        embeds = self.embedding(input_ids)  # (B, max_seq_length, input_size)

        # Pack and run through LSTM
        packed_embeds = torch.nn.utils.rnn.pack_padded_sequence(embeds, lengths=lengths, batch_first=True,
                                                                enforce_sorted=False)
        lstm_packed_out, _ = self.seq_model(packed_embeds)  # (B, max_seq_length, hidden_size)
        lstm_out, _ = torch.nn.utils.rnn.pad_packed_sequence(lstm_packed_out, batch_first=True)

        # Take the mean of the output vectors of every sequence (with consideration of their length and padding)
        seq_embeddings = (lstm_out.sum(dim=1).t() / lengths.to(lstm_out.device)).t()  # (B, hidden_size)

        # Classifier
        logits = self.classifier(seq_embeddings)  # (B, n_classes)
        # logits = logits[:, 1]  # Take only the logits corresponding to 1
        logits = logits.float()
        return logits
