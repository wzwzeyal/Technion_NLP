import torch
from arabert.preprocess import ArabertPreprocessor
from transformers import AutoTokenizer


class NERDataset:
    def __init__(self, texts, tags, label_list, model_name, max_length, nof_samples, is_perform_word_cleaning):
        if nof_samples is None:
            self.texts = texts
            self.tags = tags
        else:
            self.texts = texts[:nof_samples]
            self.tags = tags[:nof_samples]

        self.label_map = {label: i for i, label in enumerate(label_list)}
        self.preprocessor = ArabertPreprocessor(model_name.split("/")[-1])
        self.pad_token_label_id = torch.nn.CrossEntropyLoss().ignore_index
        # Use cross entropy ignore_index as padding label id so that only
        # real label ids contribute to the loss later.
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.max_length = max_length
        self.is_perform_word_cleaning = is_perform_word_cleaning

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, item):
        textlist = self.texts[item]
        tags = self.tags[item]

        tokens = []
        label_ids = []
        for word, label in zip(textlist, tags):
            clean_word = self.preprocessor.preprocess(word)
            word_tokens = self.tokenizer.tokenize(clean_word)

            if len(word_tokens) > 0:
                tokens.extend(word_tokens)
                # Use the real label id for the first token of the word, and padding ids for the remaining tokens
                label_ids.extend([self.label_map[label]] + [self.pad_token_label_id] * (len(word_tokens) - 1))

        # Account for [CLS] and [SEP] with "- 2" and with "- 3" for RoBERTa.
        special_tokens_count = self.tokenizer.num_special_tokens_to_add()
        if len(tokens) > self.max_length - special_tokens_count:
            tokens = tokens[: (self.max_length - special_tokens_count)]
            label_ids = label_ids[: (self.max_length - special_tokens_count)]

        # Add the [SEP] token
        tokens += [self.tokenizer.sep_token]
        label_ids += [self.pad_token_label_id]
        token_type_ids = [0] * len(tokens)

        # Add the [CLS] TOKEN
        tokens = [self.tokenizer.cls_token] + tokens
        label_ids = [self.pad_token_label_id] + label_ids
        token_type_ids = [0] + token_type_ids

        input_ids = self.tokenizer.convert_tokens_to_ids(tokens)

        # The mask has 1 for real tokens and 0 for padding tokens. Only real
        # tokens are attended to.
        attention_mask = [1] * len(input_ids)

        # Zero-pad up to the sequence length.
        padding_length = self.max_length - len(input_ids)

        input_ids += [self.tokenizer.pad_token_id] * padding_length
        attention_mask += [0] * padding_length
        token_type_ids += [0] * padding_length
        label_ids += [self.pad_token_label_id] * padding_length

        assert len(input_ids) == self.max_length
        assert len(attention_mask) == self.max_length
        assert len(token_type_ids) == self.max_length
        assert len(label_ids) == self.max_length

        # if item < 5:
        #   print("*** Example ***")
        #   print("tokens:", " ".join([str(x) for x in tokens]))
        #   print("input_ids:", " ".join([str(x) for x in input_ids]))
        #   print("attention_mask:", " ".join([str(x) for x in attention_mask]))
        #   print("token_type_ids:", " ".join([str(x) for x in token_type_ids]))
        #   print("label_ids:", " ".join([str(x) for x in label_ids]))

        return {
            'input_ids': torch.tensor(input_ids, dtype=torch.long),
            'attention_mask': torch.tensor(attention_mask, dtype=torch.long),
            'token_type_ids': torch.tensor(token_type_ids, dtype=torch.long),
            'labels': torch.tensor(label_ids, dtype=torch.long)
        }
