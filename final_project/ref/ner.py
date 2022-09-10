import pandas as pd
import numpy as np
import csv
from pathlib import Path
import re
from sklearn.model_selection import train_test_split
from transformers import DistilBertTokenizerFast
from transformers import DistilBertForTokenClassification, BertForTokenClassification
from transformers import DistilBertForSequenceClassification, Trainer, TrainingArguments
from transformers import BertModel, BertTokenizerFast
import torch
from datasets import load_metric

torch.cuda.empty_cache()
OUTPUT_DIR = './models'


def read_dataset_folder(folder_path):
    token_docs = []
    tag_docs = []
    files_path = [f for f in Path(folder_path, encoding='UTF-8-sig').iterdir() if f.is_file()]
    for file_path in files_path:
        raw_text = file_path.read_text(encoding='UTF-8-sig').strip()
        raw_docs = re.split(r'\n\t?\n', raw_text)
        for doc in raw_docs:
            tokens = []
            tags = []
            for line in doc.split('\n'):
                token, tag = line.split('\t')
                tokens.append(token)
                tags.append(tag)
            token_docs.append(tokens)
            tag_docs.append(tags)
    return token_docs, tag_docs


path = "../data/iahlt-release-2022-06-09/ne/he"
texts, tags = read_dataset_folder(path)
texts = texts[:4000]
tags = tags[:4000]
train_texts, val_texts, train_tags, val_tags = train_test_split(texts, tags, test_size=.2)
unique_tags = set(tag for doc in tags for tag in doc)
tag2id = {tag: id for id, tag in enumerate(unique_tags)}
id2tag = {id: tag for tag, id in tag2id.items()}
tokenizer = BertTokenizerFast.from_pretrained('onlplab/alephbert-base')
train_encodings = tokenizer(train_texts, is_split_into_words=True, return_offsets_mapping=True, padding=True,
                            truncation=True)
val_encodings = tokenizer(val_texts, is_split_into_words=True, return_offsets_mapping=True, padding=True,
                          truncation=True)


def encode_tags(tags, encodings):
    labels = [[tag2id[tag] for tag in doc] for doc in tags]
    encoded_labels = []
    for doc_labels, doc_offset in zip(labels, encodings.offset_mapping):
        # create an empty array of -100
        doc_enc_labels = np.ones(len(doc_offset), dtype=int) * -100
        arr_offset = np.array(doc_offset)
        # set labels whose first offset position is 0 and the second is not 0
        mask = (arr_offset[:, 0] == 0) & (arr_offset[:, 1] != 0)
        doc_enc_labels[mask] = doc_labels[:np.sum(mask)]
        encoded_labels.append(doc_enc_labels.tolist())
    return encoded_labels


train_labels = encode_tags(train_tags, train_encodings)
val_labels = encode_tags(val_tags, val_encodings)
unique_tags_list = list(unique_tags)
metric = load_metric("seqeval")


def compute_metrics(p):
    predictions, labels = p
    # select predicted index with maximum logit for each token
    predictions = np.argmax(predictions, axis=2)
    # Remove ignored index (special tokens)
    true_predictions = [
        [unique_tags_list[p] for (p, l) in zip(prediction, label) if l != -100]
        for prediction, label in zip(predictions, labels)
    ]
    true_labels = [
        [unique_tags_list[l] for (p, l) in zip(prediction, label) if l != -100]
        for prediction, label in zip(predictions, labels)
    ]
    results = metric.compute(predictions=true_predictions, references=true_labels)
    return {
        "precision": results["overall_precision"],
        "recall": results["overall_recall"],
        "f1": results["overall_f1"],
        "accuracy": results["overall_accuracy"],
    }


class NERDataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels)


train_encodings.pop("offset_mapping")  # we don't want to pass this to the model
val_encodings.pop("offset_mapping")
train_dataset = NERDataset(train_encodings, train_labels)
val_dataset = NERDataset(val_encodings, val_labels)
model = BertForTokenClassification.from_pretrained('onlplab/alephbert-base', num_labels=len(unique_tags))
training_args = TrainingArguments(
    output_dir='./results',  # output directory
    num_train_epochs=3,  # total number of training epochs
    per_device_train_batch_size=16,  # batch size per device during training
    per_device_eval_batch_size=64,  # batch size for evaluation,
    evaluation_strategy="epoch",
    warmup_steps=500,  # number of warmup steps for learning rate scheduler
    weight_decay=0.01,  # strength of weight decay
    logging_dir='./logs',  # directory for storing logs
    logging_steps=10,
)
trainer = Trainer(
    model=model,  # the instantiated 🤗 Transformers model to be trained
    args=training_args,  # training arguments, defined above
    train_dataset=train_dataset,  # training dataset
    eval_dataset=val_dataset,  # evaluation dataset
    compute_metrics=compute_metrics
)
trainer.train()
trainer.evaluate()
model_to_save = trainer.model.module if hasattr(trainer.model,
                                                'module') else trainer.model  # Take care of distributed/parallel training
model_to_save.save_pretrained(OUTPUT_DIR)
