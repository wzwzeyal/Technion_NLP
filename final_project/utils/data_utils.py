import csv
import glob
import os
from operator import itemgetter

import pandas as pd
from consts import *
from datasets import load_dataset
from tqdm import tqdm


def load_dataset_from_files(data_args, suffix=JSON):
    data_dir = DATA_DIR / data_args.dataset
    data_files = {
        TRAIN: data_dir / (TRAIN + suffix),
        VALIDATION: data_dir / (VALIDATION + suffix),
        TEST: data_dir / (TEST + suffix)
    }
    raw_datasets = load_dataset(suffix, data_files=data_files)
    return raw_datasets


def take_first_ner_item(ner_list):
    return [item.split('|')[0] for item in ner_list]

def create_dataset(path, pattern="*.biose", columns=None, take_first_ner=True, force_create=False):
    print("create_dataset")
    # TODO: How to handle multiple NER classifications ? (e.g. b-per | ORG)
    output_path = f"{path}_ner_data.jsonl"

    if not force_create:
        if os.path.exists(output_path):
            return output_path

    if columns is None:
        columns = ["text", "ner"]
    all_files = glob.glob(os.path.join(path, pattern))[:5]

    res = pd.DataFrame()
    for count, f in enumerate(tqdm(all_files)):
        df = pd.read_csv(
            f,
            sep='\t', quoting=csv.QUOTE_NONE, encoding='utf-8',
            skip_blank_lines=False,
            names=columns
        )

        # create a sentence_id if all the values is null
        df['sentence_id'] = df.isnull().all(axis=1).cumsum()

        # remove all the NA rows
        df.dropna(inplace=True)
        df['tokens'] = df[['sentence_id', 'text', 'ner']].groupby(['sentence_id'])['text'].apply(list)
        df['ner_tags'] = df[['sentence_id', 'text', 'ner']].groupby(['sentence_id'])['ner'].apply(list)
        df.dropna(inplace=True)
        df = df[["tokens", "ner_tags"]]

        res = pd.concat([res, df])

    if take_first_ner:
        res['ner_tags'] = res['ner_tags'].apply(
            lambda ner_list: [item.split('|')[0] for item in ner_list])
            # lambda single_ner_list: single_ner_list for ner in single_ner_list    )

    res.reset_index(drop=True, inplace=True)

    counts = res['ner_tags'].explode().value_counts()
    labels = sorted(counts.keys())
    d = dict(zip(labels, range(len(labels))))

    res['ner_ids'] = res['ner_tags'].apply(lambda ner_list: [d[ner_tag_string] for ner_tag_string in ner_list])

    res.to_json(f"{output_path}", orient="records", lines=True)

    return output_path
    # return res


def align_labels_with_tokens(labels, word_ids):
    new_labels = []
    current_word = None
    for word_id in word_ids:
        if word_id != current_word:
            # Start of a new word!
            current_word = word_id
            if word_id is not None and word_id >= len(labels):
                xx = 5
            label = -100 if word_id is None else labels[word_id]
            new_labels.append(label)
        elif word_id is None:
            # Special token
            new_labels.append(-100)
        else:
            # Same word as previous token
            label = labels[word_id]

            # If the label is B-XXX we change it to I-XXX
            # if label % 2 == 1:
            #     label += 1
            if 'B-' in label:
                label.replace('B-', 'I-')

            new_labels.append(label)

    return new_labels
