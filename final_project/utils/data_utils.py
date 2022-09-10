import glob

import pandas as pd
from datasets import load_dataset

from consts import *
from tqdm import tqdm
import os
import csv

def load_dataset_from_files(data_args, suffix=JSON):
    data_dir = DATA_DIR / data_args.dataset
    data_files = {
        TRAIN: data_dir / (TRAIN + suffix),
        VALIDATION: data_dir / (VALIDATION + suffix),
        TEST: data_dir / (TEST + suffix)
    }
    raw_datasets = load_dataset(suffix, data_files=data_files)
    return raw_datasets


def create_dataset(path, pattern="*.biose", columns=None, take_first_ner=True):
    print("create_dataset")
    # TODO: How to handle multiple NER classifications ? (e.g. b-per | ORG)
    output_path = f"{path}_ner_data.jsonl"
    if os.path.exists(output_path):
        return pd.read_json(output_path, orient="records", lines=True)
    if columns is None:
        columns = ["text", "ner"]
    all_files = glob.glob(os.path.join(path, pattern))

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
        df['words'] = df[['sentence_id', 'text', 'ner']].groupby(['sentence_id'])['text'].apply(list)
        df['ner_labels'] = df[['sentence_id', 'text', 'ner']].groupby(['sentence_id'])['ner'].apply(list)
        df.dropna(inplace=True)
        df = df[["words", "ner_labels"]]

        res = pd.concat([res, df])

    if take_first_ner:
        res['ner'] = res['ner'].apply(lambda x: x.split('|')[0])

    res.reset_index(drop=True, inplace=True)
    res.to_json(f"{output_path}", orient="records", lines=True)
    # return res

