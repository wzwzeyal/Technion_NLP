import csv
import glob
import os

import numpy as np
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


def process_ner_item(row, first_name_list, last_names_list, is_return_per_unk):
    name_tag = []
    for index, tag in enumerate(row['ner_tags']):
        if 'per' in tag.lower():
            name = row['tokens'][index]
            name_tag.append(is_first_last_or_none(name, first_name_list, last_names_list, is_return_per_unk))
        else:
            name_tag.append("O")
    return name_tag


def is_first_last_or_none(name, first_name_list, last_names_list, is_return_per_unk):
    if name in first_name_list:
        return "B-PER-F"
    elif name in last_names_list:
        return "B-PER-L"
    return "B-PER_UNK" if is_return_per_unk else "O"

    # # single item:
    # if 'per' not in item.lower():
    #     return 'O'
    # else:
    #     return item
    # return take_first_ner_item(item)


def create_dataset(path, pattern="*.biose", columns=None, force_create=False,
                   pre_process_ner_tags=False,
                   is_return_per_unk=True,
                   remove_all_only_unk=True):
    print("create_dataset")
    print(pre_process_ner_tags)

    lFirstName = []
    lLastName = []

    if pre_process_ner_tags:
        lFirstName, lLastName = np.load('./data/list_names.npy', allow_pickle=True)

    lFirstName = set(lFirstName) - set(lLastName)
    lLastName = set(lLastName) - set(lFirstName)

    # TODO: How to handle multiple NER classifications ? (e.g. b-per | ORG)
    output_path = f"{path}_ner_data.jsonl"

    if not force_create:
        if os.path.exists(output_path):
            return output_path

    if columns is None:
        columns = ["text", "ner"]
    all_files = glob.glob(os.path.join(path, pattern))
    print(len(all_files))

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

    # if take_first_ner:
    #     res['ner_tags'] = res['ner_tags'].apply(
    #         lambda ner_list: [item.split('|')[0] for item in ner_list])
    #     # lambda single_ner_list: single_ner_list for ner in single_ner_list    )

    if pre_process_ner_tags:
        res['name_tags'] = res.apply(
            lambda row: process_ner_item(row, lFirstName, lLastName, is_return_per_unk), axis=1
        )

        name_counts = res['name_tags'].explode().value_counts()
        name_features = sorted(name_counts.keys())
        name_features_dict = dict(zip(name_features, range(len(name_features))))
        res['name_ner_ids'] = res['name_tags'].apply(
            lambda ner_list: [name_features_dict[ner_tag_string] for ner_tag_string in ner_list]
        )

        # remove all the rows containing only "O"
        if remove_all_only_unk:
            res['name_ner_ids_str'] = res['name_tags'].apply(lambda item: "".join(set(item)))
            res.drop(res[res["name_ner_ids_str"] == "O"].index, inplace=True)

    res.reset_index(drop=True, inplace=True)

    tag_counts = res['ner_tags'].explode().value_counts()
    tag_features = sorted(tag_counts.keys())
    tag_features_dict = dict(zip(tag_features, range(len(tag_features))))
    res['tag_ner_ids'] = res['ner_tags'].apply(
        lambda ner_list: [tag_features_dict[ner_tag_string] for ner_tag_string in ner_list]
    )



    res.to_json(f"{output_path}", orient="records", lines=True)

    return output_path
