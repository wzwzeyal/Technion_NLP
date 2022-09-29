import csv
import glob
import os

import numpy as np
import pandas as pd
from datasets import load_dataset
from tqdm import tqdm

from consts import *

tqdm.pandas()


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
                   is_return_per_unk=True,
                   remove_all_only_unk=True):
    output_path = f"{path}_ner_data.jsonl"

    if not force_create:
        if os.path.exists(output_path):
            return output_path

    first_names, last_names = np.load('./data/list_names.npy', allow_pickle=True)

    # remove duplicates from first / last names
    first_names = set(first_names) - set(last_names)
    last_names = set(last_names) - set(first_names)

    if columns is None:
        columns = ["text", "ner"]
    all_files = glob.glob(os.path.join(path, pattern))

    res = pd.DataFrame()
    for count, f in enumerate(tqdm(all_files, desc="Read files")):
        df = pd.read_csv(
            f,
            sep='\t', quoting=csv.QUOTE_NONE, encoding='utf-8',
            skip_blank_lines=False,
            names=columns
        )

        # Extract the tokens/ner_tags

        df = extract_tokens_and_ner_tags(df)

        # Generate the name tags (B-PER-F/B-PER-L/B-PER-UNK)

        tqdm.pandas(desc="Create name_tags (B-PER-F/B-PER-L/B-PER-UNK)")
        df['name_tags'] = df.progress_apply(
            lambda row: process_ner_item(row, first_names, last_names, is_return_per_unk), axis=1
        )

        res = pd.concat([res, df])

    generate_ids(res)

    print("Finding all different ner tags ...")
    tag_counts = res['ner_tags'].explode().value_counts()
    tag_features = sorted(tag_counts.keys())
    tag_features_dict = dict(zip(tag_features, range(len(tag_features))))
    tqdm.pandas(desc="Creating the tag ner ids")
    res['tag_ner_ids'] = res['ner_tags'].progress_apply(
        lambda ner_list: [tag_features_dict[ner_tag_string] for ner_tag_string in ner_list]
    )

    # remove all the rows containing only "O"
    if remove_all_only_unk:
        tqdm.pandas(desc="Create ner ids string (temp column for removing all unk")
        res['name_ner_ids_str'] = res['name_tags'].progress_apply(lambda item: "".join(set(item)))
        res = res[res["name_ner_ids_str"] != "O"]

        tqdm.pandas(desc="Create ner ids string (temp column for removing all unk")
        res['name_ner_ids_str'] = res['ner_tags'].progress_apply(lambda item: "".join(set(item)))
        res = res[res["name_ner_ids_str"] != "O"]



    print(f"saving {output_path} ...")
    res.reset_index(drop=True, inplace=True)
    res.to_json(f"{output_path}", orient="records", lines=True)

    return output_path


def generate_ids(res, source_col, dest_col):
    name_counts = res[source_col].explode().value_counts()
    name_features = sorted(name_counts.keys())
    name_features_dict = dict(zip(name_features, range(len(name_features))))
    tqdm.pandas(desc=f"Create {dest_col} from {source_col}")
    res[dest_col] = res[source_col].progress_apply(
        lambda ner_list: [name_features_dict[ner_tag_string] for ner_tag_string in ner_list]
    )


def extract_tokens_and_ner_tags(df):
    # create a sentence_id if all the values is null
    df['sentence_id'] = df.isnull().all(axis=1).cumsum()
    df.dropna(inplace=True)
    df['tokens'] = df[['sentence_id', 'text', 'ner']].groupby(['sentence_id'])['text'].apply(list)
    df['ner_tags'] = df[['sentence_id', 'text', 'ner']].groupby(['sentence_id'])['ner'].apply(list)
    df.dropna(inplace=True)
    df = df[["tokens", "ner_tags"]]
    return df
