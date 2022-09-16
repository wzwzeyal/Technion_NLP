#!/usr/bin/env python
# coding=utf-8
# Copyright 2020 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
""" Finetuning the library models for sequence classification on GLUE."""
# You can also adapt this script on your own text classification task. Pointers for this are left as comments.

import random

from transformers import (
    AutoConfig,
    DataCollatorWithPadding,
    HfArgumentParser,
    default_data_collator,
    AutoTokenizer,
    set_seed,
)
from transformers.utils.versions import require_version

from args.data_args import DataTrainingArguments
from args.model_args import ModelArguments
from args.training_args import ProjectTrainingArguments
from consts import *
from utils.data_utils import *
from utils.train_utils import *
from utils.utils import *

# from transformers import DistilBertForTokenClassification, BertForTokenClassification


# Will error if the minimal version of Transformers is not installed. Remove at your own risks.
# check_min_version("4.16.0.dev0")

require_version("datasets>=1.8.0", "To fix: pip install -r examples/pytorch/text-classification/requirements.txt")

logger = logging.getLogger(__name__)


def align_labels(tokenizer, row):
    inputs = tokenizer(row['tokens'], is_split_into_words=True)
    labels = row["ner_tags"]
    word_ids = inputs.word_ids()
    aligned_labels = align_labels_with_tokens(labels, word_ids)
    return aligned_labels


def preprocess_datasets(data_args, model_args, training_args, raw_datasets):
    tokenizer = AutoTokenizer.from_pretrained(model_args.model_name_or_path)
    inputs = tokenizer(raw_datasets["train"][0]["tokens"], is_split_into_words=True)
    print(inputs.tokens())
    labels = raw_datasets["train"][0]["ner_tags"]
    word_ids = inputs.word_ids()
    print(labels)
    print(align_labels_with_tokens(labels, word_ids))

    def tokenize_and_align_labels(examples):
        tokenized_inputs = tokenizer(
            examples["tokens"], truncation=True, is_split_into_words=True
        )
        all_labels = examples["ner_tags"]
        new_labels = []
        for i, labels in enumerate(all_labels):
            word_ids = tokenized_inputs.word_ids(i)
            new_labels.append(align_labels_with_tokens(labels, word_ids))

        tokenized_inputs["aligned_labels"] = new_labels
        return tokenized_inputs

    # row: {'tokens': [ ... ], 'ner_tags': [ ...]}
    # tokenized_datasets = raw_datasets.map(
    #     lambda row: {'inputs': tokenizer(row['tokens'], is_split_into_words=True)}
    # )

    res = align_labels(tokenizer, raw_datasets['train'][0] )
    tokenized_datasets = raw_datasets.map(
        lambda row: {'test': align_labels(tokenizer, row)}
    )

    # tokenized_datasets = tokenized_datasets.map(
    #     lambda row: {
    #         'aligned_labels': align_labels_with_tokens(
    #             row['ner_tags'],
    #             row['inputs'].word_ids()
    #         )
    #     }
    # )

    print(tokenized_datasets)
    # from sklearn.model_selection import train_test_split
    # train_texts, val_texts, train_tags, val_tags = train_test_split(texts, tags, test_size=.2)

    # tokenized_datasets = raw_datasets.map(
    #     tokenize_and_align_labels,
    #     batched=True,
    #     remove_columns=raw_datasets["train"].column_names,
    # )
    return None
    # # Load pretrained model and tokenizer
    # # TODO: Q: what the config is used for ?
    # config = AutoConfig.from_pretrained(
    #     model_args.config_name if model_args.config_name else model_args.model_name_or_path,
    #     finetuning_task=data_args.dataset,
    #     cache_dir=model_args.cache_dir,
    #     revision=model_args.model_revision,
    #     use_auth_token=True if model_args.use_auth_token else None,
    # )
    # tokenizer = BertTokenizerFast.from_pretrained(
    #     model_args.tokenizer_name if model_args.tokenizer_name else model_args.model_name_or_path,
    #     cache_dir=model_args.cache_dir,
    #     use_fast=model_args.use_fast_tokenizer,
    #     revision=model_args.model_revision,
    #     use_auth_token=True if model_args.use_auth_token else None,
    # )
    # # tokenizer = AutoTokenizer.from_pretrained(
    # #
    # #     cache_dir=model_args.cache_dir,
    # #     use_fast=model_args.use_fast_tokenizer,
    # #     revision=model_args.model_revision,
    # #     use_auth_token=True if model_args.use_auth_token else None,
    # # )
    #
    # # Padding strategy
    # if data_args.pad_to_max_length:
    #     padding = "max_length"
    # else:
    #     # We will pad later, dynamically at batch creation, to the max sequence length in each batch
    #     padding = False
    #
    # if data_args.max_seq_length > tokenizer.model_max_length:
    #     logger.warning(
    #         f"The max_seq_length passed ({data_args.max_seq_length}) is larger than the maximum length for the"
    #         f"model ({tokenizer.model_max_length}). Using max_seq_length={tokenizer.model_max_length}."
    #     )
    # max_seq_length = min(data_args.max_seq_length, tokenizer.model_max_length)
    #
    # # TODO: change the preprocess_function
    # def preprocess_function(examples):
    #     # Tokenize the texts
    #     # args = (
    #     #     (examples['text'],)
    #     # )
    #
    #     args = (
    #         (examples['words'],)
    #     )
    #     train_texts = examples['words']
    #
    #     data_collator = DataCollatorForTokenClassification(tokenizer=tokenizer)
    #
    #     train_encodings = tokenizer(train_texts, is_split_into_words=True, return_offsets_mapping=True, padding=True,
    #                                 truncation=True)
    #     # result = tokenizer(*args, padding=padding, max_length=max_seq_length, truncation=True)
    #     return None
    #
    #     # Map labels to IDs (not necessary for GLUE tasks)
    #     if "label" in examples:
    #         result["label"] = examples["label"]
    #     return result
    #
    # with training_args.main_process_first(desc="dataset map pre-processing"):
    #     tokenized_datasets = raw_datasets.map(
    #         preprocess_function,
    #         batched=True,
    #         load_from_cache_file=not data_args.overwrite_cache,
    #         desc="Running tokenizer on dataset",
    #     )
    return tokenized_datasets


def train_model(data_args, model_args, training_args, raw_datasets, iteration=0):
    # Load pretrained model and tokenizer
    # TODO: Q: what the config is used for ?
    config = AutoConfig.from_pretrained(
        model_args.config_name if model_args.config_name else model_args.model_name_or_path,
        finetuning_task=data_args.dataset,
        cache_dir=model_args.cache_dir,
        revision=model_args.model_revision,
        use_auth_token=True if model_args.use_auth_token else None,
    )
    tokenizer = AutoTokenizer.from_pretrained(
        model_args.tokenizer_name if model_args.tokenizer_name else model_args.model_name_or_path,
        cache_dir=model_args.cache_dir,
        use_fast=model_args.use_fast_tokenizer,
        revision=model_args.model_revision,
        use_auth_token=True if model_args.use_auth_token else None,
    )
    model_obj = get_model_obj(training_args.model_type)
    model = model_obj.from_pretrained(
        model_args.model_name_or_path,
        from_tf=bool(".ckpt" in model_args.model_name_or_path),
        config=config,
        cache_dir=model_args.cache_dir,
        revision=model_args.model_revision,
        use_auth_token=True if model_args.use_auth_token else None,
    )

    # Make sure datasets are here and select a subset if specified
    if training_args.do_train:
        if TRAIN not in raw_datasets:
            raise ValueError("--do_train requires a train dataset")
        train_dataset = raw_datasets[TRAIN]
        if data_args.max_train_samples is not None:
            train_dataset = train_dataset.shuffle(training_args.seed).select(range(data_args.max_train_samples))

    if training_args.do_eval:
        if TEST not in raw_datasets:  # todo and this
            raise ValueError("--do_eval requires a validation dataset")
        eval_dataset = raw_datasets[TEST]  # todo change this back
        if data_args.max_eval_samples is not None:
            eval_dataset = eval_dataset.shuffle(training_args.seed).select(range(data_args.max_eval_samples))

    if training_args.do_predict:
        if TEST not in raw_datasets:
            raise ValueError("--do_predict requires a test dataset")
        predict_dataset = raw_datasets[TEST]
        if data_args.max_predict_samples is not None:
            predict_dataset = predict_dataset.shuffle(training_args.seed).select(range(data_args.max_predict_samples))

    # Log a few random samples from the training set:
    if training_args.do_train:
        for index in random.sample(range(len(train_dataset)), 3):
            logger.info(f"Sample {index} of the training set: {train_dataset[index]}.")

    compute_metrics = get_compute_metrics(training_args.metrics)

    # Data collator will default to DataCollatorWithPadding, so we change it if we already did the padding.
    if data_args.pad_to_max_length:
        data_collator = default_data_collator
    elif training_args.fp16:
        data_collator = DataCollatorWithPadding(tokenizer, pad_to_multiple_of=8)
    else:
        data_collator = None

    # Initialize our Trainer
    trainer_obj = get_trainer(training_args.trainer_type)
    trainer = trainer_obj(
        model=model,
        args=training_args,
        train_dataset=train_dataset if training_args.do_train else None,
        eval_dataset=eval_dataset if training_args.do_eval else None,
        compute_metrics=compute_metrics,
        tokenizer=tokenizer,
        data_collator=data_collator,
    )

    # Training
    if training_args.do_train:
        checkpoint = None
        if training_args.resume_from_checkpoint is not None:
            checkpoint = training_args.resume_from_checkpoint
        train_result = trainer.train(resume_from_checkpoint=checkpoint)
        metrics = train_result.metrics
        max_train_samples = (
            data_args.max_train_samples if data_args.max_train_samples is not None else len(train_dataset)
        )
        metrics[TRAIN_SAMPLES] = min(max_train_samples, len(train_dataset))

        # trainer.save_model()  # Saves the tokenizer too for easy upload

        trainer.log_metrics(TRAIN, metrics)
        trainer.save_metrics(TRAIN, metrics)
        trainer.save_state()

    # Evaluation
    if training_args.do_eval:
        logger.info("*** Evaluate ***")

        tasks = [data_args.dataset]
        eval_datasets = [eval_dataset]

        for eval_dataset, task in zip(eval_datasets, tasks):
            metrics = trainer.evaluate(eval_dataset=eval_dataset)

            max_eval_samples = (
                data_args.max_eval_samples if data_args.max_eval_samples is not None else len(eval_dataset)
            )
            metrics[EVAL_SAMPLES] = min(max_eval_samples, len(eval_dataset))

            trainer.log_metrics(EVAL, metrics)
            trainer.save_metrics(EVAL, metrics)

    kwargs = {"finetuned_from": model_args.model_name_or_path, "tasks": "text-classification"}
    if data_args.dataset is not None:
        kwargs["language"] = "en"
        kwargs["dataset_tags"] = data_args.dataset
        kwargs["dataset_args"] = data_args.dataset
        kwargs["dataset"] = data_args.dataset.upper()

    if training_args.push_to_hub:
        trainer.push_to_hub(**kwargs)
    else:
        trainer.create_model_card(**kwargs)

    return trainer


def main():
    parser = HfArgumentParser(
        (DataTrainingArguments, ModelArguments, ProjectTrainingArguments),
        description=DESCRIPTION,
    )
    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        # If we pass only one argument to the script and it's the path to a json file,
        # let's parse it to get our arguments.
        data_args, model_args, training_args = parser.parse_json_file(
            json_file=os.path.abspath(sys.argv[1])
        )
    else:
        data_args, model_args, training_args = parser.parse_args_into_dataclasses()

    print(f"config_version: {training_args.config_version}")

    # Set extra arguments here
    if training_args.run_name == AUTO:
        training_args.run_name = f"epochs={training_args.num_train_epochs}_batch={training_args.per_device_train_batch_size}_lr={training_args.learning_rate}"
    if training_args.output_dir == AUTO:
        training_args.output_dir = EXPERIMENTS_DIR / training_args.run_name

    # Setup logging
    setup_logging(data_args, model_args, training_args, logger)
    if training_args.report_to in [WANDB, ALL]:
        os.environ[WANDB_PROJECT] = PROJECT_NAME

    # Set seed before initializing model.
    set_seed(training_args.seed)

    # Load datasets
    # TODO: 1. replace load_dataset with my own dataset
    # https://huggingface.co/docs/datasets/loading
    # https://wandb.ai/biased-ai/Named-Entity%20Recognition%20on%20HuggingFace/reports/Named-Entity-Recognition-on-HuggingFace--Vmlldzo3NTk4NjY
    # https://www.freecodecamp.org/news/getting-started-with-ner-models-using-huggingface/
    # https://www.analyticsvidhya.com/blog/2022/06/how-to-train-an-ner-model-with-huggingface/

    dataset_path = create_dataset(data_args.dataset_path, columns=["text", "ner"], take_first_ner=False)

    # data_args.dataset_path = './data/iahlt-release-2022-06-09/ne/ar_ner_data.jsonl'
    # ner_dataset = load_dataset("json", data_files=data_path)

    # data_args.dataset = ner_dataset['train'].train_test_split(test_size=0.9, seed=42)

    # print(ner_dataset.keys())

    # TODO: Q: Is ner should be performed on sentences that are not tokenized ?
    # raw_datasets = load_dataset(data_args.dataset)
    raw_datasets_dict = load_dataset("json", data_files=dataset_path, )

    # raw_datasets = raw_datasets.select(range(4))

    # raw_datasets_dict = raw_datasets_dict['train'].train_test_split(train_size=0.9, seed=42)

    raw_datasets_dict = raw_datasets_dict['train'].train_test_split(train_size=1, test_size=1, seed=42)

    # https://stackoverflow.com/questions/67852880/how-can-i-handle-this-datasets-to-create-a-datasetdict
    # train_dataset, validation_dataset = raw_datasets['train'].train_test_split(test_size=0.1).values()
    #
    # raw_datasets = DatasetDict({'train': train_dataset, 'val': validation_dataset})

    raw_datasets_dict = preprocess_datasets(data_args, model_args, training_args, raw_datasets_dict)

    # run training
    trainer = train_model(data_args, model_args, training_args, raw_datasets)


if __name__ == "__main__":
    main()

# --output_dir . --metrics accuracy
