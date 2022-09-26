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
import torch.cuda
# from camel_tools.tokenizers.word import simple_word_tokenize
from transformers import (
    HfArgumentParser,
    set_seed,
)
from transformers import TrainingArguments
from transformers.trainer_utils import EvaluationStrategy
from transformers.utils.versions import require_version

from args.data_args import DataTrainingArguments
from args.model_args import ModelArguments
from args.training_args import ProjectTrainingArguments
from consts import *
from final_project.dataset import NERDataset
from utils.data_utils import *
from utils.train_utils import *
from utils.utils import *

# You can also adapt this script on your own text classification task. Pointers for this are left as comments.

# from transformers import DistilBertForTokenClassification, BertForTokenClassification


# Will error if the minimal version of Transformers is not installed. Remove at your own risks.
# check_min_version("4.16.0.dev0")

require_version("datasets>=1.8.0", "To fix: pip install -r examples/pytorch/text-classification/requirements.txt")

logger = logging.getLogger(__name__)


def train_model(data_args, model_args, training_args, raw_datasets, iteration=0):
    # Load pretrained model and tokenizer
    # TODO: Q: what the config is used for ?

    # if data_args.max_train_samples is not None:
    #     train_dataset = train_dataset.shuffle(training_args.seed).select(range(data_args.max_train_samples))

    train_dataset = NERDataset(
        texts=raw_datasets[TRAIN]['tokens'][:1024],
        tags=raw_datasets[TRAIN][data_args.dataset_tag_field][:1024],
        label_list=raw_datasets.label_list,
        model_name=model_args.model_name_or_path,
        max_length=data_args.max_seq_length,
        nof_samples=data_args.max_train_samples
    )

    test_dataset = NERDataset(
        texts=raw_datasets[TEST]['tokens'],
        tags=raw_datasets[TEST][data_args.dataset_tag_field],
        label_list=raw_datasets.label_list,
        model_name=model_args.model_name_or_path,
        max_length=data_args.max_seq_length,
        nof_samples=data_args.max_eval_samples
    )

    training_args = TrainingArguments("train_1_1")
    training_args.evaluate_during_training = True
    training_args.adam_epsilon = 1e-8
    training_args.learning_rate = 5e-5
    training_args.fp16 = True
    # training_args.per_device_train_batch_size = 4
    # training_args.per_device_eval_batch_size = 1
    training_args.auto_find_batch_size = True
    training_args.gradient_accumulation_steps = 2
    training_args.num_train_epochs = 2
    training_args.load_best_model_at_end = True
    training_args.output_dir = './results'

    steps_per_epoch = (len(raw_datasets[TRAIN]) // (
            training_args.per_device_train_batch_size * training_args.gradient_accumulation_steps))
    total_steps = steps_per_epoch * training_args.num_train_epochs
    print(steps_per_epoch)
    print(total_steps)
    # Warmup_ratio
    warmup_ratio = 0.1
    training_args.warmup_steps = total_steps * warmup_ratio

    training_args.evaluation_strategy = EvaluationStrategy.EPOCH
    # training_args.logging_steps = 200
    # training_args.save_steps = 100000  # don't want to save any model
    training_args.seed = 42
    training_args.disable_tqdm = False
    training_args.lr_scheduler_type = 'cosine'

    # classes = raw_datasets['classes']

    # model_config = AutoConfig.from_pretrained(
    #     model_args.config_name if model_args.config_name else model_args.model_name_or_path,
    #     finetuning_task=data_args.dataset,
    #     cache_dir=model_args.cache_dir,
    #     revision=model_args.model_revision,
    #     use_auth_token=True if model_args.use_auth_token else None,
    #     num_labels=len(classes)
    #
    # )
    # tokenizer = AutoTokenizer.from_pretrained(
    #     model_args.tokenizer_name if model_args.tokenizer_name else model_args.model_name_or_path,
    #     cache_dir=model_args.cache_dir,
    #     use_fast=model_args.use_fast_tokenizer,
    #     revision=model_args.model_revision,
    #     use_auth_token=True if model_args.use_auth_token else None,
    # )

    model_name = model_args.model_name_or_path

    model_obj = AutoModelForTokenClassification.from_pretrained(
        model_name,
        return_dict=True, num_labels=len(raw_datasets.label_list),
    )

    trainer = Trainer(
        model=model_obj,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=test_dataset,
        compute_metrics=compute_metrics,
    )

    trainer.train()

    model = trainer.model

    print(type(model))
    # model_obj = get_model_obj(training_args.model_type)
    #
    # # model = AutoModelForTokenClassification.from_pretrained("bert-base-multilingual-cased", num_labels=len(label_names))
    #
    # model = model_obj.from_pretrained(
    #     model_args.model_name_or_path,
    #     from_tf=bool(".ckpt" in model_args.model_name_or_path),
    #     config=model_config,
    #     cache_dir=model_args.cache_dir,
    #     revision=model_args.model_revision,
    #     use_auth_token=True if model_args.use_auth_token else None,
    # )
    #
    # # Make sure datasets are here and select a subset if specified
    # if training_args.do_train:
    #     if TRAIN not in raw_datasets:
    #         raise ValueError("--do_train requires a train dataset")
    #     train_dataset = raw_datasets[TRAIN]
    #     if data_args.max_train_samples is not None:
    #         train_dataset = train_dataset.shuffle(training_args.seed).select(range(data_args.max_train_samples))
    #
    # if training_args.do_eval:
    #     if TEST not in raw_datasets:  # todo and this
    #         raise ValueError("--do_eval requires a validation dataset")
    #     eval_dataset = raw_datasets[TEST]  # todo change this back
    #     if data_args.max_eval_samples is not None:
    #         eval_dataset = eval_dataset.shuffle(training_args.seed).select(range(data_args.max_eval_samples))
    #
    # if training_args.do_predict:
    #     if TEST not in raw_datasets:
    #         raise ValueError("--do_predict requires a test dataset")
    #     predict_dataset = raw_datasets[TEST]
    #     if data_args.max_predict_samples is not None:
    #         predict_dataset = predict_dataset.shuffle(training_args.seed).select(range(data_args.max_predict_samples))
    #
    # # Log a few random samples from the training set:
    # if training_args.do_train:
    #     for index in random.sample(range(len(train_dataset)), 1):
    #         logger.info(f"Sample {index} of the training set: {train_dataset[index]}.")
    #
    # compute_metrics = get_compute_metrics(training_args.metrics)
    #
    # data_collator = DataCollatorForTokenClassification(tokenizer)
    #
    # # Data collator will default to DataCollatorWithPadding, so we change it if we already did the padding.
    # if data_args.pad_to_max_length:
    #     data_collator = default_data_collator
    # elif training_args.fp16:
    #     data_collator = DataCollatorWithPadding(tokenizer, pad_to_multiple_of=8)
    # else:
    #     data_collator = None
    #
    # data_collator = DataCollatorForTokenClassification(tokenizer)
    #
    # # TODO: remove columns
    # # token_type_ids, labels, input_ids, att_ms
    # # train_dataset = train_dataset.remove_columns("label")
    #
    # # TODO: repeat for eval
    #
    # # TODO: check unk (there should not be since wordpiece is in characters)
    # # TODO: UD (normalize) if there are a lot of unk unicode
    #
    # # train_dataset['input_ids'] = train_dataset['input_ids'].squeeze(0)
    #
    # # Initialize our Trainer
    #
    # trainer_obj = get_trainer(training_args.trainer_type)
    # trainer = trainer_obj(
    #     model=model,
    #     args=training_args,
    #     train_dataset=train_dataset if training_args.do_train else None,
    #     eval_dataset=eval_dataset if training_args.do_eval else None,
    #     compute_metrics=compute_metrics,
    #     tokenizer=tokenizer,
    #     data_collator=data_collator,
    # )
    #
    # # Training
    # if training_args.do_train:
    #     checkpoint = None
    #     if training_args.resume_from_checkpoint is not None:
    #         checkpoint = training_args.resume_from_checkpoint
    #     train_result = trainer.train(resume_from_checkpoint=checkpoint)
    #     metrics = train_result.metrics
    #     max_train_samples = (
    #         data_args.max_train_samples if data_args.max_train_samples is not None else len(train_dataset)
    #     )
    #     metrics[TRAIN_SAMPLES] = min(max_train_samples, len(train_dataset))
    #
    #     # trainer.save_model()  # Saves the tokenizer too for easy upload
    #
    #     trainer.log_metrics(TRAIN, metrics)
    #     trainer.save_metrics(TRAIN, metrics)
    #     trainer.save_state()
    #
    # # Evaluation
    # if training_args.do_eval:
    #     logger.info("*** Evaluate ***")
    #
    #     tasks = [data_args.dataset]
    #     eval_datasets = [eval_dataset]
    #
    #     for eval_dataset, task in zip(eval_datasets, tasks):
    #         metrics = trainer.evaluate(eval_dataset=eval_dataset)
    #
    #         max_eval_samples = (
    #             data_args.max_eval_samples if data_args.max_eval_samples is not None else len(eval_dataset)
    #         )
    #         metrics[EVAL_SAMPLES] = min(max_eval_samples, len(eval_dataset))
    #
    #         trainer.log_metrics(EVAL, metrics)
    #         trainer.save_metrics(EVAL, metrics)
    #
    # kwargs = {"finetuned_from": model_args.model_name_or_path, "tasks": "text-classification"}
    # if data_args.dataset is not None:
    #     kwargs["language"] = "en"
    #     kwargs["dataset_tags"] = data_args.dataset
    #     kwargs["dataset_args"] = data_args.dataset
    #     kwargs["dataset"] = data_args.dataset.upper()
    #
    # if training_args.push_to_hub:
    #     trainer.push_to_hub(**kwargs)
    # else:
    #     trainer.create_model_card(**kwargs)

    return trainer


def main():
    torch.cuda.empty_cache()
    parser = HfArgumentParser(
        (DataTrainingArguments, ModelArguments, ProjectTrainingArguments),
        description=DESCRIPTION,
    )
    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        # If we pass only one argument to the script, and it's the path to a json file,
        # let's parse it to get our arguments.
        data_args, model_args, training_args = parser.parse_json_file(
            json_file=os.path.abspath(sys.argv[1])
        )
    else:
        data_args, model_args, training_args = parser.parse_args_into_dataclasses()

    print(f"config_version: {training_args.config_version}")

    os.environ["TOKENIZERS_PARALLELISM"] = str(model_args.tokenizers_parallelism).lower()

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

    # TODO: what about nested ner (take_first_ner=False)
    dataset_path = create_dataset(
        data_args.dataset_path,
        columns=["text", "ner"],
        pre_process_ner_tags=True,
        force_create=data_args.force_create
    )

    # data_args.dataset_path = './data/iahlt-release-2022-06-09/ne/ar_ner_data.jsonl'
    # ner_dataset = load_dataset("json", data_files=data_path)

    # data_args.dataset = ner_dataset['train'].train_test_split(test_size=0.9, seed=42)

    # print(ner_dataset.keys())

    # TODO: Q: Is ner should be performed on sentences that are not tokenized ?
    # raw_datasets = load_dataset(data_args.dataset)
    raw_datasets_dict = load_dataset("json", data_files=dataset_path, )

    print(data_args.dataset_tag_field)
    label_list = {x for l in raw_datasets_dict["train"][data_args.dataset_tag_field] for x in l}

    update_inv_label_map(
        {i: label for i, label in enumerate(label_list)}
    )

    raw_datasets_dict = raw_datasets_dict['train'].train_test_split(train_size=0.9, seed=42)

    raw_datasets_dict.label_list = sorted(label_list)

    # raw_datasets_dict = raw_datasets_dict['train'].train_test_split(train_size=1, test_size=1, seed=42)

    # https://stackoverflow.com/questions/67852880/how-can-i-handle-this-datasets-to-create-a-datasetdict
    # train_dataset, validation_dataset = raw_datasets['train'].train_test_split(test_size=0.1).values()
    #
    # raw_datasets = DatasetDict({'train': train_dataset, 'val': validation_dataset})

    # raw_datasets_dict = preprocess_datasets(data_args, model_args, training_args, raw_datasets_dict)

    # run training
    trainer = train_model(data_args, model_args, training_args, raw_datasets_dict)


if __name__ == "__main__":
    main()

# --output_dir . --metrics accuracy
