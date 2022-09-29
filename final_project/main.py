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
from transformers.integrations import WandbCallback
from transformers.trainer_utils import EvaluationStrategy
from transformers.utils.versions import require_version

from args.data_args import DataTrainingArguments
from args.model_args import ModelArguments
from args.training_args import ProjectTrainingArguments
from dataset import NERDataset
from utils.data_utils import *
from utils.train_utils import *
from utils.utils import *

# You can also adapt this script on your own text classification task. Pointers for this are left as comments.

# from transformers import DistilBertForTokenClassification, BertForTokenClassification


# Will error if the minimal version of Transformers is not installed. Remove at your own risks.
# check_min_version("4.16.0.dev0")

require_version("datasets>=1.8.0", "To fix: pip install -r examples/pytorch/text-classification/requirements.txt")

logger = logging.getLogger(__name__)


class MyWandbCallback(WandbCallback):
    def on_train_end(self, args, state, control, model=None, tokenizer=None, **kwargs):
        super(MyWandbCallback, self).on_train_end(args, state, control, model, tokenizer, **kwargs)
        self._wandb.log({"test": args.seed})


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
        nof_samples=data_args.max_train_samples,
        is_perform_word_cleaning=data_args.is_perform_word_cleaning
    )

    test_dataset = NERDataset(
        texts=raw_datasets[TEST]['tokens'],
        tags=raw_datasets[TEST][data_args.dataset_tag_field],
        label_list=raw_datasets.label_list,
        model_name=model_args.model_name_or_path,
        max_length=data_args.max_seq_length,
        nof_samples=data_args.max_eval_samples,
        is_perform_word_cleaning=data_args.is_perform_word_cleaning
    )

    # training_args = TrainingArguments("train_1_1")

    # training_args.adam_epsilon = 1e-8
    # training_args.learning_rate = 5e-5
    # training_args.fp16 = True
    # training_args.auto_find_batch_size = True
    # training_args.gradient_accumulation_steps = 2
    # training_args.num_train_epochs = 3
    # training_args.load_best_model_at_end = True
    # training_args.output_dir = './results'

    # steps_per_epoch = (len(raw_datasets[TRAIN]) // (
    #         training_args.per_device_train_batch_size * training_args.gradient_accumulation_steps))
    # total_steps = steps_per_epoch * training_args.num_train_epochs
    # print(steps_per_epoch)
    # print(total_steps)
    # # Warmup_ratio
    # warmup_ratio = 0.1
    # training_args.warmup_steps = total_steps * warmup_ratio
    training_args.evaluate_during_training = True
    training_args.load_best_model_at_end = True
    training_args.evaluation_strategy = EvaluationStrategy.EPOCH
    training_args.disable_tqdm = False

    model_obj = AutoModelForTokenClassification.from_pretrained(
        model_args.model_name_or_path,
        return_dict=True, num_labels=len(raw_datasets.label_list),
        ignore_mismatched_sizes=True
    )

    trainer = Trainer(
        model=model_obj,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=test_dataset,
        compute_metrics=compute_metrics,
    )

    trainer.add_callback(MyWandbCallback())

    trainer.train()

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
    # os.environ["WANDB_LOG_MODEL"] = "True"

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
        force_create=data_args.force_create,
        is_return_per_unk=data_args.is_return_per_unk
    )

    # data_args.dataset_path = './data/iahlt-release-2022-06-09/ne/ar_ner_data.jsonl'
    # ner_dataset = load_dataset("json", data_files=data_path)

    # data_args.dataset = ner_dataset['train'].train_test_split(test_size=0.9, seed=42)

    # print(ner_dataset.keys())

    # TODO: Q: Is ner should be performed on sentences that are not tokenized ?
    # raw_datasets = load_dataset(data_args.dataset)
    raw_datasets_dict = load_dataset("json", data_files=dataset_path, )

    print(data_args.dataset_tag_field)
    label_list = {x for label in raw_datasets_dict["train"][data_args.dataset_tag_field] for x in label}

    update_inv_label_map(
        {i: label for i, label in enumerate(label_list)}
    )

    raw_datasets_dict = raw_datasets_dict['train'].train_test_split(train_size=data_args.train_size)

    raw_datasets_dict.label_list = sorted(label_list)

    # run training
    trainer = train_model(data_args, model_args, training_args, raw_datasets_dict)

    trainer.model.save_pretrained(SAVED_MODELS_DIR)


if __name__ == "__main__":
    main()

# --output_dir . --metrics accuracy
