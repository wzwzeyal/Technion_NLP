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
import wandb
# from camel_tools.tokenizers.word import simple_word_tokenize
from transformers import (
    HfArgumentParser,
    set_seed,
)
from transformers.trainer_utils import EvaluationStrategy
from transformers.utils.versions import require_version

from args.data_args import DataTrainingArguments
from args.model_args import ModelArguments
from args.training_args import ProjectTrainingArguments
from consts import *
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


def train_model(data_args, model_args, training_args, raw_datasets, iteration=0):
    # Load pretrained model and tokenizer

    train_dataset = NERDataset(
        texts=raw_datasets[TRAIN]['tokens'],
        tags=raw_datasets[TRAIN][data_args.dataset_tag_field],
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

    training_args.evaluate_during_training = True
    training_args.load_best_model_at_end = True
    training_args.evaluation_strategy = EvaluationStrategy.EPOCH
    training_args.disable_tqdm = False

    model_obj = AutoModelForTokenClassification.from_pretrained(
        model_args.model_name_or_path,
        return_dict=True, num_labels=len(raw_datasets.label_list),
        ignore_mismatched_sizes=True
    )

    wandb.init(project=PROJECT_NAME, name="run", config=training_args, entity="nlpcourse")

    trainer = Trainer(
        model=model_obj,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=test_dataset,
        compute_metrics=compute_metrics,

    )

    trainer.train()

    predictions = trainer.predict(test_dataset=test_dataset)
    preds_list, out_label_list = align_predictions(predictions.predictions, predictions.label_ids)

    class_dict = classification_report(out_label_list, preds_list, digits=5, output_dict=True)
    class_df = pd.DataFrame.from_dict(class_dict)

    pd.set_option('colheader_justify', 'center')  # FOR TABLE <th>
    wandb.log({"classification_report_html": wandb.Html(class_df.to_html())})

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
    # https://huggingface.co/docs/datasets/loading
    # https://wandb.ai/biased-ai/Named-Entity%20Recognition%20on%20HuggingFace/reports/Named-Entity-Recognition-on-HuggingFace--Vmlldzo3NTk4NjY
    # https://www.freecodecamp.org/news/getting-started-with-ner-models-using-huggingface/
    # https://www.analyticsvidhya.com/blog/2022/06/how-to-train-an-ner-model-with-huggingface/

    dataset_path = create_dataset(
        data_args.dataset_path,
        columns=[TEXT, NER],
        force_create=data_args.force_create,
    )

    raw_datasets_dict = load_dataset("json", data_files=dataset_path, )

    label_list = {x for label in raw_datasets_dict["train"][data_args.dataset_tag_field] for x in label}

    # in order to get deterministic order
    label_list = sorted(label_list)

    update_inv_label_map(
        {i: label for i, label in enumerate(label_list)}
    )

    raw_datasets_dict = raw_datasets_dict['train'].train_test_split(train_size=data_args.train_size)
    raw_datasets_dict.label_list = label_list

    # run training
    trainer = train_model(data_args, model_args, training_args, raw_datasets_dict)

    trainer.model.save_pretrained(SAVED_MODELS_DIR)


if __name__ == "__main__":
    main()

# ./config/args.json
