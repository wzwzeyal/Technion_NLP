import numpy as np

from seqeval.metrics import accuracy_score, f1_score, precision_score, recall_score, classification_report

import torch
from transformers import (
    AutoModelForSequenceClassification,
    AutoModelForQuestionAnswering,
    AutoModelForTokenClassification,
    Trainer,
    EvalPrediction
)
from trainer import CustomTrainer
from datasets import load_metric

from consts import *


def get_model_obj(model_type):
    if model_type == CLS:
        return AutoModelForSequenceClassification
    elif model_type == QA:
        return AutoModelForQuestionAnswering
    elif model_type == TCLS:
        return AutoModelForTokenClassification
    else:
        raise ValueError(f"Model type {model_type} is not supported. Available types are {ALL_MODEL_TYPES}")


def get_compute_metrics(metrics):
    # Get the metric functions
    metrics = {metric: load_metric(metric) for metric in metrics}

    # You can define your custom compute_metrics function. It takes an `EvalPrediction` object (a namedtuple with a
    # predictions and label_ids field) and has to return a dictionary string to float.
    def compute_metrics(p: EvalPrediction):
        preds = p.predictions[0] if isinstance(p.predictions, tuple) else p.predictions
        preds = np.argmax(preds, axis=1)
        result = {metric: metric_fn.compute(predictions=preds, references=p.label_ids)[metric] for metric, metric_fn in metrics.items()}
        return result

    return compute_metrics

def align_predictions(predictions, label_ids, inv_label_map):
    preds = np.argmax(predictions, axis=2)

    batch_size, seq_len = preds.shape

    out_label_list = [[] for _ in range(batch_size)]
    preds_list = [[] for _ in range(batch_size)]

    for i in range(batch_size):
        for j in range(seq_len):
            if label_ids[i, j] != torch.nn.CrossEntropyLoss().ignore_index:
                out_label_list[i].append(inv_label_map[label_ids[i][j]])
                preds_list[i].append(inv_label_map[preds[i][j]])

    return preds_list, out_label_list

def compute_metrics(p):
    preds_list, out_label_list = align_predictions(p.predictions,p.label_ids)
    #print(classification_report(out_label_list, preds_list,digits=4))
    return {
        "accuracy_score": accuracy_score(out_label_list, preds_list),
        "precision": precision_score(out_label_list, preds_list),
        "recall": recall_score(out_label_list, preds_list),
        "f1": f1_score(out_label_list, preds_list),
    }




def get_trainer(trainer_type):
    if trainer_type == STANDARD:
        return Trainer
    elif trainer_type == CUSTOM:
        return CustomTrainer
    else:
        raise ValueError(f"Trainer type {trainer_type} is not supported. Available types are {ALL_TRAINER_TYPES}")
