#!/usr/bin/env python

from transformer_infrastructure.hf_evaluation import get_predictions
import torch
from transformers import AutoTokenizer, Trainer, TrainingArguments, AutoModelForSequenceClassification, BertTokenizerFast, EvalPrediction, AutoConfig
from torch.utils.data import Dataset
import os
import pandas as pd
import requests
from tqdm.auto import tqdm
import numpy as np
from seqeval.metrics import accuracy_score, f1_score, precision_score, recall_score
import re
import argparse
import logging
import gc 

def load_dataset(path, max_length):
        df  = pd.read_csv(path)
      
        df['seq_fixed'] = ["".join(seq.split()) for seq in df['sequence']]
        df['seq_fixed'] = [re.sub(r"[UZOB]", "X", seq) for seq in df['seq_fixed']]
        seqs = [ list(seq)[:max_length-2] for seq in df['seq_fixed']]

        labels = list(df['label'])

        ids = list(df['Entry_name'])
        assert len(seqs) == len(labels) == len(ids)
        return seqs, labels, ids


def encode_tags(tags, tag2id):

    encoded_labels = [tag2id[tag] for tag in tags]
    return encoded_labels


class SS3Dataset(Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels)

def compute_metrics(pred):
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)
    # calculate accuracy using sklearn's function
    return {
    "accuracy" : accuracy_score(labels, preds),
    #"precision" : precision_score(labels, preds),
    #"recall" : recall_score(labels, preds),
    #"f1"  : f1_score(labels, preds)

    }

def model_init():

  # from_config (vs. from_pretrained) is necessary for multiclass 
  config = AutoConfig.from_pretrained(model_name)
  config.num_labels = len(unique_tags)
  config.id2label = id2tag
  config.label2id = tag2id
  config.gradient_checkpointing = False
  return AutoModelForSequenceClassification.from_config(config)
  #return AutoModelForSequenceClassification.from_pretrained(model_name,
                                                       #  num_labels=len(unique_tags),
                                                       #  id2label=id2tag,
                                                       #  label2id=tag2id,
                                                       #  gradient_checkpointing=False)


