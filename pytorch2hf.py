from typing import Union, List
from transformer_infrastructure.hf_embed import parse_fasta_for_embed
from transformers import BertTokenizerFast, AdamW, BertModel, get_linear_schedule_with_warmup, PreTrainedModel, PretrainedConfig, BertConfig, BertPreTrainedModel, TrainingArguments, EarlyStoppingCallback, EvalPrediction

from Bio import SeqIO
from Bio.Align import MultipleSeqAlignment
from Bio import AlignIO
from Bio.Seq import Seq
from torch.utils.data import Dataset, TensorDataset, DataLoader, RandomSampler, SequentialSampler
from sentence_transformers import LoggingHandler, SentenceTransformer, models, evaluation
import argparse
import numpy as np
import torch
from torch import nn

import re

from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

import random
import os
import pandas as pd
import copy
import json


from pytorch_lightning import LightningModule
from torchmetrics import Accuracy
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import EarlyStopping
import huggingface_hub

 
def get_aasim_args():



    parser = argparse.ArgumentParser()
    parser.add_argument("-m", "--model", dest = "model_path", type = str, required = True,
                        help="Model directory path or name on huggingface. Ex. /path/to/model_dir Rostlab/prot_bert_bfd")

    parser.add_argument("-o", "--outdir", dest = "outdir", type = str, required = True,
                        help="Output directory to save final model")


    args = parser.parse_args()
    return(args)

 
if __name__ == "__main__":



    args = get_aasim_args()

    model_name = args.model_path
