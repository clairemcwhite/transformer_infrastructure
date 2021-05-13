#!/usr/bin/env python

#from transformer_infrastructure.hf_utils import get_sequencelabel_tags, SS3Dataset
import torch
from torch.utils.data import DataLoader
from sentence_transformers import losses, util
from sentence_transformers import LoggingHandler, SentenceTransformer, models, evaluation
from sentence_transformers.readers import InputExample
import logging
#from datetime import datetime
#import csv
import os
import pandas as pd
#from zipfile import ZipFile
import random
import argparse
import re

def get_args():

    parser = argparse.ArgumentParser()
    parser.add_argument("-m", "--model", dest = "model_path", type = str, required = True, 
                        help="Model directory path or name on huggingface. Ex. /path/to/model_dir Rostlab/prot_bert_bfd")
    parser.add_argument("-tr", "--train", dest = "train_path", type = str, required = True, 
                        help="Path to training set, containing columns named sequence1,sequence2,id1,id2,label (set label colname with --label_col) (csv)")
    parser.add_argument("-d", "--dev", dest = "dev_path", type = str, required = True, 
                        help="Path to dev/validation set (used during training), containing columns named sequence1,sequence2,id1,id2,label (set label colname with --label_col) (csv)")
    parser.add_argument("-te", "--test", dest = "test_path", type = str, required = True, 
                        help="Path to withheld test set (used after training), containing columns named sequence1,sequence2,id1,id2,label (set label colname with --label_col) (csv)")
    parser.add_argument("-o", "--outdir", dest = "outdir", type = str, required = True, 
                        help="Name of output directory")
    parser.add_argument("-maxl", "--maxseqlength", dest = "max_length", type = int, required = False, default = 1024, 
                        help="Truncate all sequences to this length (default 1024). Reduce if memory errors")
    parser.add_argument("-n", "--expname", dest = "expname", type = str, required = False, default = "transformer_run",
                        help="Experiment name, used for logging, default = transformer_run")
    parser.add_argument("-e", "--epochs", dest = "epochs", type = int, required = False, default = 10, 
                        help="Number of epochs. Increasing can help if memory error")
    parser.add_argument("-tbsize", "--train_batchsize", dest = "train_batchsize", type = int, required = False, default = 10, 
                        help="Per device train batchsize. Reduce along with val batch size if memory error")
    parser.add_argument("-vbsize", "--dev_batchsize", dest = "dev_batchsize", type = int, required = False, default = 10, 
                        help="Per device validation batchsize. Reduce if memory error")
    parser.add_argument("-l", "--label_col", dest = "label_col", type = str, required = False, default = "label", 
                        help="Name of column in datasets to use as labl, default: label")
  
    args = parser.parse_args()
    return(args)


def load_dataset_pairs(path, max_length, label_column):

        df  = pd.read_csv(path)
      
        df['seq1_fixed'] = ["".join(seq.split()) for seq in df['sequence1']]
        df['seq1_fixed'] = [re.sub(r"[UZOB]", "X", seq) for seq in df['seq1_fixed']]
        seqs1 = [ list(seq)[:max_length-2] for seq in df['seq1_fixed']]

        df['seq2_fixed'] = ["".join(seq.split()) for seq in df['sequence2']]
        df['seq2_fixed'] = [re.sub(r"[UZOB]", "X", seq) for seq in df['seq2_fixed']]
        seqs2 = [ list(seq)[:max_length-2] for seq in df['seq2_fixed']]

        labels = list(df[label_column]) # ex. label

        ids1 = list(df['id1'])
        ids2 = list(df['id1'])
        assert len(seqs1) == len(seqs2) == len(labels) == len(ids1) == len(ids2)
        return seqs1, seqs2, labels, ids1, ids2


def encode_tags(tags, tag2id):

    encoded_labels = [tag2id[tag] for tag in tags]
    return encoded_labels


if __name__ == "__main__":



    args = get_args()

    model_path = args.model_path
    max_length = args.max_length
    train_path = args.train_path
    test_path = args.test_path
    dev_path = args.dev_path

    label_column = args.label_col
    log_format = "%(asctime)s::%(levelname)s::%(name)s::"\
             "%(filename)s::%(lineno)d::%(message)s"

    expname = args.expname
    outdir = args.outdir
    os.makedirs(outdir, exist_ok=True)
    # FIX logname in other scripts
    logname = outdir + "/" + expname + ".log"
    print("logging at ", logname)
    logging.basicConfig(filename=logname, level='DEBUG', format=log_format)

    logging.info("Check for torch")
    logging.info(torch.cuda.is_available())

    epochs = args.epochs
    train_batchsize = args.train_batchsize
    dev_batchsize = args.dev_batchsize
 
    #return seqs1, seqs2, labels, ids1, ids2
    train_seqs1, train_seqs2, train_labels, train_ids1, train_ids2 = load_dataset_pairs(train_path, max_length, label_column)
    dev_seqs1, dev_seqs2, dev_labels, dev_ids1, dev_ids2 = load_dataset_pairs(dev_path, max_length, label_column)
    test_seqs1, test_seqs2, test_labels, test_ids1, test_ids2 = load_dataset_pairs(test_path, max_length, label_column)

    logging.info("datasets loaded")



    #train_examples = [InputExample(texts=['My first sentence', 'My second sentence'], label=0.8),
    #InputExample(texts=['Another pair', 'Unrelated sentence'], label=0.3)]
    #train_dataloader = DataLoader(train_examples, shuffle=True, batch_size=16)    
    #labels_encodings = encode_tags(labels, tag2id)
    train_samples = []
    for i in range(len(train_seqs1)):
         if train_labels[i] == 1:
            train_samples.append(InputExample(texts=[train_seqs1[i], train_seqs2[i]], label = train_labels[i]))

    train_dataloader = DataLoader(train_samples, shuffle=True, batch_size=train_batchsize)  

    print("Data loaded")
    # Convert transformer to sentence transformer model
    word_embedding_model = models.Transformer(model_path)
    # Default pooling strategy
    pooling_model = models.Pooling(word_embedding_model.get_word_embedding_dimension())

    model = SentenceTransformer(modules=[word_embedding_model, pooling_model])
    print("SentenceTransformer model created")


    train_loss = losses.MultipleNegativesRankingLoss(model)

    # Set up a set of different performatnce evaluators
 
    evaluators = []
   
    ###### Classification ######
    # Given (quesiton1, question2), is this a duplicate or not?
    # The evaluator will compute the embeddings for both questions and then compute
    # a cosine similarity. If the similarity is above a threshold, we have a duplicate.
    binary_acc_evaluator = evaluation.BinaryClassificationEvaluator(dev_seqs1, dev_seqs2, dev_labels)
    evaluators.append(binary_acc_evaluator)
 
    binary_acc_evaluator = evaluation.BinaryClassificationEvaluator(train_seqs1, train_seqs2, train_labels)
    evaluators.append(binary_acc_evaluator)
   
    logging.info("binary acc evaluator added")
    dev_seq_dict = {}
    dev_duplicates = []

    # create dict of id:seq
    for i in range(len(dev_seqs1)):
       dev_seq_dict[dev_ids1[i]] = dev_seqs1[i]
       dev_seq_dict[dev_ids2[i]] = dev_seqs2[i]
    # Create pairs list of duplicate ids
    for i in range(len(dev_seqs1)):
       if dev_labels[i] == 1:
         dev_duplicates.append([dev_ids1[i], dev_ids2[i]])


    # The ParaphraseMiningEvaluator computes the cosine similarity between all sentences and
    # extracts a list with the pairs that have the highest similarity. Given the duplicate
    # information in dev_duplicates, it then computes and F1 score how well our duplicate mining worked
    paraphrase_mining_evaluator = evaluation.ParaphraseMiningEvaluator(dev_seq_dict, dev_duplicates, name='dev', show_progress_bar = True)
    evaluators.append(paraphrase_mining_evaluator)

    logging.info("paraphrase evaluator added")
   
    seq_evaluator = evaluation.SequentialEvaluator(evaluators, main_score_function=lambda scores: scores[-1])

    os.makedirs(outdir, exist_ok=True)
    logging.info("Evaluate model without training")
    seq_evaluator(model, epoch=0, steps=0, output_path=outdir)

    model.fit(train_objectives=[(train_dataloader, train_loss)],
          evaluator=seq_evaluator,
          epochs=epochs,
          warmup_steps=1000,
          output_path=outdir
          )




