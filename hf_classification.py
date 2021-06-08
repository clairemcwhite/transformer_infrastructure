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

def get_args():

    parser = argparse.ArgumentParser()
    parser.add_argument("-m", "--model", dest = "model_name", type = str, required = True, 
                        help="Model directory path or name on huggingface. Ex. /path/to/model_dir Rostlab/prot_bert_bfd")
    parser.add_argument("-tr", "--train", dest = "train_path", type = str, required = True, 
                        help="Path to training set, containing columns named sequence,label (csv)")
    parser.add_argument("-v", "--val", dest = "val_path", type = str, required = True, 
                        help="Path to validation set (used during training), containing columns named sequence,label (csv)")
    parser.add_argument("-te", "--test", dest = "test_path", type = str, required = True, 
                        help="Path to withheld test set (used after training), containing columns named sequence,label (csv)")
    parser.add_argument("-o", "--outdir", dest = "outdir", type = str, required = True, 
                        help="Name of output directory")
    parser.add_argument("-maxl", "--maxseqlength", dest = "max_length", type = int, required = False, default = 1024, 
                        help="Truncate all sequences to this length (default 1024). Reduce if memory errors")
    parser.add_argument("-n", "--expname", dest = "expname", type = str, required = False, default = "transformer_run",
                        help="Experiment name, used for logging, default = transformer_run")
    parser.add_argument("-c", "--checkpoint", dest = "checkpoint", type = str, required = False,
                        help="Checkpoint directory to continue training")
    parser.add_argument("-e", "--epochs", dest = "epochs", type = int, required = False, default = 10, 
                        help="Number of epochs. Increasing can help if memory error")
    parser.add_argument("-tbsize", "--train_batchsize", dest = "train_batchsize", type = int, required = False, default = 10, 
                        help="Per device train batchsize. Reduce along with val batch size if memory error")
    parser.add_argument("-vbsize", "--val_batchsize", dest = "val_batchsize", type = int, required = False, default = 10, 
                        help="Per device validation batchsize. Reduce if memory error")

 

    
    args = parser.parse_args()
    return(args)


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


def setup_trainer(epochs, train_batchsize, val_batchsize, outdir, expname):


    training_args = TrainingArguments(
        output_dir=outdir,          # output directory
        num_train_epochs=epochs,              # total number of training epochs
        per_device_train_batch_size=train_batchsize,   # batch size per device during training
        per_device_eval_batch_size=val_batchsize,   # batch size for evaluation
        warmup_steps=200,                # number of warmup steps for learning rate scheduler
        learning_rate=3e-05,             # learning rate
        weight_decay=0.0,                # strength of weight decay
        logging_dir='./logs',            # directory for storing logs
        logging_steps=200,               # How often to print logs
        do_train=True,                   # Perform training
        do_eval=True,                    # Perform evaluation
        evaluation_strategy="epoch",     # evalute after each epoch
        gradient_accumulation_steps=32,  # total number of steps before back propagation
        #fp16=True,                       # Use mixed precision
        #fp16_opt_level="02",             # mixed precision mode
        run_name=expname,      # experiment name
        seed=3,                         # Seed for experiment reproducibility
        load_best_model_at_end=True,
        metric_for_best_model="eval_accuracy",
        greater_is_better=True,
    
    )

    return(training_args)


if __name__ == "__main__":


 

    #max_length = 1024
    #train_path = '/home/jupyter/chloro_loc/chloro_labeledsetTrain.csv'
    #test_path = '/home/jupyter/chloro_loc/chloro_labeledsetTest.csv'
    #val_path = '/home/jupyter/chloro_loc/chloro_labeledsetVal.csv'
    #   if args.n_gpu > 1#:
    #    model = torch.nn.DataParallel(model)

    args = get_args()

    model_name = args.model_name
    max_length = args.max_length
    train_path = args.train_path
    test_path = args.test_path
    val_path = args.val_path

    log_format = "%(asctime)s::%(levelname)s::%(name)s::"\
             "%(filename)s::%(lineno)d::%(message)s"

    expname = args.expname
    outdir = args.outdir
    logname = outdir + expname + "_" + model_name.strip("/") + ".log"
    print("logging at ", logname)
    logging.basicConfig(filename=logname, level='DEBUG', format=log_format)

    logging.info("Check for torch")
    logging.info(torch.cuda.is_available())

    epochs = args.epochs
    checkpoint = args.checkpoint
    train_batchsize = args.train_batchsize
    val_batchsize = args.val_batchsize
 

    train_seqs, train_labels, train_ids = load_dataset(train_path, max_length)
    val_seqs, val_labels, val_ids = load_dataset(test_path, max_length)
    test_seqs, test_labels, test_ids = load_dataset(val_path, max_length)

    logging.info("datasets loaded")


    seq_tokenizer = BertTokenizerFast.from_pretrained(model_name, do_lower_case=False)
    logging.info("sequences tokenizer loaded")



    train_seqs_encodings = seq_tokenizer(train_seqs, is_split_into_words=True, return_offsets_mapping=True, truncation=True, padding=True)
    val_seqs_encodings = seq_tokenizer(val_seqs, is_split_into_words=True, return_offsets_mapping=True, truncation=True, padding=True)
    test_seqs_encodings = seq_tokenizer(test_seqs, is_split_into_words=True, return_offsets_mapping=True, truncation=True, padding=True)
    logging.info("sequences tokenized")

    # Consider each label as a tag for each token
    unique_tags = set(train_labels)
    unique_tags  = sorted(list(unique_tags))  # make the order of the labels unchanged
    tag2id = {tag: id for id, tag in enumerate(unique_tags)}
    id2tag = {id: tag for tag, id in tag2id.items()}

    logging.info("unique_tags")
    logging.info(unique_tags)
    logging.info("id2tag")
    logging.info(id2tag)
    logging.info("tag2id")
    logging.info(tag2id)


    train_labels_encodings = encode_tags(train_labels, tag2id)
    val_labels_encodings = encode_tags(val_labels,  tag2id)
    test_labels_encodings = encode_tags(test_labels,  tag2id)
    logging.info("labels encoded")


    _ = train_seqs_encodings.pop("offset_mapping")
    _ = val_seqs_encodings.pop("offset_mapping")
    _ = test_seqs_encodings.pop("offset_mapping")
    logging.info("offset_mapping popped")


    train_dataset = SS3Dataset(train_seqs_encodings, train_labels_encodings)
    val_dataset = SS3Dataset(val_seqs_encodings, val_labels_encodings)
    test_dataset = SS3Dataset(test_seqs_encodings, test_labels_encodings)
    logging.info("SS3 datasets constructed")

    training_args = setup_trainer(epochs, train_batchsize, val_batchsize, outdir, expname)

    gc.collect() 

    trainer = Trainer(
        model_init=model_init,                # the instantiated Transformers model to be trained
        args=training_args,                   # training arguments, defined above
        train_dataset=train_dataset,          # training dataset
        eval_dataset=val_dataset,             # evaluation dataset
        compute_metrics = compute_metrics,    # evaluation metrics
    )


    logging.info("trainer initiated")
   
    if args.checkpoint:
        trainer.train(checkpoint)
    else:
        trainer.train()


    logging.info("training complete")
    trainer.save_model(outdir)
    seq_tokenizer.save_pretrained(outdir)
  
    logging.info("model saved")
    logging.info(outdir)


    get_predictions(outdir, test_path, len(tag2id), max_length, "withheldtest")
    get_predictions(outdir, val_path, len(tag2id), max_length, "val")
    get_predictions(outdir, train_path, len(tag2id), max_length, "train")
    #test_predictions, test_label_ids, test_metrics = trainer.predict(test_dataset)
    #logging.info("test metrics (withheld)")
    #logging.info(test_metrics)
    #outtest  = outdir + "/" + expname + "_test_predictions.csv"
    #np.savetxt(outtest, test_predictions, delimiter=',')

    #train_predictions, train_label_ids, train_metrics = trainer.predict(train_dataset)
    #logging.info("train metrics")
    #logging.info(train_metrics)
    #outtrain  = outdir + "/" + expname + "_train_predictions.csv"
    #np.savetxt(outtrain, train_predictions, delimiter=',')


    #val_predictions, val_label_ids, val_metrics = trainer.predict(val_dataset)
    #logging.info("val metrics (seen during training)")
    #logging.info(val_metrics)
    #outval  = outdir + "/" + expname + "_val_predictions.csv"
    #np.savetxt(outval, val_predictions, delimiter=',')

 



    #idx = 2
    #sample_ground_truth = test_dataset[idx]['labels']
    #sample_predictions =  np.argmax(predictions[idx])
    #
    #
    ## In[57]:
    #
    #
    #sample_sequence = seq_tokenizer.decode(list(test_dataset[idx]['input_ids']), skip_special_tokens=True)
    #
    #
    ## In[58]:
    #
    #
    #
    #print(sample_ground_truth)
    #print(sample_predictions)
    #
    #
    ## In[41]:
    #
    #
    #print("Sequence       : {} \nGround Truth is: {}\nprediction is  : {}".format(#sample_sequence,
    #                                                                      sample_ground_truth,
    #                                                                      # Remove the first token on prediction becuase its CLS token
    #                                                                      # and only show up to the input length
    #                                                                      sample_predictions))
    #      
    #
    #
    ## **14. Save the model**
    #
    ## **15. Check Tensorboard**
    #
    ## In[ ]:
    #
    #
    #get_ipython().run_line_magic('load_ext', 'tensorboard')
    #get_ipython().run_line_magic('tensorboard', '--logdir logs')
    #
