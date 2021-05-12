#!/usr/bin/env python

from transformer_infrastructure.hf_classification import *
 
from transformers import AutoConfig

from torch.utils.data import DataLoader

import torch.nn.functional as F
from scipy.special import logsumexp 
from sklearn.metrics import confusion_matrix, classification_report, precision_recall_curve

import argparse
import pandas as pd

def softmax(x, axis=None):
     return np.exp(x - logsumexp(x, axis=axis, keepdims=True))


def validation(dataloader, model, device_, true_index):
  r"""Validation function to evaluate model performance on a 
  separate set of data.
 
  This function will return the true and predicted labels so we can use later
  to evaluate the model's performance.
 
  This function is built with reusability in mind: it can be used as is as long
    as the `dataloader` outputs a batch in dictionary format that can be passed 
    straight into the model - `model(**batch)`.
 
  Arguments:
 
    dataloader (:obj:`torch.utils.data.dataloader.DataLoader`):
          Parsed data into batches of tensors.
 
    device_ (:obj:`torch.device`):
          Device used to load tensors before feeding to model.
 
  Returns:
     
    :obj:`List[List[int], List[int], float]`: List of [True Labels, Predicted
        Labels, Train Average Loss]
  Original author George Mihaila https://www.topbots.com/fine-tune-transformers-in-pytorch/
  """
 
  # Use global variable for model.
  #global model
 
  # Tracking variables
  predictions_labels = []
  true_labels = []
  predictions_max = []
  predictions_probs = []
  predictions_trueprobs = []
  #total loss for this epoch.
  total_loss = 0
 
  # Put the model in evaluation mode--the dropout layers behave differently
  # during evaluation.
  model.to(device_)

  model.eval()
 
  # Evaluate data for one epoch
  for batch in tqdm(dataloader, total=len(dataloader)):
 
    # add original labels
    true_labels += batch['labels'].numpy().flatten().tolist()
 
    # move batch to device
    batch = {k:v.type(torch.long).to(device_) for k,v in batch.items()}
 
    # Telling the model not to compute or store gradients, saving memory and
    # speeding up validation
    with torch.no_grad():        
 
        # Forward pass, calculate logit predictions.
        # This will return the logits rather than the loss because we have
        # not provided labels.
        # token_type_ids is the same as the "segment ids", which 
        # differentiates sentence 1 and 2 in 2-sentence tasks.
        # The documentation for this `model` function is here: 
        # https://huggingface.co/transformers/v2.2.0/model_doc/bert.html#transformers.BertForSequenceClassification
        outputs = model(**batch)
 
        # The call to `model` always returns a tuple, so we need to pull the 
        # loss value out of the tuple along with the logits. We will use logits
        # later to to calculate training accuracy.
        loss, logits = outputs[:2]
         
        # Move logits and labels to CPU
        logits = logits.detach().cpu().numpy()
 
        # Accumulate the training loss over all of the batches so that we can
        # calculate the average loss at the end. `loss` is a Tensor containing a
        # single value; the `.item()` function just returns the Python value 
        # from the tensor.
        total_loss += loss.item()
         
        # get predicitons to list
        predict_content = logits.argmax(axis=-1).flatten().tolist()
        # predict_score = logits.max(axis=-1).flatten().tolist()

        # get probabilities
        # Only works with batchsize 1
        probabilities_pairs = softmax(logits, axis=-1).tolist()
        probabilities = [max(x) for x in probabilities_pairs]
        true_probs = [x[true_index] for x in probabilities_pairs]
        #probabilities = [softmax(logits, axis=-1).flatten().tolist()]
        #probabilities = [max(softmax(logits, axis=-1)).flatten().tolist()]
         
 
        # update list
        predictions_labels += predict_content
        predictions_probs += probabilities
        predictions_trueprobs += true_probs 


  # Return all true labels and prediciton for future evaluations.
  return true_labels, predictions_labels, predictions_probs, predictions_trueprobs


def get_predictions(model_path, dataset_path, max_length, name, pos_label):
    #max_length = 1024
    #val_path = "/scratch/gpfs/cmcwhite/chloro_loc_model/chloro_labeledsetVal.csv"
    #n_labels = 2
    model_config = AutoConfig.from_pretrained(model_path)
    
    seq_tokenizer = BertTokenizerFast.from_pretrained(model_path)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = AutoModelForSequenceClassification.from_pretrained(model_path, config = model_config)
    #model.to(device)
    

    seqs, labels, ids = load_dataset(dataset_path, max_length)

    seqs_encodings = seq_tokenizer(seqs, is_split_into_words=True, return_offsets_mapping=True, truncation=True, padding=True)
    _ = seqs_encodings.pop("offset_mapping")
    
    unique_tags = set(labels)
    unique_tags  = sorted(list(unique_tags))  # make the order of the labels unchanged
    tag2id = {tag: id for id, tag in enumerate(unique_tags)}
    id2tag = {id: tag for tag, id in tag2id.items()}
    
    print(tag2id)
    print(id2tag)   
 
    labels_encodings = encode_tags(labels,  tag2id)
    dataset = SS3Dataset(seqs_encodings, labels_encodings)
    
    #valid_dataloader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False)
    batch_size = 10
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
   
    pos_index = tag2id[pos_label]
 
    true_labels, predictions_labels, probs, true_probs = validation(dataloader, model, device, pos_index)
    
    print(classification_report(true_labels, predictions_labels))
    
    text_true = [id2tag[x] for x in true_labels]
    text_pred = [id2tag[x] for x in predictions_labels]
    conf = confusion_matrix(text_true, text_pred)
    print(conf)
    outconf = model_path + "/output_confusion_" + name + ".csv"
    np.savetxt(outconf, conf, delimiter = ",")


    print(ids)
    print(text_true)
    print(text_pred)
    print(probs)
    print(true_probs)

    print(len(ids))
    print(len(text_true))
    print(len(text_pred))
    print(len(probs))
    print(len(true_probs))
     
 
    outdict = {"id": ids, "true_labels": text_true, "predicted_labels": text_pred, "prob" : probs, "true_probs" : true_probs}

    outdf = pd.DataFrame(outdict)
   
    outdf = outdf.sort_values(by=['true_probs'],  ascending=False)

    print(outdf)
 
    outdf_path = model_path + "/output_predictions_" + name + ".csv"
    outdf.to_csv(outdf_path)

    precision, recall, thresholds = precision_recall_curve(true_labels, true_probs)
    print(precision)
    print(recall)
    thresholds = np.concatenate(([0], thresholds))
    print(thresholds)
 

    prdict = {"precision" : precision, "recall" : recall, "threshold": thresholds}


    prdf= pd.DataFrame(prdict)
    print(prdf)
    prdf_path = model_path + "/output_prcurve_" + name + ".csv"
    prdf.to_csv(prdf_path)


def get_eval_args():

    parser = argparse.ArgumentParser()
    parser.add_argument("-m", "--model", dest = "model_path", type = str, required = True,
                        help="Model directory Ex. /path/to/model_dir")
    parser.add_argument("-l", "--labeledset", dest = "dataset_path", type = str, required = True,
                        help="Path to labeled set to evaluate, containing columns named Entry_name,sequence,label (csv)")
    parser.add_argument("-s", "--set", dest = "name", type = str, required = True,
                        help="Name of the set being evaluated, ex. test, train")

    parser.add_argument("-maxl", "--maxseqlength", dest = "max_length", type = int, required = False, default = 1024,
                        help="Truncate all sequences to this length (default 1024). Reduce if memory errors")
    parser.add_argument("-p", "--poslabel", dest = "pos_label", type = str, required = True,
                        help="The positive label (for pr curves) ex. Chloroplast")


    args = parser.parse_args()
    return(args)

if __name__ == "__main__":
    args = get_eval_args()
    get_predictions(args.model_path, args.dataset_path, args.max_length, args.name, args.pos_label)


