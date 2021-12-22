from typing import Union, List
from transformer_infrastructure.hf_embed import parse_fasta_for_embed
from transformers import BertTokenizerFast, AutoModel, AdamW 
from Bio import SeqIO
from Bio.Align import MultipleSeqAlignment
from Bio import AlignIO
from Bio.Seq import Seq
from torch.utils.data import Dataset, TensorDataset, DataLoader, RandomSampler, SequentialSampler
from sentence_transformers import LoggingHandler, SentenceTransformer, models, evaluation
import argparse
import numpy as np
import torch
import re
import random
import os
import pandas as pd
import copy
import json


 
def get_aasim_args():



    parser = argparse.ArgumentParser()
    parser.add_argument("-m", "--model", dest = "model_path", type = str, required = True,
                        help="Model directory path or name on huggingface. Ex. /path/to/model_dir Rostlab/prot_bert_bfd")
    parser.add_argument("-trfasta", "--trainfastas", dest = "train_fastas", type = str, required = True,
                        help="Path to file containing one fasta file per line. Each fasta corresponds to one gold standard alignment")

    parser.add_argument("-traln", "--trainalns", dest = "train_alns", type = str, required = True,
                        help="Path to files containine one gold standard alignment file in fasta format per line, with dashes for gaps")

    parser.add_argument("-dvfasta", "--devfastas", dest = "dev_fastas", type = str, required = True,
                        help="Path to file containing one fasta file per line. Each fasta corresponds to one gold standard alignment")

    parser.add_argument("-dvaln", "--devalns", dest = "dev_alns", type = str, required = True,
                        help="Path to files containine one gold standard alignment file in fasta format per line, with dashes for gaps")

    parser.add_argument("-tsfasta", "--testfastas", dest = "test_fastas", type = str, required = True,
                        help="Path to file containing one fasta file per line. Each fasta corresponds to one gold standard alignment")

    parser.add_argument("-tsaln", "--testalns", dest = "test_alns", type = str, required = True,
                        help="Path to files containine one gold standard alignment file in fasta format per line, with dashes for gaps")

    parser.add_argument("-o", "--outdir", dest = "outdir", type = str, required = True,
                        help="Output directory to save final model")


    #parser.add_argument("-d", "--dev", dest = "dev_path", type = str, required = True,
    #                    help="Path to dev/validation set (used during training), containing columns named sequence1,sequence2,id1,id2,label (set label colname with --label_col) (csv)")
    #parser.add_argument("-te", "--test", dest = "test_path", type = str, required = True,
    #                    help="Path to withheld test set (used after training), containing columns named sequence1,sequence2,id1,id2,label (set label colname with --label_col) (csv)")
    #parser.add_argument("-o", "--outdir", dest = "outdir", type = str, required = True,
    #                    help="Name of output directory")
    parser.add_argument("-maxl", "--maxseqlength", dest = "max_length", type = int, required = False, default = 1024,
                        help="Truncate all sequences to this length (default 1024). Reduce if memory errors")
    parser.add_argument("-n", "--expname", dest = "expname", type = str, required = False, default = "transformer_run",
                        help="Experiment name, used for logging, default = transformer_run")
    parser.add_argument("-e", "--epochs", dest = "epochs", type = int, required = False, default = 10,
                        help="Number of epochs. Increasing can help if memory error")
    parser.add_argument("-trbsize", "--train_batchsize", dest = "train_batchsize", type = int, required = False, default = 5,
                        help="Per device train batchsize. Reduce along with val batch size if memory error")
    parser.add_argument("-dvbsize", "--dev_batchsize", dest = "dev_batchsize", type = int, required = False, default = 5,
                        help="Per device validation batchsize. Reduce if memory error")

    args = parser.parse_args()
    return(args)

def load_dataset_alnpairs(fasta_list, aln_list, max_length):
    # Match fasta to reference alignment with dictionaries
    seqdict = {}
    with open(fasta_list, "r") as f:
        for idx, fasta_file in enumerate(f):
            
         record_dict = {}
         print(fasta_file)
         protgroup = fasta_file.split("/")[-1].split(".")[0]
         seq_names, seqs, seqs_spaced = parse_fasta_for_embed(fasta_file.replace("\n", ""), extra_padding = False)
         # Only first three sequences are in the gold standard
         for i in [0,1,2]:
             record_dict[seq_names[i]] = seqs_spaced[i]
         seqdict[protgroup] = record_dict 

    alndict = {}
    with open(aln_list, "r") as f:
       for idx, aln_file in enumerate(f):
 

           protgroup = aln_file.split("/")[-1].split(".")[0]

           aln_record_dict = {}
           with open(aln_file.replace("\n", ""), "r") as input_handle:
             alignment= AlignIO.read(input_handle, format = "fasta")
             for i in [0,1,2]:
                #poslist = []
                #for j in range(len(alignment[i])): 
                #        if alignment[i][j] != "-":
                #              poslist.append(j)
      
                aln_record_dict[alignment[i].id] = str(alignment[i].seq)
             alndict[protgroup] = aln_record_dict    



    allprots = seqdict.keys()
    #trainset = []
  
    seqs1 = []
    seqs2 =[]
    pos1 =[]
    pos2 =[]
    seqnames1 =[]
    seqnames2 =[]
    labels = []
 
   
     
    for protgroup in allprots:
        prot_seqs =seqdict[protgroup]
        prot_alns =alndict[protgroup]
        allseqnames = prot_seqs.keys()
        complete = []
        for seqname1 in allseqnames:
              
           complete.append(seqname1)
           for seqname2 in allseqnames:
               if seqname2 in complete:
                    continue                
               seq1 = prot_seqs[seqname1]
               seq2 = prot_seqs[seqname2]
               aln1 = prot_alns[seqname1]
               aln2 = prot_alns[seqname2]
               #print(seqname1, seqname2)
               #print(seq1, seq2)
               #print(aln1, aln2) 
               # Both alignments are the same length
               seqpos1 = 0
               seqpos2 = 0
               equi_positions = []
               for i in range(len(aln1)):
                   char1 = aln1[i]
                   char2 = aln2[i]
                   #print(char1, char2)
                   if char1 != "-" and char2 != "-":
                      #print(char1, char2, seqpos1, seqpos2)
                      # Don't take positions beyond the max length 
                      if seqpos1 <= max_length-2:
                          if seqpos2 <= max_length - 2:
                              equi_positions.append([seqpos1, seqpos2])

                   if char1 != "-":
                      seqpos1 = seqpos1 + 1
                   if char2 != "-":
                      seqpos2 = seqpos2 + 1
              

               seq1_fixed = "".join(seq1.split())
               seq1_fixed = re.sub(r"[UZOB]", "X", seq1_fixed)
               seq1_fixed = list(seq1_fixed)[:max_length-2]             

               seq2_fixed = "".join(seq2.split())
               seq2_fixed = re.sub(r"[UZOB]", "X", seq2_fixed)
               seq2_fixed = list(seq2_fixed)[:max_length-2]             


               print(seqname1, seqname2)
               for equi  in equi_positions:
  
                   print(equi) 
                   seqs1.append(seq1_fixed)
                   seqs2.append(seq2_fixed)
                   pos1.append(equi[0])
                   pos2.append(equi[1])
                   seqnames1.append(seqname1)
                   seqnames2.append(seqname2)
                   labels.append(1)

               # Make some random pair negatives
               for i in range(len(equi_positions)):
                   non_equi  = [random.choice(range(len(seq1_fixed))), random.choice(range(len(seq2_fixed)))]
                   if non_equi in equi_positions:
                       continue
                   seqs1.append(seq1_fixed)
                   seqs2.append(seq2_fixed)
                   pos1.append(non_equi[0])
                   pos2.append(non_equi[1])
                   seqnames1.append(seqname1)
                   seqnames2.append(seqname2)
                   labels.append(0)

               # Make close negatives, one after correct second position
               for i in range(len(equi_positions)):
                   if non_equi[1] < len(seq2_fixed):
                       seqs1.append(seq1_fixed)
                       seqs2.append(seq2_fixed)
                       pos1.append(non_equi[0])
                       pos2.append(non_equi[1] + 1)
                       seqnames1.append(seqname1)
                       seqnames2.append(seqname2)
                       labels.append(0)

                     
                   


                     #trainset.append([seq1, seq2, equi[0], equi[1], seqname1, seqname2])
    return(seqs1, seqs2, pos1, pos2, labels, seqnames1, seqnames2 )     


def encode_tags(labels, labels1, encodings1, labels2, encodings2, max_length):
    
    input_ids = []
    token_type_ids = [] 
    attention_masks = []
    enc_labels1 = [] 
    enc_labels2 = []   

    for label1, input_id1, label2, input_id2, in zip(labels1, encodings1.input_ids, labels2, encodings2.input_ids): #encodings.offset_mapping):


        input_id = np.concatenate((input_id1, input_id2), axis = 0)

        # The length of the first sequence
        ones1 =  np.ones(len(input_id1),dtype=int)
        zeroes1 = ones1 * 0

        # The length of the second sequence
        ones2 = np.ones(len(input_id2), dtype = int) 
        zeroes2 = ones2 * 0 

        # The remaining pad to max_length
        ones_tail = np.ones(max_length - len(ones1) - len(ones2), dtype = int) 
        zeroes_tail = ones_tail * 0       

        # label1 = 0
        # [0, 1, 0, 0]
        enc_label1 = zeroes1.copy()
        enc_label1[label1 + 1] = 1

        # label1 = 1
        # [0, 0, 1, 0]
        enc_label2 = zeroes2.copy()
        enc_label2[label2 + 1] = 1
        
        # tokens1
        # [1, 4, 3, 5]
      
        # tokens2
        # [1, 3, 7, 5]
     
        # Token_ids
        # [1, 4, 3, 5, 1, 3, 7, 5, 0, 0]
        token_id = np.concatenate((input_id, zeroes_tail), axis = 0)

        # enc_labels_1
        # [0, 1, 0, 0, 0, 0, 0, 0, 0, 0]
        enc_label1 = np.concatenate((enc_label1, zeroes2, zeroes_tail), axis = 0)

        # enc_labels2
        # [0, 0, 0, 0, 0, 1, 0, 0, 0, 0]
        enc_label2 = np.concatenate((zeroes1, enc_label2, zeroes_tail), axis = 0)
    
        # Mark positions of diff sequences
        # [0, 0, 0, 1, 1, 1, 1, 1, 1, 1]
        token_type_id = np.concatenate((zeroes1, ones2, ones_tail), axis = 0)

        # Mark positions that are sequence vs. padding
        # [1, 1, 1, 1, 1, 1, 1, 1, 0, 0]
        attention_mask = np.concatenate((ones1, ones2, zeroes_tail), axis = 0)

        input_ids.append(token_id)
        enc_labels1.append([enc_label1])
        enc_labels2.append([enc_label2])
        token_type_ids.append(token_type_id)
        attention_masks.append(attention_mask)

    indexes = range(0, len(input_ids))
    assert (len(indexes) == len(input_ids) == len(token_type_ids) == len(attention_masks) == len(labels) == len(enc_labels1) == len(enc_labels2)), "Unequal lengths!"
    data_dict = {"index": indexes, "input_ids": input_ids, "token_type_ids": token_type_ids, "attention_masks":attention_masks, "labels": labels, "enc_labels1": enc_labels1, "enc_labels2": enc_labels2}

    return(data_dict )

# Copied from cs github
# Function to calculate the accuracy of our predictions vs labels
def flat_accuracy(preds, labels, return_predict_correctness = False):
  pred_flat = np.argmax(preds, axis=1).flatten()
  labels_flat = labels.flatten()
  if return_predict_correctness:
    return np.sum(pred_flat == labels_flat) / len(labels_flat), pred_flat == labels_flat
  else:
    return np.sum(pred_flat == labels_flat) / len(labels_flat)

def flat_predictions(preds):
  pred_flat = np.argmax(preds, axis=1).flatten()
  return pred_flat == 1

def load_datasets(train_fastas, train_alns, dev_fastas, dev_alns, test_fastas, test_alns, train_batchsize, dev_batchsize):


    train_seqs1, train_seqs2, train_pos1, train_pos2, train_labels, train_seqnames1, train_seqnames2 = load_dataset_alnpairs(train_fasta_list, train_aln_list, int(max_length/2))




    seq_tokenizer = BertTokenizerFast.from_pretrained(model_name, do_lower_case=False)

    # Going to be concatenating sequence pair
    # With one mask showing sequence vs. padding: attention_mask
    # With one mask showing which sequence is which: token_type_ids
    train_seqs1_encodings = seq_tokenizer(train_seqs1, is_split_into_words=True, return_offsets_mapping=False, truncation=True, padding=False, max_length = int(max_length/2))
    train_seqs2_encodings = seq_tokenizer(train_seqs2, is_split_into_words=True, return_offsets_mapping=False, truncation=True, padding=False, max_length = int(max_length/2))



    #train_seqs_encodings, train_pos1_encodings, train_pos2_encodings = 
    train_data_dict = encode_tags(train_labels, train_pos1, train_seqs1_encodings, train_pos2, train_seqs2_encodings, max_length)

    outtrain = pd.DataFrame.from_dict(train_data_dict)
    outtrain['seqs1'] = train_seqs1
    outtrain['seqs2'] = train_seqs2
    outtrain['pos1'] = train_pos1
    outtrain['pos2'] = train_pos2
    outtrain.to_csv(os.path.join(outdir, "traindata.csv"), index=False)



    np.set_printoptions(threshold=np.inf)
    print("input ids1")
    print(train_seqs1_encodings.input_ids[0])

    print("input ids2")
    print(train_seqs2_encodings.input_ids[0])

    print("encoded input")
    print(train_data_dict['input_ids'][0])

    print("equivalent positions")
    print(train_pos1[0], train_pos2[0])

    print("sequence groups")
    print(train_data_dict['token_type_ids'][0])

    print("sequence positions")
    print(train_data_dict['attention_masks'][0])

    print("position1 label")
    print(train_data_dict['enc_labels1'][0])

    print("position2 label")
    print(train_data_dict['enc_labels2'][0])

    print("label")
    print(train_data_dict['labels'][0])

    print("index")
    print(train_data_dict['index'][0])

 
    #https://github.com/llightts/CSI5138_Project/blob/master/RoBERTa_WiC_baseline.ipynb

  #      print(enc_labels1)
#        enc_labels1 = enc_labels1.type(torch.LongTensor)
   #     print(enc_labels1)

    #    enc_labels2 = enc_labels2.type(torch.LongTensor)
    print("lengths")
    print(len(train_data_dict["input_ids"]))
    print(len(train_data_dict["token_type_ids"]))
    print(len(train_data_dict["attention_masks"]))
    print(len(train_data_dict["labels"]))
    print(len(train_data_dict["enc_labels1"]))
    print(len(train_data_dict["enc_labels2"]))
    print(len(train_data_dict["index"]))

    train_data = TensorDataset(
      torch.tensor(train_data_dict["input_ids"]),
      torch.tensor(train_data_dict["token_type_ids"]),
      torch.tensor(train_data_dict["attention_masks"]),
      torch.tensor(train_data_dict["labels"]),
      torch.tensor(train_data_dict["enc_labels1"]).type(torch.FloatTensor),
      torch.tensor(train_data_dict["enc_labels2"]).type(torch.FloatTensor),
      torch.tensor(train_data_dict["index"])
    )


    
    # Create a sampler and loader
    train_sampler = RandomSampler(train_data)
    train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=train_batchsize)



    # Dev set
    dev_seqs1, dev_seqs2, dev_pos1, dev_pos2, dev_labels, dev_seqnames1, dev_seqnames2 = load_dataset_alnpairs(dev_fasta_list, dev_aln_list, int(max_length/2))

    dev_seqs1_encodings = seq_tokenizer(dev_seqs1, is_split_into_words=True, return_offsets_mapping=False, truncation=True, padding=False, max_length = int(max_length/2))
    dev_seqs2_encodings = seq_tokenizer(dev_seqs2, is_split_into_words=True, return_offsets_mapping=False, truncation=True, padding=False, max_length = int(max_length/2))
    dev_data_dict = encode_tags(dev_labels, dev_pos1, dev_seqs1_encodings, dev_pos2, dev_seqs2_encodings, max_length)


    outdev = pd.DataFrame.from_dict(dev_data_dict)
    outdev['seqs1'] = dev_seqs1
    outdev['seqs2'] = dev_seqs2
    outdev['pos1'] = dev_pos1
    outdev['pos2'] = dev_pos2
    outdev.to_csv(os.path.join(outdir, "devdata.csv"), index=False)





    dev_data = TensorDataset(
      torch.tensor(dev_data_dict["input_ids"]),
      torch.tensor(dev_data_dict["token_type_ids"]),
      torch.tensor(dev_data_dict["attention_masks"]),
      torch.tensor(dev_data_dict["labels"]),
      torch.tensor(dev_data_dict["enc_labels1"]).type(torch.FloatTensor),
      torch.tensor(dev_data_dict["enc_labels2"]).type(torch.FloatTensor),

      torch.tensor(dev_data_dict["index"])
    )

    # Create a sampler and loader
    dev_sampler = SequentialSampler(dev_data)
    dev_dataloader = DataLoader(dev_data, sampler=dev_sampler, batch_size=dev_batchsize)

    # Test, unlabelled
    test_seqs1, test_seqs2, test_pos1, test_pos2, test_labels, test_seqnames1, test_seqnames2 = load_dataset_alnpairs(test_fasta_list, test_aln_list, int(max_length/2))
    test_seqs1_encodings = seq_tokenizer(test_seqs1, is_split_into_words=True, return_offsets_mapping=False, truncation=True, padding=False, max_length = int(max_length/2))
    test_seqs2_encodings = seq_tokenizer(test_seqs2, is_split_into_words=True, return_offsets_mapping=False, truncation=True, padding=False, max_length = int(max_length/2))
    test_data_dict = encode_tags(test_labels, test_pos1, test_seqs1_encodings, test_pos2, test_seqs2_encodings, max_length)


    outtest = pd.DataFrame.from_dict(test_data_dict)
    outtest['seqs1'] = test_seqs1
    outtest['seqs2'] = test_seqs2
    outtest['pos1'] = test_pos1
    outtest['pos2'] = test_pos2
    outtest.to_csv(os.path.join(outdir, "testdata.csv"), index=False)



    test_data = TensorDataset(
      torch.tensor(test_data_dict["input_ids"]),
      torch.tensor(test_data_dict["token_type_ids"]),
      torch.tensor(test_data_dict["attention_masks"]),
      torch.tensor(test_data_dict["enc_labels1"]).type(torch.LongTensor),
      torch.tensor(test_data_dict["enc_labels2"]).type(torch.LongTensor),
      torch.tensor(test_data_dict["index"])
    )

    # Create a sampler and loader
    test_sampler = RandomSampler(test_data)
    test_dataloader = DataLoader(test_data, sampler=test_sampler, batch_size=dev_batchsize)

    return(train_dataloader, dev_dataloader, test_dataloader)


class WiC_Head(torch.nn.Module):
    def __init__(self, model, embedding_size = 1024):
        """
        Keeps a reference to the provided RoBERTa model. 
        It then adds a linear layer that takes the distance between two 
        """
        super(WiC_Head, self).__init__()

        self.bert = BertModel(config)

        self.embedding_size = embedding_size
        self.embedder = model
        self.linear_diff = torch.nn.Linear(embedding_size, 250, bias = True)
        self.linear_separator = torch.nn.Linear(250, 2, bias = True)
        self.loss = torch.nn.CrossEntropyLoss()
        self.activation = torch.nn.ReLU()
        self.softmax = torch.nn.Softmax()

    def forward(self, input_ids=None, attention_mask=None, labels=None,
                enc_labels1 = None, enc_labels2 = None):
        """
        Takes in the same argument as RoBERTa forward plus two tensors for the location of the 2 words to compare
        """
        if enc_labels1 is None or enc_labels2 is None:
          raise ValueError("The tensors (enc_labels1, enc_labels2) containing the location of the words to compare in the input vector must be provided.")
        elif input_ids is None:
          raise ValueError("The input_ids tensor must be provided.")
        elif enc_labels1.shape[0] != input_ids.shape[0] or enc_labels2.shape[0] != input_ids.shape[0]:
          raise ValueError("All provided vectors should have the same batch size.")
        batch_size = enc_labels1.shape[0]
        # Get the embeddings (?)
        #print(batch_size)
        outputs = self.embedder(input_ids=input_ids, attention_mask=attention_mask)
        # Get the words
        #print(outputs)
        # Is a BaseModelOutputWithPoolingAndCrossAttentions     
        embs = outputs[0] # Get last hidden state
        #embs, _=  self.embedder(input_ids=input_ids, attention_mask=attention_mask).embeddings()


        word1s = torch.matmul(enc_labels1, embs).view(batch_size, self.embedding_size)
        
        #.view(batch_size, self.embedding_size)
        word2s = torch.matmul(enc_labels2, embs).view(batch_size, self.embedding_size)
  
        diff = word1s - word2s
        # Calculate outputs using activation
        layer1_results = self.activation(self.linear_diff(diff))
        logits = self.softmax(self.linear_separator(layer1_results))
        outputs = logits
        # Calculate the loss
        if labels is not None:
            #  We want seperation like a SVM so use Hinge loss
            loss = self.loss(logits.view(-1, 2), labels.view(-1))
            outputs = (loss, logits)
        return outputs



if __name__ == "__main__":



    args = get_aasim_args()

    model_name = args.model_path
    max_length = args.max_length
    train_fasta_list = args.train_fastas
    train_aln_list = args.train_alns

    dev_fasta_list = args.dev_fastas
    dev_aln_list = args.dev_alns

    test_fasta_list = args.test_fastas
    test_aln_list = args.test_alns
    outdir = args.outdir
   
    train_batchsize = args.train_batchsize
    dev_batchsize = args.dev_batchsize
    epochs = args.epochs

    if not os.path.exists(outdir):
         os.makedirs(outdir) 


    #train_batchsize = 10
    #epochs = 20
    patience = 10
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    n_gpu = torch.cuda.device_count()
    torch.cuda.get_device_name(0)

    train_dataloader, dev_dataloader, test_dataloader = load_datasets(train_fasta_list, train_aln_list, dev_fasta_list, dev_aln_list, test_fasta_list, test_aln_list, train_batchsize, dev_batchsize)

    model = AutoModel.from_pretrained(model_name)

    class_model = WiC_Head(model, embedding_size = 1024)

    print(class_model)


    # Variable for minimal accuracy
    
    #MIN_ACCURACY = 0.99 # Based on the average accuracy
    #REACHED_MIN_ACCURACY = False
    best_weights = class_model.state_dict()
    # Want to maximize accuracy
    max_val_acc = (0, 0)
    # Put the model in GPU
    class_model.cuda()
    # Create the optimizer
    param_optimizer = list(class_model.named_parameters())
    no_decay = ['bias', 'gamma', 'beta']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)],
         'weight_decay_rate': 0.01},
        {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)],
         'weight_decay_rate': 0.0}
    ]
    # I use the one that comes with the models, but any other optimizer could be used
    optimizer = AdamW(optimizer_grouped_parameters, lr=1e-5)
    # Store our loss and accuracy for plotting
    fit_history = {"loss": [],  "accuracy": [], "val_loss": [], "val_accuracy": []}
    epoch_number = 0
    epoch_since_max = 0
    continue_learning = True
    while epoch_number < epochs and continue_learning:
      epoch_number += 1
      print(f"Training epoch #{epoch_number}")
      # Tracking variables
      tr_loss, tr_accuracy = 0, 0
      nb_tr_examples, nb_tr_steps = 0, 0
      eval_loss, eval_accuracy = 0, 0
      nb_eval_steps, nb_eval_examples = 0, 0
      # Training
      # Set our model to training mode (as opposed to evaluation mode)
      class_model.train()
      # Freeze RoBERTa weights
      #class_model.embedder.eval()
      # This froze the BERT weights
      #class_model.embedder.requires_grad_ = False
      # Train the data for one epoch
      for step, batch in enumerate(train_dataloader):
        # Add batch to GPU
        batch = tuple(t.cuda() for t in batch)
        # Unpack the inputs from our dataloader
        # CHECK THIS ORDER


        b_input_ids, b_token_ids, b_input_mask, b_labels, b_word1, b_word2, b_index = batch
        # Clear out the gradients (by default they accumulate)
        optimizer.zero_grad()
        # Forward pass
        #loss, logits = class_model(b_input_ids, token_type_ids=b_token_ids, attention_mask=b_input_mask, labels=b_labels)   
        loss, logits = class_model(b_input_ids, attention_mask=b_input_mask, 
                                   labels=b_labels, enc_labels1 = b_word1, enc_labels2 = b_word2) 
        # Backward pass
        loss.backward()
        # Update parameters and take a step using the computed gradient
        optimizer.step()
        # Move logits and labels to CPU
        logits = logits.detach().cpu().numpy()
        label_ids = b_labels.cpu().numpy()
        # Calculate the accuracy
        b_accuracy = flat_accuracy(logits, label_ids) # For RobertaForClassification
        # Append to fit history
        fit_history["loss"].append(loss.item()) 
        fit_history["accuracy"].append(b_accuracy) 
        # Update tracking variables
        tr_loss += loss.item()
        tr_accuracy += b_accuracy
        nb_tr_examples += b_input_ids.size(0)
        nb_tr_steps += 1
        if nb_tr_steps%10 == 0:
          print("\t\tTraining Batch {}: Loss: {}; Accuracy: {}".format(nb_tr_steps, loss.item(), b_accuracy))
      print("Training:\n\tLoss: {}; Accuracy: {}".format(tr_loss/nb_tr_steps, tr_accuracy/nb_tr_steps))
      # Validation
      # Put model in evaluation mode to evaluate loss on the validation set
      class_model.eval()
      # Evaluate data for one epoch
      for batch in dev_dataloader:
        # Add batch to GPU
        batch = tuple(t.cuda() for t in batch)
        # Unpack the inputs from our dataloader
  
        b_input_ids, b_token_ids, b_input_mask, b_labels, b_word1, b_word2, b_index = batch
        # Telling the model not to compute or store gradients, saving memory and speeding up validation
        with torch.no_grad():
          # Forward pass, calculate logit predictions
          #loss, logits = class_model(b_input_ids, token_type_ids=b_token_ids, attention_mask=b_input_mask, labels=b_labels)
          loss, logits = class_model(b_input_ids, attention_mask=b_input_mask, 
                                     labels=b_labels, enc_labels1 = b_word1, enc_labels2 = b_word2)
        # Move logits and labels to CPU
        logits = logits.detach().cpu().numpy()
        label_ids = b_labels.cpu().numpy()
        # Calculate the accuracy
        b_accuracy = flat_accuracy(logits, label_ids) # For RobertaForClassification
        # Append to fit history
        fit_history["val_loss"].append(loss.item()) 
        fit_history["val_accuracy"].append(b_accuracy) 
        # Update tracking variables
        eval_loss += loss.item()
        eval_accuracy += b_accuracy
        nb_eval_examples += b_input_ids.size(0)
        nb_eval_steps += 1
        if nb_eval_steps%10 == 0:
          print("\t\tValidation Batch {}: Loss: {}; Accuracy: {}".format(nb_eval_steps, loss.item(), b_accuracy))
      eval_acc = eval_accuracy/nb_eval_steps
      if epoch_number == 1:   
          torch.save(class_model.state_dict(), os.path.join(outdir,'ProtWiCHead_epoch1.pt'))

      if eval_acc >= max_val_acc[0]:
        max_val_acc = (eval_acc, epoch_number)
        continue_learning = True
        epoch_since_max = 0 # New max
        best_weights = copy.deepcopy(class_model.state_dict()) # Keep the best weights
        # See if we have reached min_accuracy
        #if eval_acc >= MIN_ACCURACY:
        #  REACHED_MIN_ACCURACY = True
        # Save to file only if it has reached min acc


        #if REACHED_MIN_ACCURACY:
          # Save the best weights to file
        #  torch.save(best_weights, os.path.join(outdir,'ProtWiCHead.pt'))
        #  continue_learning = False # Stop learning. Reached baseline acc for this model
      else:
        epoch_since_max += 1
        if epoch_since_max > patience:
          continue_learning = False # Stop learning, starting to overfit
      print("Validation:\n\tLoss={}; Accuracy: {}".format(eval_loss/nb_eval_steps, eval_accuracy/nb_eval_steps))
    print(f"Best accuracy ({max_val_acc[0]}) obtained at epoch #{max_val_acc[1]}.")
    # Reload the best weights (from memory)
    class_model.load_state_dict(best_weights)

    with open("fit_history.json", 'w') as json_file:
      json.dump(fit_history, json_file)

    #model_to_save = model.module if hasattr(model, 'module') else model
    torch.save(best_weights, os.path.join(outdir, 'ProtWiCHead.pt'))
    #output_model_file = os.path.join(outdir, WEIGHTS_NAME)
    #output_config_file = os.path.join(outdir, CONFIG_NAME)
    #model_to_save.config.to_json_file(output_config_file)
    #tokenizer.save_vocabulary(outdir)   
    class_model.save_pretrained(outdir)
    seq_tokenizer.save_pretrained(outdir)

   
