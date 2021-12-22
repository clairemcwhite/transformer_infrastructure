#!/usr/bin/env python3



from transformer_infrastructure.aa_simtrain2 import load_dataset_alnpairs
from transformer_infrastructure.hf_aligner2 import get_seq_groups, AA, get_besthits
from transformers import AutoTokenizer, AutoModel
import numpy as np

from transformer_infrastructure.hf_utils import build_index
from transformer_infrastructure.hf_embed import parse_fasta_for_embed, get_embeddings 

from Bio import SeqIO
#from Bio.Seq import Seq
from Bio.Align import MultipleSeqAlignment
from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord


import faiss
#import unittest
#from sentence_transformers import util
#from iteration_utilities import  duplicates

import pickle
import argparse

import os
import sys
import igraph
from pandas.core.common import flatten 
import pandas as pd 

from collections import Counter


import logging

def get_looser_scores(aa, index, hidden_states):
     '''Get all scores with a particular amino acid''' 
     hidden_state_aa = np.take(hidden_states, [aa.index], axis = 0)
     # Search the total number of amino acids
     # Cost of returning higher n is minimal
     n_aa = hidden_states.shape[0]
     D_aa, I_aa =  index.search(hidden_state_aa, k = n_aa)
     #print("looser scores")
     #print(aa)
     #print(D_aa.tolist())
     #print(I_aa.tolist())
     return(list(zip(D_aa.tolist()[0], I_aa.tolist()[0])))


      
def get_particular_score(D, I, aa1, aa2):
        ''' Use with squish, replace with get_looser_scores '''

        #print(aa1, aa2)
        #seqnum different_from index
        print(D.shape)
        print(aa1.index)
        print(aa2.index)
        scores = D[aa1.index][aa1.seqpos][aa2.index]
        #print(scores)
        ids = I[aa1.index][aa1.seqpos][aa2.index]
        #print(ids)
        for i in range(len(ids)):
           #print(aa1, score_aa, scores[i])
           if ids[i] == aa2:
              #print(aa1, aa2, ids[i], scores[i])
              return(scores[i])
        else:
           return(0) 
 
def reshape_flat(hstates_list):

    # Go from (numseqs, seqlen, emb) to (numseqs * seqlen, emb)
    hidden_states = np.reshape(hstates_list, (hstates_list.shape[0]*hstates_list.shape[1], hstates_list.shape[2]))
    return(hidden_states)

def split_distances_to_sequence(D, I, seqnums, index_to_aa, numseqs, padded_seqlen):
   I_tmp = []
   D_tmp = []
   print(D.shape)
   print(I.shape)
   # For each amino acid...
   for i in range(len(I)):
      #print(i)
      # Make empty list of lists, one per sequence
      I_query =  [[] for i in range(numseqs)]
      D_query = [[] for i in range(numseqs)]
     
      for j in range(len(I[i])):
           try:
              aa = index_to_aa[I[i][j]]

              seqnum = aa.seqnum
              seqnum_index = seqnums.index(seqnum)
              I_query[seqnum_index].append(aa) 
              D_query[seqnum_index].append(D[i][j])
           except Exception as E:
               continue
      #print(len(I_query[i]), len(D_query[i]))
      #if len(I_query[i]) != len(D_query[i]):
      #      print("ISSUE")

      I_tmp.append(I_query)
      D_tmp.append(D_query)
   print(padded_seqlen)
   D =  [D_tmp[i:i + padded_seqlen] for i in range(0, len(D_tmp), padded_seqlen)]
   I =  [I_tmp[i:i + padded_seqlen] for i in range(0, len(I_tmp), padded_seqlen)]

  
   return(D, I)


def get_similarity_network(seqs, seq_names, seqnums, hstates_list, padding = 5):
    """
    Control for running whole alignment process
    Last four layers [-4, -3, -2, -1] is a good choice for layers
    seqs should be spaced
    padding tells amount of padding to remove from seqs
    model = prot_bert_bfd
    """
    padded_seqlen = hstates_list.shape[1]
    
    numseqs = len(seqs)

    
   # Drop X's from here
    #print(hstates_list.shape)
    # Remove first and last X padding

    # After encoding, remove spaces from sequences
    seqlens = [len(x) for x in seqs]
    #for seq in seqs:
    #   hidden_states = get_hidden_states([seq], model, tokenizer, layers)
    #   hidden_states_list.append(hidden_states)


    # Build index from all amino acids 
    #d = hidden_states[0].shape[1]

    # Go from (numseqs, seqlen, emb) to (numseqs * seqlen, emb)

    logging.info("Flattening hidden states list")
    hidden_states = np.array(reshape_flat(hstates_list))  
    logging.info("embedding_shape: {}".format(hidden_states.shape))


 
    logging.info("Convert index position to amino acid position")
    #index_to_aa = {}

    #for i in raggnge(len(seqs)):
    #    for j in range(padded_seqlen):
    #       if j >= seqlens[i]:
    #         continue 
    #       aa = "s{}-{}-{}".format(i, j, seqs[i][j])    
    #       
    #       index_to_aa[i * padded_seqlen + j] = aa
    #print(index_to_aa)

    # Write sequences with aa ids
    seqs_aas = []
  

    for i in range(len(seqs)):
        #print(seqs[i])
        
        seq_aas = []
        seqnum = seqnums[i]
        for j in range(len(seqs[i])):
           aa = AA()
           aa.seqnum = seqnum
           aa.seqpos = j
           aa.seqaa =  seqs[i][j]

           seq_aas.append(aa)
        seqs_aas.append(seq_aas)
    
   # print(seqs_aas)
    # Can this be combined with previous?
    #print(seqs_aas)

    index_to_aa = {}
    for i in range(len(seqs_aas)):
        for j in range(padded_seqlen):
           if j >= seqlens[i]:
             continue 
           aa = seqs_aas[i][j]
           index_num = i * padded_seqlen + j
           aa.index = index_num           
           index_to_aa[index_num] = aa
    #print(index_to_aa) 
    logging.info("Build index") 
    print("Build index")
   
    index = build_index(hidden_states)
    logging.info("Search index") 
    print("search index")
    D1, I1 =  index.search(hidden_states, k = numseqs*10) 

    logging.info("Split results into proteins") 
    print("Split results into proteins") 
    # Still annoyingly slow
    D2, I2 = split_distances_to_sequence(D1, I1, seqnums, index_to_aa, numseqs, padded_seqlen) 
    #print(I2)
    #logging.info("get best hitlist")
    print("get best hitlist")

 
    hitlist_all = get_besthits(D2, I2, seqnums, index_to_aa, padded_seqlen, minscore = 0)
    for x in hitlist_all:
         print(x)

# Make parameter actually control this
def format_sequences(fasta, padding =  5):
   
    # What are the arguments to this? what is test.fasta? 
    seq_names, seqs, seqs_spaced = parse_fasta_for_embed(fasta, extra_padding = True)
    
    return(seq_names, seqs, seqs_spaced)


def get_align_args():

    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--in", dest = "fasta_path", type = str, required = True,
                        help="Path to fasta")
    
    parser.add_argument("-a", "--aln", dest = "alignment", type = str, required = True,
                        help="Path to reference alignment")
 

    parser.add_argument("-o", "--outfile", dest = "out_path", type = str, required = True,
                        help="Path to outfile")



    parser.add_argument("-l", "--layers", dest = "layers", nargs="+", type=int, default = [-4,-3,-2,-1],
                        help="Additionally exclude outlier sequences from final alignment")
    parser.add_argument("-m", "--model", dest = "model_name",  type=str, required = True,
                        help="Model name or path to local model")




     

    args = parser.parse_args()

    return(args)


 
if __name__ == '__main__':

    args = get_align_args()

    fasta_path = args.fasta_path
    outfile = args.out_path
    layers = args.layers
    alignment = args.alignment
    model_name = args.model_name
 
    padding = 5 
    minscore1 = 0.5

    logging.info("model: {}".format(model_name))
    logging.info("fasta: {}".format(fasta_path))
    logging.info("padding: {}".format(padding))
    
   #fasta = '/scratch/gpfs/cmcwhite/quantest2/QuanTest2/Test/ung.vie'


    seq_names, seqs, seqs_spaced= format_sequences(fasta_path, padding = padding)#, truncate = [0,20])

    protnames, seqs1, seqs2, pos1, pos2, labels, seqnames1, seqnames2 = load_dataset_alnpairs(fasta_path, alignment, max_length = 256, max_records = 3) 

    
    print(seqs1)
    seqs = seqs1 + seqs2
    seqs_spaced = [" ".join(x) for x in seqs]
    seqs_spaced = list(dict.fromkeys(seqs_spaced))
    seqs = [x.replace(" ", "") for x in seqs_spaced]
    seqlens = [len(x) for x in seqs]
    seq_names = seqnames1 + seqnames2
    seq_names = list(dict.fromkeys(seq_names))
    seq_idx = {}
    for idx, seqname in enumerate(seq_names):
         seq_idx[seqname] = idx
    print(seq_idx)
    print(seqs_spaced)
    print(seqlens)
    embedding_dict = get_embeddings(seqs_spaced,
                                    model_name,
                                    seqlens = seqlens,
                                    get_sequence_embeddings = True,
                                    get_aa_embeddings = True,
                                    layers = layers,
                                    padding = padding)

    print(padding) 
    cluster_seqnums_list, cluster_seqs_list,  cluster_names_list, cluster_hstates_list, to_exclude = get_seq_groups(seqs_spaced ,seq_names, embedding_dict, logging, padding = False, exclude = False, do_clustering = False, )

    
    print(seqs)
    print(seqs_spaced)
    print(seq_idx.values())
    print(embedding_dict['aa_embeddings'])
    print(padding)
   

    index, hidden_states, index_to_aa = get_similarity_network(seqs,  seq_names, list(seq_idx.values()), embedding_dict['aa_embeddings'], padding = padding)
