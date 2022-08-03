#!/usr/bin/env python3

import torch
from transformers import AutoTokenizer, AutoModel
import numpy as np

import random
from transformer_infrastructure.hf_utils import build_index_flat, build_index_voronoi
from transformer_infrastructure.run_tests import run_tests
from transformer_infrastructure.hf_embed import parse_fasta_for_embed, get_embeddings 
import copy
from Bio import SeqIO
from Bio.Align import MultipleSeqAlignment
from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord


from sklearn.preprocessing import normalize
import faiss

import pickle
import argparse

import os
import sys
import igraph
from pandas.core.common import flatten 
import pandas as pd 

from collections import Counter
import matplotlib.pyplot as plt
import logging

from sklearn.metrics.pairwise import cosine_similarity
from transformer_infrastructure.hf_utils import build_index_flat

#def get_index(np_embeddings, index = None):
#    print(np_embeddings)
#    if not index:
#        print("Create index")
#        d = np_embeddings.shape[1]
#        index = faiss.index_factory(d, "Flat", faiss.METRIC_INNER_PRODUCT)
#
#    faiss.normalize_L2(np_embeddings)
#    index.add(np_embeddings)
#    return(index)


def get_index_args():

    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--in", dest = "fasta_paths", nargs = "+", type = str, required = True,
                        help="Path to fastas (list)")
    
    parser.add_argument("-e", "--emb", dest = "embedding_path", type = str, required = False,
                        help="Path to embeddings")
    parser.add_argument("-ml", "--minlength", dest = "minlength", type = int, required = False,
                        help="Minimum length of sequences to add to the index")

    parser.add_argument("-b", "--base_outfile", dest = "base_outfile", type = str, required = True,
                        help="Path to outfile basename to store index(es) and id mapping, b.mean.faissindex, b.sigma.faissindex, b.faissindex.idmapping")
    parser.add_argument("-ss", "--strategy", dest = "strat", type = str, required = False, default = "meansig", choices = ['mean','meansig'],
                        help="Save index of mean, or two indexes, one mean, one sigma")

    parser.add_argument("-l", "--layers", dest = "layers", nargs="+", type = int,
                        help="Which layers (of 30 in protbert) to select")
    parser.add_argument("-hd", "--heads", dest = "heads", type = str,
                        help="File will one head identifier per line, format layer1_head3")

    parser.add_argument("-m", "--model", dest = "model_name",  type = str, required = True,
                        help="Model name or path to local model")

    parser.add_argument("-p", "--pca_plot", dest = "pca_plot",  action = "store_true", required = False, 
                        help="If flagged, output 2D pca plot of amino acid clusters")

    parser.add_argument("-l2", "--headnorm", dest = "headnorm",  action = "store_true", required = False, 
                        help="Take L2 normalization of each head")
    parser.add_argument("-ap", "--append", dest = "append",  action = "store_true", required = False, 
                        help="Append to existing index instead of starting a new one")
    parser.add_argument("-t", "--truncate", dest = "truncate",  type = int, required = False, default = 12000,
                        help="Default 12000. (23000 is too long)")
    parser.add_argument("-s", "--scoretype", dest = "scoretype",  type = str, required = False, default = "euclidean", choices = ["cosinesim", "euclidean"],
                        help="How to calculate initial sequence similarity score") 

    args = parser.parse_args()

    return(args)


def add_to_index(embedding_dict, mean_index, mean_outfile, sigma_index, sigma_outfile, strat = "meansig", scoretype = "euclidean"): 
    mean_embeddings = np.array(embedding_dict['sequence_embeddings']).astype(np.float32)
    if scoretype == "euclidean": 
        mean_index = build_index_flat(mean_embeddings, index = mean_index, scoretype = scoretype, normalize_l2 = False, return_norm = False)
    else:
        mean_index, norm = build_index_flat(mean_embeddings, index = mean_index, scoretype = scoretype, normalize_l2 = True, return_norm = True)

    faiss.write_index(mean_index, mean_outfile)
    if strat == "meansig":
        sigma_embeddings = np.array(embedding_dict['sequence_embeddings_sigma']).astype(np.float32) 
        sigma_index = build_index_flat(sigma_embeddings, index = sigma_index, scoretype = "euclidean", normalize_l2 = False, return_norm = False)
        faiss.write_index(sigma_index, sigma_outfile)
    return(mean_index, sigma_index)

if __name__ == '__main__':

    args = get_index_args()
    append = args.append
    fasta_paths = args.fasta_paths
    embedding_path = args.embedding_path
    base_outfile = args.base_outfile
    layers = args.layers
    heads = args.heads
    model_name = args.model_name
    pca_plot = args.pca_plot
    headnorm = args.headnorm
    truncate = args.truncate
    strat = args.strat
    minlength = args.minlength
    scoretype = args.scoretype
    # Keep to demonstrate effect of clustering or not
 
    faiss.omp_set_num_threads(10)
    if heads is not None:
       with open(heads, "r") as f:
         headnames = f.readlines()
         print(headnames)
         headnames = [x.replace("\n", "") for x in headnames]

         print(headnames)
    else:
       headnames = None
    logging.info("Check for torch")
    logging.info(torch.cuda.is_available())

    padding = 0 

    logging.info("model: {}".format(model_name))
    logging.info("fastas: {}".format(fasta_paths))
    logging.info("padding: {}".format(padding))
    mean_index = None 
    sigma_index = None 
    count = 0

    index_key_outfile = "{}.faissindex.idmapping".format(base_outfile) 
    mean_outfile =  "{}.mean.faissindex".format(base_outfile) 
    sigma_outfile =  "{}.sigma.faissindex".format(base_outfile) 

    if os.path.exists(index_key_outfile):
      if append == False:        
         os.remove(index_key_outfile)
         os.remove(mean_outfile)
         os.remove(sigma_outfile)

    with open(index_key_outfile, "a") as ok:
        if append == False:
             count = 0
        else:
             # continue with the last idx + 1
             with open(index_key_outfile, "r") as i:
                 count = int(i.readlines()[-1].split(",")[1]) + 1


        for fasta_path in fasta_paths: 
            if minlength: 
                seq_names, seqs, seqs_spaced = parse_fasta_for_embed(fasta_path, truncate = truncate, padding = padding, minlength=minlength)

            else:
                seq_names, seqs, seqs_spaced = parse_fasta_for_embed(fasta_path, truncate = truncate, padding = padding)
            print("Sequences loaded") 
            #avoid loading too many sequences into memory

            for i in range(0, len(seq_names), 1000):
                chunk_seq_names  = seq_names[i:i + 1000]
                chunk_seqs_spaced  = seqs_spaced[i:i + 1000]
                chunk_seqs = seqs[i:i + 1000]
                seqlens = [len(x) for x in chunk_seqs] 
                print(seqlens) 
                embedding_dict = get_embeddings(chunk_seqs_spaced,
                                        model_name,
                                        seqlens = seqlens,
                                        get_sequence_embeddings = True,
                                        get_aa_embeddings = False,
                                        layers = layers,  
                                        padding = padding,
                                        heads = headnames, 
                                        strat = strat)

                mean_index, sigma_index = add_to_index(embedding_dict, mean_index, mean_outfile, sigma_index, sigma_outfile, strat = strat, scoretype = scoretype) 
                # print("{} added to index".format(fasta_path))
                # Write mapping of sequence name o
                for seq_name in chunk_seq_names:
                     key_string =  "{},{}\n".format(seq_name, count)
                     ok.write(key_string)
                     count = count + 1 
    


