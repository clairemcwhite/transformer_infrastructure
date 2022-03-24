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


def get_index(embedding_dict, index = None):
    np_embeddings = np.array(embedding_dict['sequence_embeddings']) 
    if not index:
        print("Create index")
        d = np_embeddings.shape[1]
        index = faiss.index_factory(d, "Flat", faiss.METRIC_INNER_PRODUCT)

    faiss.normalize_L2(np_embeddings)
    index.add(np_embeddings)
    return(index)


def get_index_args():

    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--in", dest = "fasta_paths", nargs = "+", type = str, required = True,
                        help="Path to fastas (list)")
    
    parser.add_argument("-e", "--emb", dest = "embedding_path", type = str, required = False,
                        help="Path to embeddings")

    parser.add_argument("-o", "--outfile", dest = "out_path", type = str, required = True,
                        help="Path to outfile to store index")

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

    parser.add_argument("-t", "--truncate", dest = "truncate",  type = int, required = False, default = 12000,
                        help="Default 12000. (23000 is too long)")
 

    args = parser.parse_args()

    return(args)


if __name__ == '__main__':

    args = get_index_args()

    fasta_paths = args.fasta_paths
    embedding_path = args.embedding_path
    outfile = args.out_path
    layers = args.layers
    heads = args.heads
    model_name = args.model_name
    pca_plot = args.pca_plot
    headnorm = args.headnorm
    truncate = args.truncate
    # Keep to demonstrate effect of clustering or not
 

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
    index = None 
    count = 0

    index_key_outfile = "{}.idmapping".format(outfile) 

    if os.path.exists(index_key_outfile):
       print("Warning, appending to existing file")
       print("If unwanted, remove previous index before starting")

    with open(index_key_outfile, "a") as ok:
        for fasta_path in fasta_paths: 
        
            seq_names, seqs, seqs_spaced = parse_fasta_for_embed(fasta_path, truncate = truncate, padding = padding)
            seqlens = [len(x) for x in seqs] 
            embedding_dict = get_embeddings(seqs_spaced,
                                    model_name,
                                    seqlens = seqlens,
                                    get_sequence_embeddings = True,
                                    get_aa_embeddings = False,
                                    layers = layers,  
                                    padding = padding,
                                    heads = headnames)
            index = get_index(embedding_dict, index)
            faiss.write_index(index, outfile)
            # print("{} added to index".format(fasta_path))
            # Write mapping of sequence name o
            for seq_name in seq_names:
                 key_string =  "{},{}\n".format(seq_name, count)
                 ok.write(key_string)
                 count = count + 1 



