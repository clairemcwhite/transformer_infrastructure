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


def get_clusterindex_args():

    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--in", dest = "fasta_paths", nargs = "+", type = str, required = False,
                        help="Path to fastas (list), only required if no existing sequence index")

    parser.add_argument("-c", "--clusters", dest = "cluster_file", required = True,
                        help="File of clusters, one per line, each cluster member tab separated (same as mcl output)")
    parser.add_argument("-dx", "--index_means", dest = "index_file", required = False,
                        help="Prebuilt index of means")
    parser.add_argument("-dxs", "--index_sigmas", dest = "index_file_sigmas", required = False,
                        help="Prebuilt index of sigmas (standard deviations)")
    parser.add_argument("-dxn", "--index_names", dest = "index_names_file", required = False,
                        help="Prebuilt index names, One protein name per line, in order added to index")

    
    parser.add_argument("-e", "--emb", dest = "embedding_path", type = str, required = False,
                        help="Path to embeddings")

    parser.add_argument("-b", "--base_outfile", dest = "base_outfile", type = str, required = True,
                        help="Path to outfile basename to store index(es) and id mapping, b.mean.faissindex, b.sigma.faissindex, b.faissindex.idmapping")
    parser.add_argument("-ss", "--strategy", dest = "strat", type = str, required = False, default = "meansig", choices = ['mean','meansig'],
                        help="Save index of mean, or two indexes, one mean, one sigma")

    parser.add_argument("-l", "--layers", dest = "layers", nargs="+", type = int, required = False,
                        help="Which layers (of 30 in protbert) to select")
    parser.add_argument("-hd", "--heads", dest = "heads", type = str,
                        help="File will one head identifier per line, format layer1_head3")

    parser.add_argument("-m", "--model", dest = "model_name",  type = str, required = False,
                        help="Model name or path to local model")

    parser.add_argument("-p", "--pca_plot", dest = "pca_plot",  action = "store_true", required = False, 
                        help="If flagged, output 2D pca plot of clusters")

    parser.add_argument("-l2", "--headnorm", dest = "headnorm",  action = "store_true", required = False, 
                        help="Take L2 normalization of each head")

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

    args = get_clusterindex_args()

    fasta_paths = args.fasta_paths
    embedding_path = args.embedding_path
    base_outfile = args.base_outfile
    layers = args.layers
    heads = args.heads
    cluster_file = args.cluster_file
    model_name = args.model_name
    pca_plot = args.pca_plot
    headnorm = args.headnorm
    truncate = args.truncate
    strat = args.strat
    scoretype = args.scoretype
    index_file = args.index_file
    index_file_sigmas = args.index_file_sigmas
    index_names_file = args.index_names_file

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
    cluster_mean_index = None 
    cluster_sigma_index = None 
    count = 0

    index_key_outfile = "{}.faissindex.clustidmapping".format(base_outfile) 
    mean_outfile =  "{}.mean.clusterfaissindex".format(base_outfile) 
    sigma_outfile =  "{}.sigma.clusterfaissindex".format(base_outfile) 

    if os.path.exists(index_key_outfile):
       os.remove(index_key_outfile)
       os.remove(mean_outfile)
       os.remove(sigma_outfile)
       #print("Warning, appending to existing file")
       #print("If unwanted, remove previous index before starting")


    print("Read cluster file (mcl output)") 
    clust_tbl = pd.read_csv(cluster_file, header = None) # Gets one column, containing string of tab separate clutside
    clust_tbl['clustid'] = clust_tbl.index + 1
    clust_tbl = clust_tbl.set_index(['clustid'])
    clust_dict_tmp = clust_tbl.to_dict()[0]
    cluster_dict = {"cluster{}".format(k): v.split("\t") for k, v in clust_dict_tmp.items()} 

    if index_file:
         print("Get sequences from existing index")
         if not index_names_file:
            print("Provide file of index names in order added to index")
            exit(1)
         else:
            with open(index_names_file, "r") as infile:
                df = pd.read_csv(infile, header= None)
                df.columns = ['prot', 'idx']

                index_names_prot_idx = dict(zip(df.prot,df.idx))

         # Don't use seqnames from input fasta, use index seqnames
         mean_index = faiss.read_index(index_file)
         if strat == "meansig":
            if index_file_sigmas:
                sigma_index = faiss.read_index(index_file_sigmas)


    seq_dict = {}
    with open(index_key_outfile, "a") as ok:
        if not index_file:
             # Only need fasta if no existing index
             for fasta_path in fasta_paths: 
        
                seq_names, seqs, seqs_spaced = parse_fasta_for_embed(fasta_path, truncate = truncate, padding = padding, minlength=140)
                new_seqs_dict = dict(zip(seq_names, seqs_spaced))
                seq_dict = { **seq_dict, **new_seqs_dict }
           
             print("Sequences loaded") 
        # Get a group of sequences

        # { clustid:[seqname1, seqname2] }:
        for clustid, clust in cluster_dict.items():
            clust_seq_names = clust
           
            if index_file:
                prots_in_mapping = [x for x in clust if x in index_names_prot_idx.keys()]
                print(prots_in_mapping)
                clust_idx = [index_names_prot_idx[x] for x in prots_in_mapping]
                   
                print(clustid, clust_idx) 
                print(clust)
                if len(clust_idx) == 0:
                    continue
                allseqs_mean_embeddings =  np.array([mean_index.reconstruct(int(x)) for x in clust_idx]).astype(np.float32)
                allseqs_sigma_embeddings = np.array([ sigma_index.reconstruct(int(x)) for x in clust_idx]).astype(np.float32)

                #print("allseqs mean", allseqs_mean_embeddings)
                #print("allseqs sigma", allseqs_sigma_embeddings)
            else:

                clust_seqs_spaced = [seq_dict[x] for x in clust_seq_names]
                seqlens = [ len(x) for x in clust_seqs_spaced ] 

                embedding_dict = get_embeddings(clust_seqs_spaced,
                                        model_name,
                                        seqlens = seqlens,
                                        get_sequence_embeddings = True,
                                        get_aa_embeddings = False,
                                        layers = layers,  
                                        padding = padding,
                                        heads = headnames, 
                                        strat = strat)
       
                allseqs_mean_embeddings = np.array(embedding_dict['sequence_embeddings']).astype(np.float32)

                if strat == "meansig":
                    allseqs_sigma_embeddings = np.array(embedding_dict['sequence_embeddings_sigma']).astype(np.float32)

            cluster_mean_embedding = np.array([np.mean(allseqs_mean_embeddings, axis = 0)]).astype(np.float32)

            #print("cluster mean embedding", cluster_mean_embedding)
            if scoretype == "euclidean":
                cluster_mean_index = build_index_flat(cluster_mean_embedding, index = cluster_mean_index, scoretype = scoretype, normalize_l2 = False, return_norm = False)
            else:
                cluster_mean_index, norm = build_index_flat(cluster_mean_embedding, index = cluster_mean_index, scoretype = scoretype, normalize_l2 = True, return_norm = True)
        
            faiss.write_index(cluster_mean_index, mean_outfile)
            if strat == "meansig":

                cluster_sigma_embedding = np.array([np.sum(allseqs_sigma_embeddings, axis = 0)/allseqs_sigma_embeddings.shape[0]**2 ]).astype(np.float32)

                cluster_sigma_index = build_index_flat(cluster_sigma_embedding, index = cluster_sigma_index, scoretype = "euclidean", normalize_l2 = False, return_norm = False)
                faiss.write_index(cluster_sigma_index, sigma_outfile)

            key_string =  "{},{}\n".format(clustid, count)
            ok.write(key_string)
            count = count + 1 
    


