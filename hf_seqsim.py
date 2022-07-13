#!/usr/bin/env python3


#from statistics import NormalDist

# sigma = standarddev/sqrt(n)

# Store 1 array of mu
# Store 1 array of sigma
# search with mu to limit
# In limited space, do below 

#NormalDist(mu=2.5, sigma=1).overlap(NormalDist(mu=5.0, sigma=1))

# take average of all overlaps
#[0.1, 0.9, 0.3, 0.3] -> 0.8

# sqrt(sum of squares) for example 
# third root (sum of x^3's)
# The higher power, the more you pick up maximums. 
# Higher weights values closer to 1 higher

import torch
from transformers import AutoTokenizer, AutoModel
import numpy as np
from statistics import NormalDist
from scipy.stats import entropy
from scipy.sparse import diags
from scipy.spatial.distance import euclidean
import random
from transformer_infrastructure.hf_utils import build_index_flat, build_index_voronoi
from transformer_infrastructure.run_tests import run_tests
from transformer_infrastructure.hf_embed import parse_fasta_for_embed, get_embeddings 
import copy
from Bio import SeqIO
from Bio.Align import MultipleSeqAlignment
from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord
from scipy.stats import multivariate_normal
#import tensorflow as tf
from time import time
from sklearn.preprocessing import normalize
import faiss

import pickle
import argparse

import os
import sys
import igraph
from pandas.core.common import flatten 
import pandas as pd 

from numba import njit

from collections import Counter
import matplotlib.pyplot as plt
import logging

from sklearn.metrics.pairwise import cosine_similarity

# This is in the goal of finding sequences that poorly match before aligning
# SEQSIM




def kl_mvn(m0, S0, m1, S1):
    """
    Kullback-Liebler divergence from Gaussian pm,pv to Gaussian qm,qv.
    Also computes KL divergence from a single Gaussian pm,pv to a set
    of Gaussians qm,qv.
    

    From wikipedia
    KL( (m0, S0) || (m1, S1))
         = .5 * ( tr(S1^{-1} S0) + log |S1|/|S0| + 
                  (m1 - m0)^T S1^{-1} (m1 - m0) - N )
    """
    # store inv diag covariance of S1 and diff between means
    N = m0.shape[0]
    iS1 = np.linalg.inv(S1)
    diff = m1 - m0

    # kl is made of three terms
    tr_term   = np.trace(iS1 @ S0)
    det_term  = np.log(np.linalg.det(S1)/np.linalg.det(S0)) #np.sum(np.log(S1)) - np.sum(np.log(S0))
    quad_term = diff.T @ np.linalg.inv(S1) @ diff #np.sum( (diff*diff) * iS1, axis=1)
    #print(tr_term,det_term,quad_term)
    return .5 * (tr_term + det_term + quad_term - N) 

def graph_from_distindex(index, dist):  
    #print("Create graph from dist index with threshold {}".format(seqsim_thresh))
    edges = []
    weights = []
    complete = []
    for i in range(len(index)):
       #complete.append(i)
       for j in range(len(index[i])):
          #if j in complete:
          #    continue
          # Index should not return negative
          weight = dist[i,j]
          if weight < 0:
             #print("Index {}, {} returned negative similarity, replacing with 0.001".format(i,j))
             weight = 0.001
          edge = (i, index[i, j])
          #if edge not in order_edges:
          # Break up highly connected networks, simplify clustering
          #if scoretype == "cosinesim":
              #if weight >= seqsim_thresh:
          edges.append(edge)
          weights.append(weight)
          #if scoretype == "euclidean":
          #    if weight <= seqsim_thresh:
          #        edges.append(edge)
          #        weights.append(weight)

    print("edge preview", edges[0:15])
    G = igraph.Graph.TupleList(edges=edges, directed=True) # Prevent target from being placed first in edges
    G.es['weight'] = weights
    #G = G.simplify(combine_edges = "first")  # symmetrical, doesn't matter

    return(G)


# If removing a protein leads to less of a drop in total edgeweight that other proteins

def candidate_to_remove(G, v_names,z = -5):


    weights = {}  
    num_prots = len(G.vs())
    print("num_prots")
    if num_prots <=3:
        return([])

    for i in v_names:
        # Potentially put in function
        g_new = G.copy()
        vs = g_new.vs.find(name = i)
        weight = sum(g_new.es.select(_source=vs)['weight'])
        weights[i] = weight
        #weights.append(weight)
    questionable_z = []
    #print("Sequence z scores, current threshold: ", z)
    for i in v_names:

        others = []
        for key,value in weights.items():
            if key == i:
                own_value = value
            else:
                others.append(value)  

        #others = [weights[x] for x in range(len(weights)) if x != i]
        print(own_value, others)
        seq_z = (own_value - np.mean(others))/np.std(others)
        #seq_z = (weights[i] - np.mean(others))/np.std(others)
        print("sequence ", i, " zscore ", seq_z)

        # This should scale with # of sequences?
        # If on average high similarity, don't call as questionable even if high z
        # Avoid 1.65, 1.72, 1.71 three protein case. 
        #if (own_value / (num_prots - 1)) < 0.7:

        if seq_z < z:
            questionable_z.append(i)
       
    print("questionalbe_z", questionable_z) 
    return(questionable_z)



def get_seq_groups2(seqs, seq_names, embedding_dict, logging, exclude, do_clustering, seqsim_thresh= 0.75):
    numseqs = len(seqs)

    
    #hstates_list, sentence_embeddings = get_hidden_states(seqs, model, tokenizer, layers, return_sentence = True)
    #logging.info("Hidden states complete")
    #print("end hidden states")

    #if padding:
    #    logging.info("Removing {} characters of neutral padding X".format(padding))
    #    hstates_list = hstates_list[:,padding:-padding,:]

    #padded_seqlen = embedding_dict['aa_embeddings'].shape[1]
    #logging.info("Padded sequence length: {}".format(padded_seqlen))


    k_select = numseqs 
    sentence_array = np.array(embedding_dict['sequence_embeddings']) 

    #print("sentnece array shape", sentence_array.shape)
    if sentence_array.shape[1] > 1024:
       sentence_array = sentence_array[:,:1024]
    #print(sentence_array.shape)

    #print("sentence_array", sentence_array)

    #print(sentence_array.shape)
    s_index = build_index_flat(sentence_array)
    #print(numseqs, k_select)
    s_distance, s_index2 = s_index.search(sentence_array, k = k_select)
    #print(s_distance) 
    #print(s_index2)
    G = G.simplify(combine_edges = "first")  # symmetrical, doesn't matter
    #print(G)
    to_exclude = []

   
    group_hstates_list = []
    cluster_seqnums_list = []
    cluster_names_list = []
    cluster_seqs_list = []
   

    # TODO use two variable names for spaced and unspaced seqs
    logging.info("Removing spaces from sequences")
    #if padding:
    #    seqs = [x.replace(" ", "")[padding:-padding] for x in seqs]
    #else:
    #    seqs = [x.replace(" ", "") for x in seqs]
    #prev_to_exclude = []
    if do_clustering == True:
        #print("fastgreedy")
        #print(G)
    
      #repeat = True
      #
      #while repeat == True:
      d = sentence_array.shape[1]
      for k in range(1, 20):
         kmeans = faiss.Kmeans(d = d, k = k, niter = 20)
         kmeans.train(sentence_array)
    
   
         D, I = kmeans.index.search(sentence_array, 1) 
         print("D", D)
         print("I", I)
         clusters = I.squeeze()
         labels = list(zip(G.vs()['name'], clusters))
         #for x in labels:
         #    print("labels", x[0], x[1])


         group_hstates_list = []
         cluster_seqnums_list = []
         cluster_names_list = []
         cluster_seqs_list = []
 
         prev_to_exclude = to_exclude
        
         means = []
         for clustid in list(set(clusters)):
             print("eval clust", clustid)
             clust_seqs = [x[0] for x in labels if x[1] == clustid] 
             print("clust_seqs", clust_seqs)
             #print("labels from loop", labels)
             #for lab in labels:
             #     print("labels", lab, lab[0], lab[1], clustid) 
             #     if lab[1] == clustid:
             # 
              #         print("yes")
             #print("GG", G.vs()['name'])
             #print("GG", G.es()['weight'])
             #edgelist = []
             weightlist = []
             for edge in G.es():
                  #print(edge, edge['weight'])
                  #print(G.vs[edge.target]["name"], G.vs[edge.source]["name"])
                  if G.vs[edge.target]["name"] in clust_seqs:
                       if G.vs[edge.source]["name"] in clust_seqs:
                          weightlist.append(edge['weight'])
                          print(G.vs[edge.target]["name"], G.vs[edge.source]["name"], edge['weight'])
             print(weightlist)
             print("clust {} mean {}".format(clustid, np.mean(weightlist)))
             means.append(np.mean(weightlist))
         print("k {} overall mean {}".format(clustid, np.mean(means)))    

      #return(0)

def seq_index_search(sentence_array, k_select, s_index = None):

    #print("sentence_array", sentence_array)
    if not s_index:
        s_index = build_index_flat(sentence_array, scoretype = "cosinesim")

    #sentence_array, norm = normalize(sentence_array, norm='l2', axis=0, copy=True, return_norm=True)
    #faiss.normalize_L2(sentence_array)

    s_distance, s_index2 = s_index.search(sentence_array, k = k_select)
    return(s_distance, s_index2)

def get_seqsims(sentence_array, k = None, s_index = None):

    print("k", k)
    if k:
       k_select = k
    else:
       k_select = numseqs
    start_time = time()

    print("Searching index")
    s_distance, s_index2 = seq_index_search(sentence_array, k_select, s_index)
    
    #s_distance = (2-s_distance)/2

    end_time = time()
    print("Index searched for {} sequences in {} seconds".format(numseqs, end_time - start_time))
    #if s_sigma_index:
    #print("get_seqsims:s_index2:",s_index2) 
        

    
    #else:
    start_time = time()
    G = graph_from_distindex(s_index2, s_distance)
    end_time = time()
    print("Index converted to edges in {} seconds".format(end_time - start_time))
    return(G)


def get_seqsim_args():

    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--in", dest = "fasta_path", type = str, required = True,
                        help="Path to fasta")
    
    parser.add_argument("-e", "--emb", dest = "embedding_path", type = str, required = False,
                        help="Path to embeddings")

    parser.add_argument("-o", "--outfile", dest = "out_path", type = str, required = True,
                        help="Path to outfile")
    parser.add_argument("-ml", "--minlength", dest = "minlength", type = int, required = False,
                        help="If present, minimum length of sequences to search against index")

    parser.add_argument("-ex", "--exclude", dest = "exclude", action = "store_true",
                        help="Exclude outlier sequences from initial alignment process")

    parser.add_argument("-dx", "--index_means", dest = "index_file", required = False,
                        help="Prebuilt index of means")
    parser.add_argument("-dxs", "--index_sigmas", dest = "index_file_sigmas", required = False,
                        help="Prebuilt index of sigmas (standard deviations)")
    parser.add_argument("-dxn", "--index_names", dest = "index_names_file", required = False,
                        help="Prebuilt index names, One protein name per line, in order added to index")

    parser.add_argument("-ss", "--strategy", dest = "strat", type = str, required = False, default = "mean", choices = ['mean','meansig'],
                        help="Whether to search with cosine similarity of mean only, or follow by comparison of gaussians")

    parser.add_argument("-fx", "--fully_exclude", dest = "fully_exclude", action = "store_true",
                        help="Additionally exclude outlier sequences from final alignment")

    parser.add_argument("-l", "--layers", dest = "layers", nargs="+", type = int,
                        help="Which layers (of 30 in protbert) to select")
    parser.add_argument("-hd", "--heads", dest = "heads", type = str,
                        help="File will one head identifier per line, format layer1_head3")

    parser.add_argument("-st", "--seqsimthresh", dest = "seqsimthresh",  type = float, required = False, default = 0.75,
                        help="Similarity threshold for clustering sequences")
    parser.add_argument("-s", "--scoretype", dest = "scoretype",  type = str, required = False, default = "cosinesim", choices = ["cosinesim", "euclidean"],
                        help="How to calculate initial sequence similarity score")
    parser.add_argument("-k", "--knn", dest = "k",  type = int, required = False,
                        help="Limit edges to k nearest neighbors")


    parser.add_argument("-m", "--model", dest = "model_name",  type = str, required = True,
                        help="Model name or path to local model")

    parser.add_argument("-p", "--pca_plot", dest = "pca_plot",  action = "store_true", required = False, 
                        help="If flagged, output 2D pca plot of amino acid clusters")

    parser.add_argument("-l2", "--headnorm", dest = "headnorm",  action = "store_true", required = False, 
                        help="Take L2 normalization of each head")
 

    args = parser.parse_args()

    return(args)


def kl_gauss(m1, m2, s1, s2):
    kl = np.log(s2/s1) + (s1**2 + (m1-m2)**2)/(2*s2**2) - 1/2
    return(kl)



def get_ovl(m1, m2, s1, s2):
   ovl = NormalDist(mu=m1, sigma=s1).overlap(NormalDist(mu=m2, sigma=s2)) 
   return(ovl)

def get_w2(m1, m2, s1, s2):
 
    w2 = np.sqrt((m1 - m2)**2 + (s1 - s2)**2)
    return(w2) 

@njit
def numba_w2(m1, m2, s1, s2, w2):

    for i in range(m1.shape[0]):
        w2[int(i)] = np.sqrt((m1[i] - m2[i])**2 + (s1[i] - s2[i])**2)
 
    return w2

@njit
def numba_ovl(m1, m2, s1, s2, o):
    for i in range(m1.shape[0]):
          o[int(i)] = NormalDist(mu=m1[i], sigma=s1[i]).overlap(NormalDist(mu=m2[i], sigma=s2[i])) 
    return(o) 
    

if __name__ == '__main__':
    true_start = time()
    args = get_seqsim_args()
    print("args parsed", time() - true_start)
    fasta_path = args.fasta_path
    embedding_path = args.embedding_path
    minlength = args.minlength
    outfile = args.out_path
    exclude = args.exclude
    fully_exclude = args.fully_exclude
    layers = args.layers
    heads = args.heads
    index_file = args.index_file
    index_file_sigmas = args.index_file_sigmas
    index_names_file = args.index_names_file
    model_name = args.model_name
    pca_plot = args.pca_plot
    headnorm = args.headnorm
    seqsim_thresh  = args.seqsimthresh 
    k = args.k
    strat = args.strat 
    scoretype = args.scoretype
    # Keep to demonstrate effect of clustering or not
    #do_clust   return(ovl)ering = True
 
    logname = "align.log"
    #print("logging at ", logname)
    log_format = "%(asctime)s::%(levelname)s::"\
             "%(filename)s::%(lineno)d::%(message)s"
    logging.basicConfig(filename=logname, level='DEBUG', format=log_format)



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
    logging.info("fasta: {}".format(fasta_path))
    logging.info("padding: {}".format(padding))

    faiss.omp_set_num_threads(10) 
    if minlength:
      seq_names, seqs, seqs_spaced = parse_fasta_for_embed(fasta_path, padding = padding, minlength = minlength)
    else:
      seq_names, seqs, seqs_spaced = parse_fasta_for_embed(fasta_path, padding = padding)
    print("seqs parsed", time() - true_start)
   
    if embedding_path:
       with open(embedding_path, "rb") as f:
             embedding_dict = pickle.load(f)

    else:
        seqlens = [len(x) for x in seqs]

        embedding_dict = get_embeddings(seqs_spaced,
                                    model_name,
                                    seqlens = seqlens,
                                    get_sequence_embeddings = True,
                                    get_aa_embeddings = False,
                                    layers = layers,  
                                    padding = padding,
                                    heads = headnames, 
                                    strat = strat)
    print("embeddings made", time() - true_start)
    print("getting sequence similarities") 
    if index_file:
         if not index_names_file:
            print("Provide file of index names in order added to index")
            exit(1)
         else:
            with open(index_names_file, "r") as infile:
                #index_names = infile.readlines()
                #index_names = [x.replace("\n", "").split(",") for x in index_names]
                #Read as {idx:proteinID}
                df = pd.read_csv(infile, header= None)
                df.columns = ['prot', 'idx']

                index_names = dict(zip(df.idx,df.prot))
                #index_names = index_names.set_index(['idx'])
                #print(index_names)
                #index_names = index_names.to_dict('index')
                #print(index_names) 
                
         # Don't use seqnames from input fasta, use index seqnames
         start_time = time()
         s_index = faiss.read_index(index_file)
         if strat == "meansig":
            if index_file_sigmas:
                s_sigma_index = faiss.read_index(index_file_sigmas)
            else:
                s_sigma_index = None
                s_sigma_embeddings = np.array(embedding_dict['sequence_embeddings_sigma']).astype(np.float32)
                s_sigma_index = build_index_flat(sigma_embeddings, s_sigma_index)

         end_time = time()

         print("Loaded index(es) in {} seconds".format(end_time - start_time))

    else:
         s_index = None
         s_sigma_index = None
         index_names = seq_names
            
    #kl = tf.keras.losses.KLDivergence()
    # Step 1: Use means to get local area of sequences
    sentence_array = np.array(embedding_dict['sequence_embeddings']).astype(np.float32) 
    if not k:
       k = len(seqs)
    G = get_seqsims(sentence_array, k = k, s_index = s_index)

    print("similarities made", time() - true_start)
    print(outfile)
    print("#_vertices", len(G.vs()))
    print("query_names", len(seq_names))
    print("index_names", len(index_names))


    named_vertex_list = G.vs()["name"]
    print(named_vertex_list)
    retrieve_start_time = time()
    target_mean_dict =  dict([(x, s_index.reconstruct(int(x))) for x in named_vertex_list])     
    target_sigma_dict = dict([(x, s_sigma_index.reconstruct(int(x))) for x in named_vertex_list])     
    retrieve_end_time = time()
    amount = retrieve_end_time - retrieve_start_time
    print("Vectors retrieved from index in ", amount)
    vec_kl_gauss = np.vectorize(kl_gauss)
    vec_get_ovl = np.vectorize(get_ovl)
    vec_get_w2 = np.vectorize(get_w2)
    sentence_array =  embedding_dict['sequence_embeddings'] 
    #faiss.normalize_L2(sentence_array)
    sigma_array = embedding_dict['sequence_embeddings_sigma']
    #faiss.normalize_L2(sigma_array)
    #sentence_array_l2norm = normalize(sentence_array, norm='l2', axis=1, copy=True)


    with open(outfile, "w") as o:
        #o.write("source,target,score,overlap,kl,w2_mean,w2_vec,euc_mean,euc_sigma\n")
        o.write("source,target,distance,cosinesim,w2_mean,w2_mean_neg_e,w2_mean_neg_e_1_10\n")
        e_start = time()
        for edge in G.es():
           #print(edge)
           #print(G.vs()[edge.source], G.vs()[edge.target], edge['weight'])
           source_idx = int(G.vs()[edge.source]['name'])
           target_idx = int(G.vs()[edge.target]['name'])
           #print(source_idx, target_idx)
           if source_idx == -1:
               continue
           if target_idx == -1:
               continue
           source = seq_names[source_idx]
           target = index_names[target_idx] 
           weight = edge['weight']
          
           d_start = time()
           source_mean = sentence_array[source_idx]
           source_sigma =  sigma_array[source_idx]
           #print(source_mean)
           #source_mean = vertex_mean_dict[source_idx]
           #source_sigma = vertex_sigma_dict[source_idx]
           target_mean = target_mean_dict[target_idx]
           #print(target_mean)
           target_sigma = target_sigma_dict[target_idx]
           #print(source_mean)
           #print(target_mean)
           #print(source_sigma)
           #print(target_sigma)
           #d_end = time()
           d_span = time()  -d_start

           cosinesim = cosine_similarity([source_mean], [target_mean]) 
           #print(cosinesim)
           #print(cosinesim[0][0])
           cosinesim = cosinesim[0][0]
           ##source_mean  =  s_index.reconstruct(source_idx)
           #source_sigma  =  s_sigma_index.reconstruct(source_idx)
           #target_mean  =  s_index.reconstruct(target_idx)
           #target_sigma  =  s_sigma_index.reconstruct(target_idx)

           #print("source_mean", source_mean)
           #print("source_sigma", source_sigma)
           #
           # Do overlaps of each row
           #arr =np.array([source_mean, target_mean, source_sigma, target_sigma])
           #print(arr)
           #o_start = time()

           # This is too slow
           #overlaps = [NormalDist(mu=m1, sigma=s1).overlap(NormalDist(mu=m2, sigma=s2)) for m1, m2, s1, s2 in zip(source_mean, target_mean, source_sigma, target_sigma)]
 
           #mean_overlap = np.mean(overlaps)
           #o_end = time()
           #o_span = time() - o_start
           #print(overlaps[0:5])
           #overlaps = NormalDist(mu=source_mean, sigma=source_sigma).overlap(NormalDist(mu=target_mean, sigma=target_sigma))

           ###m_start = time()

           #print("start kl")
           ###kls = vec_kl_gauss(source_mean, target_mean, source_sigma, target_sigma)
           #print(kls)
           ###kl_out = 1- np.mean(kls)       
  
           #kl = kl_mvn(source_mean, source_sigma, target_mean, target_sigma)
           #dim = len(source_mean)
           #source_cov = diags(source_sigma, 0).toarray()
           #target_cov = diags(target_sigma, 0).toarray()
           #source_cov = np.zeros((dim,dim))
           #np.fill_diagonal(source_cov, source_sigma) # This is inplace
           #target_cov = np.zeros((dim,dim))
           #np.fill_diagonal(target_cov, target_sigma) # This is inplace

           #np.random.seed(10)
           ###m_span = time() - m_start
          ### k_start = time()
           #kls = [kl_gauss(m1, s1, m2, s2) for  m1, m2, s1, s2 in zip(source_mean, target_mean, source_sigma, target_sigma)]    
           #ovls = vec_get_ovl(source_mean, target_mean, source_sigma, target_sigma)
           #ovl = np.mean(ovls)
           ###k_span = time() - k_start
           #x = np.random.normal(source_mean, source_sigma)
           #y = np.random.normal(target_mean, target_sigma)
           #x = np.random.default_rng().multivariate_normal(source_mean, source_cov, method = "cholesky", size = 1)
           #y = np.random.default_rng().multivariate_normal(target_mean, target_cov, method = "cholesky", size  =1)
          
           #print(x)
           #print(y)
           #print("calc entropy") 
           #kl_out = 0# entropy(x+ 0.0001,y+ 0.0001)
           #print("end kl") 
           #kl_out = kl(x, y).numpy()

           #kl_out = kl_mvn(source_mean, source_cov, target_mean, target_cov)
           #rv = multivariate_normal([mu_x, mu_y], [[sigma_x, 0], [0, sigma_y]])

           ##w2_start = time()
           ##w2s = vec_get_w2(source_mean, target_mean, source_sigma, target_sigma)

           #print("source_mean",source_mean)
           #print("target_mean", target_mean)
           #print("source_sigma",source_sigma)
           #print("target_sigma", target_sigma)
           #print("maxes", max(source_mean), max(target_mean), max(source_sigma), max(target_sigma))
           #print("w2s", w2s)
           ##w2_out = 1 - np.mean(w2s)
           ##w2_span = time() - w2_start

           ##w2_vect_start = time()
           ##w2_vect = 1 - (np.sqrt(euclidean(source_mean, target_mean)**2 + euclidean(source_sigma, target_sigma)**2))/len(source_mean)
           ##w2_vect_span = time() - w2_vect_start
           #e_span = time() - e_start

           nb_w2_vect_start = time()
           nb_w2_vect = np.empty(source_mean.shape[0] , dtype=np.float32)
           #print(nb_w2_vect)
           nb_w2_vect = numba_w2(source_mean, target_mean,source_sigma, target_sigma, nb_w2_vect)
           nb_w2_vect_span = time() - nb_w2_vect_start
           #print("nb vect", nb_w2_vect)
           #w2_out = 1 - np.mean(nb_w2_vect) # Wrong, not bounded by 1
           mean_w2 = np.mean(nb_w2_vect)
           w2_out = 1/(1 + mean_w2)  # somewhat flips
           w2_e_out = np.exp(-mean_w2)
           w2_ediv_out = np.exp(-mean_w2/10)
           #nb_o_vect_start = time()
           #nb_o_vect = np.empty(source_mean.shape[0] , dtype=np.float32)
           #print(nb_o_vect)
           #nb_o_vect = numba_ovl(source_mean, target_mean,source_sigma, target_sigma, nb_o_vect)
           #nb_o_vect_span = time() - nb_o_vect_start


           ##euc_mean = euclidean(source_mean, target_mean)
           ##euc_sigma = euclidean(source_sigma, target_sigma)

           #print( "ovl", ovl, "kl", kl_out, "avg_w2", w2_out, "nb_avg_w2", nb_w2_vect, "w2_vect", w2_vect,  "cossim", edge['weight'],  "total_time", e_span,  "dict_time", d_span,  "vec_overlap time", k_span, "kl_time", m_span, "w2_time", w2_span, "w2_v_time", w2_vect_span, "nb_w2_time", nb_w2_vect_span)
           

           if source == target:
                if weight < 0.99:
                      print("Warning, score for {} and {} should be close to 1, but is {}. check indices".format(source, target, weight))
                #continue
           #print(source,target,weight,cosinesim,w2_out, ovl)

           #o.write("{},{},{:.5f},{:.5f},{:.5f},{:.5f},{:.10f},{},{},{}\n".format(source, target, weight, ovl, kl_out, w2_out, w2_vect, euc_mean, euc_sigma, nb_w2_vect))   
           o.write("{},{},{:.5f},{:.5f},{:.5f},{:.5f},{:.5f}\n".format(source,target,weight,cosinesim,w2_out,w2_e_out,w2_ediv_out))
    e_span = time() - e_start
    print("second similarty taken in {} seconds".format(e_span))
    print("outfile made", time() - true_start)
  
    # Step 2: A this point take everything about mean similarity threshold and do distribution comparison
#    for edge in G.es():

           #np.take(embedding_dict['sequence_embeddings'], [source_idx], axis = 0)   
           #source_sigma  =# np.take(embedding_dict['sequence_embeddings_sigma'], [source_idx], axis = 0)   

