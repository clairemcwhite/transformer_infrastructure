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

from collections import Counter
import matplotlib.pyplot as plt
import logging

from sklearn.metrics.pairwise import cosine_similarity

# This is in the goal of finding sequences that poorly match before aligning
# SEQSIM
def graph_from_distindex(index, dist, seqsim_thresh = 0):  
    # THIS ISN'T right for prebuilt index
    # Search get index of query? 
    # Or add query sequences to index?
    print("Create graph from dist index with threshold {}".format(seqsim_thresh))
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
          if weight >= seqsim_thresh:
              edges.append(edge)
              weights.append(weight)
    print("edge preview", edges[0:10])
    G = igraph.Graph.TupleList(edges=edges, directed=True) # Prevent target from being placed first in edges
    G.es['weight'] = weights
    G = G.simplify(combine_edges = "first")  # symmetrical, doesn't matter

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

    padded_seqlen = embedding_dict['aa_embeddings'].shape[1]
    logging.info("Padded sequence length: {}".format(padded_seqlen))


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
        s_index = build_index_flat(sentence_array)

    faiss.normalize_L2(sentence_array)
    s_distance, s_index2 = s_index.search(sentence_array, k = k_select)
    return(s_distance, s_index2)

def get_seqsims(seqs, embedding_dict, seqsim_thresh = 0.75, k = None, s_index = None):
    numseqs = len(seqs)

    print("seqsim_thresh", seqsim_thresh)
    print("k", k)
    #padded_seqlen = embedding_dict['aa_embeddings'].shape[1]
    #logging.info("Padded sequence length: {}".format(padded_seqlen))
    #print(embedding_dict['aa_embeddings'].shape)
    # numseqs x seqlen x  dimension
    #aa_array = np.array(embedding_dict['aa_embeddings'])
    #print("AA", aa_array.shape)
    ##sentence_array = aa_array[:,:, :4*1024]
    #print("AA", aa_array.shape)
      
    ## Get sentence embeddings by averaging aa embeddings
    #sentence_array = np.mean(aa_array, axis = 1)
    #print("sent", sentence_array.shape)
    # print(sentence_array)
    if k:
       k_select = k
    else:
       k_select = numseqs
    sentence_array = np.array(embedding_dict['sequence_embeddings']) 
    start_time = time()

    print("Searching index")
    s_distance, s_index2 = seq_index_search(sentence_array, k_select, s_index)
    end_time = time()
    print("Index searched for {} sequences in {} seconds".format(numseqs, end_time - start_time))
    start_time = time()
    G = graph_from_distindex(s_index2, s_distance, seqsim_thresh)
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


    parser.add_argument("-ex", "--exclude", dest = "exclude", action = "store_true",
                        help="Exclude outlier sequences from initial alignment process")

    parser.add_argument("-dx", "--index", dest = "index_file", required = False,
                        help="Prebuilt index")
    parser.add_argument("-dxn", "--index_names", dest = "index_names_file", required = False,
                        help="Prebuilt index names, One protein name per line, in order added to index")


    parser.add_argument("-fx", "--fully_exclude", dest = "fully_exclude", action = "store_true",
                        help="Additionally exclude outlier sequences from final alignment")

    parser.add_argument("-l", "--layers", dest = "layers", nargs="+", type = int,
                        help="Which layers (of 30 in protbert) to select")
    parser.add_argument("-hd", "--heads", dest = "heads", type = str,
                        help="File will one head identifier per line, format layer1_head3")

    parser.add_argument("-st", "--seqsimthresh", dest = "seqsimthresh",  type = float, required = False, default = 0.75,
                        help="Similarity threshold for clustering sequences")

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


if __name__ == '__main__':
    true_start = time()
    args = get_seqsim_args()
    print("args parsed", time() - true_start)
    fasta_path = args.fasta_path
    embedding_path = args.embedding_path
    outfile = args.out_path
    exclude = args.exclude
    fully_exclude = args.fully_exclude
    layers = args.layers
    heads = args.heads
    index_file = args.index_file
    index_names_file = args.index_names_file
    model_name = args.model_name
    pca_plot = args.pca_plot
    headnorm = args.headnorm
    seqsim_thresh  = args.seqsimthresh 
    k = args.k

    # Keep to demonstrate effect of clustering or not
    #do_clustering = True
 
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
                                    heads = headnames)
    print("embeddings made", time() - true_start)
    print("getting sequence similarities") 
    if index_file:
         if not index_names_file:
            print("Provide file of index names in order added to index")
            exit(1)
         else:
            with open(index_names_file, "r") as infile:
                index_names = infile.readlines()
                index_names = [x.replace("\n", "") for x in index_names]
         # Don't use seqnames from input fasta, use index seqnames
         #seq_names = index_names
         start_time = time()
         s_index = faiss.read_index(index_file)
         end_time = time()
         print("Loaded index in {} seconds".format(end_time - start_time))

    else:
         s_index = None
         index_names = seq_names
            


    G = get_seqsims(seqs, embedding_dict, seqsim_thresh = seqsim_thresh, k = k, s_index = s_index)

    print("similarities made", time() - true_start)
    print(outfile)
    print("#_vertices", len(G.vs()))
    print("query_names", len(seq_names))
    print("index_names", len(index_names))
    with open(outfile, "w") as o:
        
        for edge in G.es():
           #print(edge)
           #print(G.vs()[edge.source], G.vs()[edge.target])
           source = seq_names[G.vs()[edge.source]['name']]
           target = index_names[G.vs()[edge.target]['name']]
           weight = edge['weight']
           if source == target:
                if weight < 0.99:
                      print("Warning, score for {} and {} should be close to 1, but is {}. check indices".format(source, target, weight))
                continue
           o.write("{},{},{:.5f}\n".format(source, target, weight))   
    print("outfile made", time() - true_start)

    # Padding irrelevant at this point 
    #cluster_seqnums_list, cluster_seqs_list,  cluster_names_list, cluster_hstates_list, to_exclude = get_seq_groups(seqs ,seq_names, embedding_dict, logging, exclude, do_clustering, seqsim_thresh = seqsim_thresh)


#def cluster_seqsims(edges, weights):
#
#    # Convert to graph object
#    
#    to_exclude = []
#
#   
#    group_hstates_list = []
#    cluster_seqnums_list = []
#    cluster_names_list = []
#    cluster_seqs_list = []
#   
#
#    # TODO use two variable names for spaced and unspaced seqs
#    logging.info("Removing spaces from sequences")
#    #if padding:
#    #    seqs = [x.replace(" ", "")[padding:-padding] for x in seqs]
#    #else:
#    #    seqs = [x.replace(" ", "") for x in seqs]
#    #prev_to_exclude = []
#    if do_clustering == True:
#        #print("fastgreedy")
#        #print(G)
#    
#      repeat = True
#      while repeat == True:
#
#        group_hstates_list = []
#        cluster_seqnums_list = []
#        cluster_names_list = []
#        cluster_seqs_list = []
# 
#        prev_to_exclude = to_exclude
#        
#
#    
#        print("GG", G.vs()['name'])
#        print("GG", G.es()['weight'])
#        edgelist = []
#        weightlist = []
#        for edge in G.es():
#             print(edge, edge['weight'])
#             if G.vs[edge.target]["name"] not in to_exclude:
#                  if G.vs[edge.source]["name"] not in to_exclude:
#                     edgelist.append([ G.vs[edge.source]["name"], G.vs[edge.target]["name"]])
#                     weightlist.append(edge['weight'])
#        # Rebuild G
#        G = igraph.Graph.TupleList(edges=edgelist, directed=False)
#        G.es['weight'] = weightlist
#        print("G", G)
#
#
#
#        seq_clusters = G.community_multilevel(weights = 'weight')
#        ## The issue with walktrap is that the seq sim graph is near fully connected
#        print("multilevel", seq_clusters)
#        seq_clusters = G.community_walktrap(steps = 3, weights = 'weight').as_clustering() 
#        print("walktrap", seq_clusters)
#        seq_clusters = G.community_fastgreedy(weights = 'weight').as_clustering() 
#        print("fastgreedy", seq_clusters)
#
#        seq_clusters = G.community_walktrap(steps = 3, weights = 'weight').as_clustering() 
#        #for x in seq_clusters.subgraphs():
#        #     print("subgraph", x)      
#        if len(seq_clusters.subgraphs()) == len(G.vs()):
#        #     #seq_cluster = seq_clusters.vs()['name']
#             seq_clusters = G.clusters(mode = "weak") # walktrap can cluster nodes individually. See UBQ
#         #                     # If that happens, use original graph
#         #    print("is this happening")
#
#        # Spinglass doesn't work on disconnected graphs
#        # Spinglass wins. See  Eg. Extradiol_dioxy 
#        #G_weak  = G.clusters(mode = "weak")
#        #for sub_G in G_weak.subgraphs():
#        #    sub_seq_clusters = sub_G.community_spinglass(weights = 'weight') 
#        #        
#        #    print("spinglass", sub_seq_clusters)
#        #    for seq_cluster_G in sub_seq_clusters.subgraphs():
#        print("walktrap", seq_clusters)
#        for seq_cluster_G in seq_clusters.subgraphs():
#        
#                # Do exclusion within clusters
#                print("seq_clusters", seq_cluster_G)
#                if exclude == True:
#    
#                    clust_names = seq_cluster_G.vs()["name"]
#                    print("clust_names", clust_names)
#                    cluster_to_exclude = candidate_to_remove(seq_cluster_G, clust_names, z = -5)
#                    print(cluster_to_exclude)
#                       
#                    #print('name', to_exclude)
#                    to_delete_ids_sub_G = [v.index for v in seq_cluster_G.vs if v['name'] in cluster_to_exclude]
#                    #print('vertix_id', to_delete_ids)
#                    seq_cluster_G.delete_vertices(to_delete_ids_sub_G) 
#    
#                    #to_delete_ids_G = [v.index for v in G.vs if v['name'] in cluster_to_exclude]
#                    #G.delete_vertices(to_delete_ids_G)
#    
#                    print("to_exclude_pre", to_exclude)
#                    to_exclude = to_exclude + cluster_to_exclude
#                    to_exclude = list(set(to_exclude))
#                    print("to_exclude_post", to_exclude)
#                    if to_exclude:       
#                        logging.info("Excluding following sequences: {}".format(",".join([str(x) for x in to_exclude])))
#                        print("Excluding following sequences: {}".format(",".join([str(x) for x in to_exclude])))
#    
#                hstates = []
#                seq_cluster = seq_cluster_G.vs()['name']
#                seq_cluster.sort()
#                print(seq_cluster)
#                cluster_seqnums_list.append(seq_cluster)
#        
#                filter_indices = seq_cluster
#                group_hstates = np.take(embedding_dict['aa_embeddings'], filter_indices, axis = 0)
#                group_hstates_list.append(group_hstates)
#                #Aprint(group_hstates.shape)
#        
#                cluster_names = [seq_names[i] for i in filter_indices]
#                cluster_names_list.append(cluster_names)
#           
#                cluster_seq = [seqs[i] for i in filter_indices]
#                cluster_seqs_list.append(cluster_seq)
#                to_exclude = list(set(to_exclude))
#        print("eq check", to_exclude, prev_to_exclude)
#        if set(to_exclude) == set(prev_to_exclude):
#           repeat = False
#        else:
#               cluster_seqs_list = [] 
#               cluster_seqnums_list = []
#               group_hstates_list = []
#               cluster_names_list= []
#    else:
#         if exclude == True:
#            clust_names = G.vs()["name"] 
#            to_exclude = candidate_to_remove(G, clust_names, z = -3)
#            print('name', to_exclude)
#            to_delete_ids = [v.index for v in G.vs if v['name'] in to_exclude]
#            #print('vertix_id', to_delete_ids)
#            G.delete_vertices(to_delete_ids) 
#    
#            logging.info("Excluding following sequences: {}".format(",".join([str(x) for x in to_exclude])))
#    
#         else:
#           logging.info("Not removing outlier sequences")
#           to_exclude = []
# 
# 
#        # #print([v['name'] for v in G.vs])
#         cluster_seqnums_list =  [v['name'] for v in G.vs]
#         print(cluster_seqnums_list, to_exclude)
#         cluster_seqnums_list = list(set(cluster_seqnums_list))
#         cluster_seqnums_list.sort()
#         # Make sure this is removing to_exclude corectly
#         cluster_seqs_list = [[seqs[i] for i in cluster_seqnums_list]]
#         cluster_names_list = [[seq_names[i] for i in cluster_seqnums_list]]
#         group_hstates_list = [np.take(embedding_dict['aa_embeddings'], cluster_seqnums_list, axis = 0)]
#         cluster_seqnums_list = [cluster_seqnums_list] 
#         to_exclude = list(set(to_exclude))
#
#    print("seqnum clusters", cluster_seqnums_list)
#    print(cluster_names_list)
#    return(cluster_seqnums_list, cluster_seqs_list,  cluster_names_list, group_hstates_list, to_exclude)
#
#
