#!/usr/bin/env python3

import torch
from transformers import AutoTokenizer, AutoModel
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import random
from transformer_infrastructure.hf_utils import build_index_flat, build_index_voronoi
from transformer_infrastructure.run_tests import run_tests
from transformer_infrastructure.hf_embed import parse_fasta_for_embed, get_embeddings 
from transformer_infrastructure.hf_seqsim import get_seqsims

# Printing for debugging with icecream
try:
    from icecream import ic
    ic.configureOutput(includeContext=True, outputFunction=print) # Prints line number and function
except ImportError:  # Graceful fallback if IceCream isn't installed.
    ic = lambda *a: None if not a else (a[0] if len(a) == 1 else a)  # noqa

# This is combat with patsy requirement removed
from transformer_infrastructure.combat2 import combat

#import line_profiler
#import atexit
#profile = line_profiler.LineProfiler()
#atexit.register(profile.ic_stats)
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
import shutil
import sys
import igraph
from pandas.core.common import flatten 
import pandas as pd 

from collections import Counter
import matplotlib.pyplot as plt
import logging

from sklearn.metrics.pairwise import cosine_similarity

#class Config:
#    def __init__(self):
#       self.input_fasta = ""
#       self.seqnames = ""
#       self.seqsimthresh = ""
#       self.model_name = ""



class AA:
   
    def __init__(self):
       self.seqnum = ""
       self.seqindex = ""
       self.seqpos = ""
       self.seqaa = ""
       self.index = ""
       self.clustid = ""
       self.prevaa = ""
       self.nextaa = ""
   

   #__str__ and __repr__ are for pretty #icing
   
    def __str__(self):
        return("{}-{}-{}".format(self.seqnum, self.seqpos, self.seqaa))

   
    def __repr__(self):
        return str(self)
 

class Alignment:
    def __init__(self, alignment, seqnames = []):
        self.alignment = alignment
        if not seqnames:
            self.seqnames = list(range(0, len(self.alignment)))    
        else:
            self.seqnames = seqnames
        self.numseqs = len(self.alignment)
        self.width = len(self.alignment[0])
        self.numassigned = len([x for x in flatten(self.alignment) if x != "-"])
        self.numgaps = len([x for x in flatten(self.alignment) if x == "-"])
        self.str_formatted = self.str_format(alignment)


    def str_format(self, alignment):
        str_alignment = []
        for line in alignment:
          row_str = ""
          for aa in line:
                if aa == "-":
                   row_str  = row_str + aa
                else:
                   row_str = row_str + aa.seqaa
          str_alignment.append(row_str)
     
        return(str_alignment)



 
    def format_aln(self, style = "clustal"):
        records = []
        for i in range(len(self.str_formatted)):
             ic(self.str_formatted[i])
             
             alignment_str = "".join([self.str_formatted[i]])
             records.append(SeqRecord(Seq(alignment_str), id=str(self.seqnames[i]),  description = "", name = ""))
        align = MultipleSeqAlignment(records)
        if style == "clustal":

            formatted = format(align, 'clustal')
        elif style == "fasta":
            formatted = format(align, 'fasta')

        return(formatted)

    def __str__(self):
        return(self.format_aln( "clustal"))

    def __repr__(self):
        return str(self)

def get_seqs_aas(seqs, seqnums):
    seqs_aas = []
    seq_to_length = {}
 

    for i in range(len(seqs)):
        
        seq_aas = []
        seqnum = seqnums[i]
        seq_to_length[i] = len(seqs[i])
        for j in range(len(seqs[i])):
           # If first round, start new AA
           # Otherwise, use the next aa as the current aa
           if j == 0: 
               aa = AA()
               aa.seqnum = seqnum
               aa.seqpos = j
               aa.seqaa =  seqs[i][j]
               
            
           else:
               aa = nextaa
               aa.prevaa = prevaa
           prevaa = aa
           if j < len(seqs[i]) - 1:
              nextaa = AA()
              nextaa.seqnum = seqnum
              nextaa.seqpos = j + 1
              nextaa.seqaa = seqs[i][j + 1]
              aa.nextaa = nextaa
         
           
           seq_aas.append(aa)
           
        seqs_aas.append(seq_aas)
    return(seqs_aas, seq_to_length) 

def remove_maxlen_padding(hidden_states, seqs_aas, padded_seqlen):
    # Initial index to remove maxlen padding from input embeddings
    index_to_aa = {}
    aa_indices = []
    for i in range(len(seqs_aas)):
        for j in range(padded_seqlen):
           if j >= seqlens[i]:
             continue 
           
           aa = seqs_aas[i][j]
           index_num = i * padded_seqlen + j
           index_to_aa[index_num] = aa
           aa_indices.append(index_num)         


    # Remove maxlen padding from aa embeddings
    ic(hidden_states.shape)

    hidden_states = np.take(hidden_states, list(index_to_aa.keys()), 0)
    ic(hidden_states.shape)

    index_to_aa = {}
    count_index = 0
    batch_list = []
    seqnum_to_index = {}
    for i in range(len(seqs_aas)):
       seqnum_to_index[i] = []
       for j in range(0, seqlens[i]):
           batch_list.append(i)
           aa = seqs_aas[i][j]
           aa.index = count_index
           aa.seqindex = i
           seqnum_to_index[i].append(count_index)
           index_to_aa[count_index] = aa
           count_index = count_index + 1
    #ic(index_to_aa)           
        
    logging.info("Build index of amino acid embeddings")     
    ic(batch_list)

    return(index_to_aa, hidden_states, seqnum_to_index, batch_list) 

def do_batch_correct(hidden_states, levels, batch_list): 
    hidden_states_pd = pd.DataFrame(hidden_states.T) # So that each aa in a column
    ic(hidden_states_pd)

    batch_series = pd.Series(batch_list)
    #levels = list(range(len(seqs_aas)))
    design_list = [(batch_series == level) * 1 for level in levels]
    design = pd.concat(design_list, axis = 1)
    hidden_states_batch = combat(hidden_states_pd, batch_list, design)
    ic(hidden_states_batch)
    hidden_states_corrected = np.array(hidden_states_batch).T.astype(np.float32) 
    return(hidden_states_corrected)


#@profile
def graph_from_cluster_orders(cluster_orders_lol):

    order_edges = []
    for order in cluster_orders_lol:
       for i in range(len(order) - 1):
          edge = (order[i], order[i + 1])
          #if edge not in order_edges:
          order_edges.append(edge)
          
          #ic(edge)

    G_order = igraph.Graph.TupleList(edges=order_edges, directed=True)
    return(G_order, order_edges)


#@profile
def get_topological_sort(cluster_orders_lol):
    #ic("start topological sort")
    cluster_orders_nonempty = [x for x in cluster_orders_lol if len(x) > 0]
    #with open("tester2.txt", "w") as f:
    #   for x in cluster_orders_nonempty:
    #      f.write("{}\n". format(x))

    
    dag_or_not = graph_from_cluster_orders(cluster_orders_nonempty)[0].simplify().is_dag()
    # 
    

    #ic ("Dag or Not?, dag check immediately before topogical sort", dag_or_not)
  
    G_order = graph_from_cluster_orders(cluster_orders_nonempty)[0]
    G_order = G_order.simplify()

    topo_sort_indices = G_order.topological_sorting()
    cluster_order = []

    # Note: this is in vertex indices. Need to convert to name to get clustid
    for i in topo_sort_indices:
       cluster_order.append(G_order.vs[i]['name'])
    return(cluster_order) #, clustid_to_clust_dag)


#@profile
def remove_order_conflicts(cluster_order, seqs_aas, pos_to_clustid):
   #ic("remove_order_conflicts, before: ", cluster_order)
   bad_clustids = []
   for x in seqs_aas:
      prevpos = -1  
      for posid in x:

          try:
              clustid = pos_to_clustid[posid]
          except Exception as E:
              continue

          pos = posid.seqpos   
          if pos < prevpos:
              #ic("Order violation", posid, clustid)
              bad_clustids.append(clustid)
   cluster_order =  [x for x in cluster_order if x not in bad_clustids]
   return(cluster_order)


#@profile
def candidate_to_remove(G, v_names,z = -5):


    weights = {}  
    num_prots = len(G.vs())
    ic("num_prots")
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
    #ic("Sequence z scores, current threshold: ", z)
    for i in v_names:

        others = []
        for key,value in weights.items():
            if key == i:
                own_value = value
            else:
                others.append(value)  

        #others = [weights[x] for x in range(len(weights)) if x != i]
        ic(own_value, others)
        seq_z = (own_value - np.mean(others))/np.std(others)
        #seq_z = (weights[i] - np.mean(others))/np.std(others)
        ic("sequence ", i, " zscore ", seq_z)

        # This should scale with # of sequences?
        # If on average high similarity, don't call as questionable even if high z
        # Avoid 1.65, 1.72, 1.71 three protein case. 
        #if (own_value / (num_prots - 1)) < 0.7:

        if seq_z < z:
            questionable_z.append(i)
       
    ic("questionalbe_z", questionable_z) 
    return(questionable_z)




#@profile
def make_alignment(cluster_order, seqnums, clustid_to_clust, seqnames):
    # Set up a bunch of vectors of "-"
    # Replace with matches
    # cluster_order = list in the order that clusters go
    ic("Alignment clusters")
    for clustid, clust in clustid_to_clust.items():
        ic(clustid, clust)

    numseqs = len(seqnums)
    alignment_lol =  [["-"] * len(cluster_order) for i in range(numseqs)]
    #ic(cluster_order)
   # #ic("test cluster order", cluster_order)
    for order in range(len(cluster_order)):
       cluster = clustid_to_clust[cluster_order[order]]
       c_dict = {}
       for x in cluster:
           #for pos in x:
           c_dict[x.seqnum]  = x # x.seqaa
       for seqnum_index in range(numseqs):
               try:
                  # convert list index position to actual seqnum
                  seqnum = seqnums[seqnum_index]
                  alignment_lol[seqnum_index][order] = c_dict[seqnum]
               except Exception as E:
                   continue
    #ic a preview)
    alignment_str = ""
    ic("Alignment")

    alignment = Alignment(alignment_lol, seqnames)        
    str_alignment = alignment.str_formatted
    for row_str in str_alignment: 
       ic("Align: ", row_str[0:170])

    return(alignment)

##@profile
#def alignment_ic(alignment, seq_names):
#       
#        records = []
#        #alignment = ["".join(x) for x in alignment]
#        alignment = obj_aln_to_str(alignment)
#              
#        for i in range(len(alignment)):
#             #ic(seq_names[i], alignment[i])
#             #ic(alignment[i], seq_names[i])
#             records.append(SeqRecord(Seq(alignment[i]), id=seq_names[i]))
#        align = MultipleSeqAlignment(records)
#        clustal_form = format(align, 'clustal')
#        fasta_form = format(align, 'fasta')
#        return(clustal_form, fasta_form)



#@profile
def get_ranges(seqs_aas, cluster_order, starting_clustid, ending_clustid, pos_to_clustid):
    #ic("start get ranges")
  
    #ic(cluster_order, starting_clustid, ending_clustid) 
    #ic('get_ranges:', seqs_aas)
    #ic('get_ranges: start, end', starting_clustid, ending_clustid)
    # if not x evaluates to true if x is zero 
    # If unassigned sequence goes to the end of the sequence
    if not ending_clustid and ending_clustid != 0:
       ending_clustid = np.inf    
    # If unassigned sequence extends before the sequence
    if not starting_clustid and starting_clustid != 0:
       starting_clustid = -np.inf   
    #ic('get_ranges: start, end', starting_clustid, ending_clustid)

    # cluster_order must be zero:n
    # Add assertion
    pos_lists = []
    for x in seqs_aas:
            #ic('get_ranges:x:', x)
            pos_list = []
            startfound = False

            #ic("aa", x)
            # If no starting clustid, add sequence until hit ending_clustid
            if starting_clustid == -np.inf:
                 startfound = True
                
            prevclust = "" 
            for pos in x:
                if pos in pos_to_clustid.keys(): 
                    pos_clust = pos_to_clustid[pos]
                    prevclust = pos_clust

                    # Stop looking if clustid after ending clustid
                    if pos_clust >= ending_clustid:
                         break
                    # If the clustid is between start and end, append the position
                    elif pos_clust > starting_clustid and pos_clust < ending_clustid:
                        pos_list.append(pos)
                        startfound = True

                    # If no overlap (total gap) make sure next gap sequence added
                    elif pos_clust == starting_clustid:
                         startfound = True
                        #ic(pos_clust, starting_clustid, ending_clustid)
                else:
                        #ic(startfound, "exception", pos, prevclust, starting_clustid, ending_clustid)
                        if startfound == True or prevclust == cluster_order[-1]:
                           if prevclust:
                               if prevclust >= starting_clustid and prevclust <= ending_clustid:    
                                   pos_list.append(pos)
                           else:
                              pos_list.append(pos)
                         


            pos_lists.append(pos_list)
    return(pos_lists)






#@profile
def get_unassigned_aas(seqs_aas, pos_to_clustid, too_small = []):
    ''' 
    Get amino acids that aren't in a sequence
    '''
    too_small_list = list(flatten(too_small))
    #ic(pos_to_clustid)
    unassigned = []
    for i in range(len(seqs_aas)):
        prevclust = []
        nextclust = []
        unsorted = []
        last_unsorted = -1
        for j in range(len(seqs_aas[i])):
           if j <= last_unsorted:
               continue

           key = seqs_aas[i][j]
  
           if key in pos_to_clustid.keys():
              # Read to first cluster hit
              clust = pos_to_clustid[key]
              prevclust = clust
           # If it's not in a clust, it's unsorted
           else:
              unsorted = []
              unsorted.append(key)
              for k in range(j + 1, len(seqs_aas[i])):
                  key = seqs_aas[i][k]
                  if key in pos_to_clustid.keys():
                     nextclust = pos_to_clustid[key]
                     #ic(nextclust)
                     break
                  # Go until you hit next clust or end of seq
                  else:
               

                     unsorted.append(key)
                     last_unsorted = k

              unsorted = [x for x in unsorted if x not in too_small_list] 
              unassigned.append([prevclust, unsorted, nextclust, i])
              nextclust = []
              prevclust = []
    return(unassigned)








#@profile
def get_looser_scores(aa, index, hidden_states):
     '''Get all scores with a particular amino acid''' 
     hidden_state_aa = np.take(hidden_states, [aa.index], axis = 0)
     # Search the total number of amino acids
     n_aa = hidden_states.shape[0]
     D_aa, I_aa =  index.search(hidden_state_aa, k = n_aa)
     return(list(zip(D_aa.tolist()[0], I_aa.tolist()[0])))


      

#@profile
def get_particular_score(D, I, aa1, aa2):
        ''' Use with squish, replace with get_looser_scores '''

        #ic(aa1, aa2)
        #seqnum different_from index
        #ic(D.shape)
        #ic(aa1.index)
        #ic(aa2.index)
        scores = D[aa1.index][aa1.seqpos][aa2.index]
        #ic(scores)
        ids = I[aa1.index][aa1.seqpos][aa2.index]
        #ic(ids)
        for i in range(len(ids)):
           #ic(aa1, score_aa, scores[i])
           if ids[i] == aa2:
              #ic(aa1, aa2, ids[i], scores[i])
              return(scores[i])
        else:
           return(0) 





#@profile
def address_isolated_aas(unassigned_aa, cohort_aas, D, I, minscore):
    '''
    Maybe overwrite score? 
    Or match to cluster with higher degree
    '''
    #ic("Address isolated aas")
    connections = []
    for cohort_aa in cohort_aas:
        score = get_particular_score(unassigned_aa, cohort_aa, D, I)
        #ic(unassigned_aa, cohort_aa, score)
 
    return(cluster)



#@profile
def clusts_from_alignment(alignment):
   # Pass alignment object around. 
   # Contains both cluster order and clustid_to_clust info
   clustid_to_clust = {}
   
   #align_length = len(alignment[0])

   cluster_order = range(0, alignment.width)
   for i in cluster_order:
       clust = [x[i] for x in alignment.alignment if not x[i] == "-"]

       clustid_to_clust[i] = clust 

   return(cluster_order, clustid_to_clust)



#@profile
def address_stranded3(alignment):
    cluster_order, clustid_to_clust = clusts_from_alignment(alignment)
    to_remove =[]
    #clustered_aas = list(flatten(clustid_to_clust.values())) 
    new_cluster_order = []
    new_clustid_to_clust = {}
    #ic(clustered_aas)
    for i in range(0, len(cluster_order)):

         # If it's the first cluster
         if i == 0:
             prevclust = []

         else:
             prevclustid =cluster_order[i - 1]
             prevclust = clustid_to_clust[prevclustid]  
 
         # If it's the last cluster 
         if i == len(cluster_order) - 1:
             nextclust = []
         else:
             nextclustid =cluster_order[i + 1]
             nextclust = clustid_to_clust[nextclustid]

         currclustid =cluster_order[i]         
         currclust = clustid_to_clust[currclustid]
         removeclust = False
         # DON'T do stranding before bestmatch
         # Because a good column can be sorted into gaps
         for aa in currclust:
             #ic("cluster ", i,  aa.prevaa, aa.nextaa,prevclust,nextclust)
             if aa.prevaa not in prevclust and aa.nextaa not in nextclust:
                  ic("cluster ", i,  aa.prevaa, aa.nextaa,prevclust,nextclust)
                  ic(aa, "in clust", currclust, "is stranded")
                  ic("removing")
                  removeclust = True
         if removeclust == False:
             new_cluster_order.append(currclustid)
             new_clustid_to_clust[currclustid] = currclust
         else:
             ic("Found stranding, Removing stranded clust", currclust) 
    return(new_cluster_order, new_clustid_to_clust)





#@profile
def squish_clusters2(alignment, index, hidden_states, index_to_aa):
    
    '''
    There are cases where adjacent clusters should be one cluster. 
    If any quality scores, squish them together(tetris style)
    XA-X  ->  XAX
    X-AX  ->  XAX
    XA-X  ->  XAX
    Start with doing this at the end
    With checks for unassigned aa's could do earlier
    Get total score between adjacent clusters
    Only record if no conflicts
    Set up network
    Merge highest score out edge fom each cluster
    Repeat a few times

    '''
    ic("attempt squish")
    cluster_order, clustid_to_clust = clusts_from_alignment(alignment)




    candidate_merge_list = []
    for i in range(len(cluster_order)-1):

       c1 = clustid_to_clust[cluster_order[i]]
       # skip cluster that was 2nd half of previous squish
       if len(c1) == 0:
         continue
       c2 = clustid_to_clust[cluster_order[i + 1]]
       c1_seqnums = [x.seqnum for x in c1]
       c2_seqnums = [x.seqnum for x in c2]
       seqnum_overlap = set(c1_seqnums).intersection(set(c2_seqnums))
       
       # Can't merge if two clusters already have same sequences represented
       if len(seqnum_overlap) > 0:
          continue            
       else:
          intra_clust_hits= []
          for aa1 in c1:
            candidates = get_looser_scores(aa1, index, hidden_states)
            for candidate in candidates:
                #try:
                   score = candidate[0]
                   candidate_index = candidate[1]
                   if candidate_index == -1: 
                      continue
                   target_aa = index_to_aa[candidate_index]
                   #ic("target_aa", target_aa)
                   if target_aa in c2:
                       if score > 0:
                           intra_clust_hits.append(score )
                           ic(aa1, target_aa, score)
                #except Exception as E:
                #   # Not all indices correspond to an aa, yes they do
                #   continue
          ic("intra clust hits", intra_clust_hits)
          ic("c1", c1)
          ic("c2", c2)
          combo = c1 + c2
          #scores = [x[2] for x in intra_clust_hits if x is not None]
          candidate_merge_list.append([cluster_order[i], cluster_order[i + 1], sum(intra_clust_hits)])
          ic("candidate merge list", candidate_merge_list) 
 
    removed_clustids = []
    edges = []
    weights = []
    for x in candidate_merge_list:
         edges.append((x[0], x[1]))
         weights.append(x[2])
    to_merge = []
   
    # Repititions deal with particular case 
    # 1-2:0.5  2-3:0.4 3-4:0.3
    # which simplifies to 
    # 1-2:0.5 2-3:0.4
    # (best hit for 2 best hit for 3)    
    for squish in [1,2, 3, 4, 5, 6, 7, 8, 9, 10]:                   
   
        # Start with scores between adjacent clusters
        # Want to merge the higher score when there's a choice
        #ic(edges)        
        G = igraph.Graph.TupleList(edges=edges, directed=False)
        G.es['weight'] = weights
        islands = G.clusters(mode = "weak")
        edges = []
        weights = []
        for sub_G in islands.subgraphs():
            n = len(sub_G.vs())
    
            ic(sub_G)
            #ic(n)
            
            node_highest = {} 
            # If isolated pair, no choice needed
            if n == 2:
                 to_merge.append([x['name'] for x in sub_G.vs()])

            for vertex in sub_G.vs():
               node_highest[vertex['name']] = 0
               if vertex.degree() == 1:
                  continue
               vertex_id= vertex.index
               sub_edges = sub_G.es.select(_source = vertex_id)
               max_weight = max(sub_edges['weight'])
               #ic(max_weight)
    
               maybe = sub_edges.select(weight_eq = max_weight)
    
               ic(vertex)
               for e in maybe:
                  highest_edge = [x['name'] for x in sub_G.vs() if x.index  in e.tuple]
                  ic(highest_edge, max_weight)
                  #if max_weight > node_highest[highest_edge[0]]:
                  #      node_highest[highest_edge[0]] = max_weight
                  if highest_edge not in edges:
                      edges.append(highest_edge)
                      weights.append(max_weight)
    
    
                  #ic(highest_edge)
                  #ic(node_highest)
                  #if highest_edge not in to_merge:
                  #   to_merge.append(highest_edge)
     
    
    ic("to_merge", to_merge)
    
    for c in to_merge: 
              #c = [cluster1, cluster2]
              removed_clustids.append(c[1])
              clustid_to_clust[c[0]] =   clustid_to_clust[c[0]] + clustid_to_clust[c[1]]
              clustid_to_clust[c[1]] = []

    #ic("Old cluster order", cluster_order)
    cluster_order = [x for x in cluster_order if x not in removed_clustids]
        #ifor vs in sub_G.vs():
            
      
    return(cluster_order, clustid_to_clust)

 
 

#@profile
def remove_overlap_with_old_clusters(new_clusters, prior_clusters):
    '''
    Discard any new clusters that contain elements of old clusters
    Only modify new clusters in best match-to-cluster process
    '''
    
    aas_in_prior_clusters = list(flatten(prior_clusters))
    #ic("aas in prior", aas_in_prior_clusters)
      
    final_new_clusters = []
    for n in new_clusters:
        #for p in prior_clusters:
        overlap =  list(set(aas_in_prior_clusters).intersection(set(n)))
        if len(overlap) > 0:
             #ic("prior", p)
             #ic("new with overlap of old ", n)
             continue
        elif n in final_new_clusters:
             continue
        else:
             final_new_clusters.append(n)

    return(final_new_clusters)
    
 

      



#@profile
def remove_feedback_edges(cluster_orders_dict, clustid_to_clust, gapfilling_attempt, remove_both = True, alignment_group = 0, attempt = 0, all_alternates_dict = {}, args = None):

    """
    Remove both improves quality of initial alignment by remove both aas that are found out of order
    For final refinement, only remove the first one that occurs out of order

    """
    ic("argssss", args)
    record_dir = args.record_dir
    outfile_name = args.outfile_name
 
    ic("before feedback_edges")
    ic(clustid_to_clust)
    G_order, order_edges = graph_from_cluster_orders(list(cluster_orders_dict.values()))

    #ic(G_order)
    weights = [1] * len(G_order.es)



    # Remove multiedges and self loops
    #ic(G_order)
    G_order.es['weight'] = weights
    G_order = G_order.simplify(combine_edges=sum)




    ic("after combine")
    #ic(G_order)
    dag_or_not = G_order.is_dag()



    # The edges to remove to make a directed acyclical graph
    # Corresponds to "look backs"
    # With weight, fas, with try to remove lighter edges
    # Feedback arc sets are edges that point backward in directed graph
    
    fas = G_order.feedback_arc_set(weights = 'weight')

  
    ic("feedback arc set")
    for x in fas:
        ic("arc", x) 

    write_ordernet = True
    if write_ordernet == True:
 
       outnet = "{}/{}.ordernet_{}_attempt-{}_gapfilling-{:04}.csv".format(record_dir, outfile_name, alignment_group, attempt, gapfilling_attempt)
       ic("outnet", outnet, gapfilling_attempt)
       with open(outnet, "w") as outfile:
          outfile.write("c1,c2,aas1,aas2,gidx1,gidx2,weight,feedback\n")
          # If do reverse first, don't have to do second resort
          for edge in G_order.es():
             feedback = "no"
             if edge.index in fas:
                feedback = "yes"
             source_name = G_order.vs[edge.source]["name"]
             target_name = G_order.vs[edge.target]["name"]
             source_aas = "_".join([str(x) for x in clustid_to_clust[source_name]])
             target_aas = "_".join([str(x) for x in clustid_to_clust[target_name]])
             outstring = "{},{},{},{},{},{},{},{}\n".format(source_name, target_name, source_aas, target_aas , edge.source, edge.target, edge['weight'], feedback)        
             outfile.write(outstring)

    #i = 0
    to_remove = []
    removed_edges = []
    for feedback_arc in fas:
       edge = G_order.es()[feedback_arc]
       source_name = G_order.vs[edge.source]["name"]
       target_name = G_order.vs[edge.target]["name"]
     
       ic("Feedback edge {}, index {}, edge.source {} edge.target {} source_name {}, target_name{}" .format(edge, edge.index, edge.source, edge.target, source_name, target_name))
       removed_edges.append((edge.source, edge.target, source_name, target_name))

    # Delete feed back arc edges
    G_order.delete_edges(fas)
 
    # Check if graph is still dag if edge is added back.
    # If so, keep it
    for removed_edge in removed_edges:
         ic("try to return", removed_edge[2:])
         G_order.add_edges(  [removed_edge[0:2]]) # vertex id pairs 
         #G_order.add_edges([removed_edge[2:]]) # vertex id pairs 
         ic(G_order.is_dag())
         # Retain edges that aren't actually feedback loops
         # Some edges identified by feedback_arc aren't actually cycles (???)
         if not G_order.is_dag():
             G_order.delete_edges([removed_edge[0:2]])        
             #G_order.delete_edges([removed_edge[2:]])        

             # try alternate cluster conformations
                          
             to_remove.append(removed_edge[2:4]) # list of clustid pairs 


    #ic(G_order)

    ic("to_remove", to_remove)   
    remove_dict = {}
   
    #ic("cluster_orders_dict", cluster_orders_dict)
    if remove_both == True: 
        to_remove_flat = list(flatten(to_remove))
    else:
        to_remove_flat = [x[0] for x in to_remove]
    #ic("to_remove 2", to_remove_flat)
      
    

    clusters_to_add_back = {} # Dictionary of list of lists containing pairs of clusters to add back with modifications 
    #group_count = 0
    for seqnum, clustorder in cluster_orders_dict.items():
      remove_dict[seqnum] = []
      remove = []
      if len(clustorder) == 1:
          if clustorder[0] in to_remove_flat: 
              remove_dict[seqnum] = [clustorder[0]]
      #ic(clustorder)
      
      for j in range(len(clustorder) - 1):
          
           new_clusts_i = []
           new_clusts_j = []
           if (clustorder[j], clustorder[j +1]) in to_remove:
               clust_i = clustid_to_clust[clustorder[j]]
               clust_j = clustid_to_clust[clustorder[j + 1]]
               clusters_to_add_back_list = []
               for aa in clust_i:
                    if aa in all_alternates_dict.keys():
                        for alternate in all_alternates_dict[aa]:
                            ic("replacing {} with {}".format(aa, alternate))
                            new_clust_i = [x for x in clust_i if x != aa] + [alternate] 
                            new_clusts_i.append(new_clust_i)
                               #clusters_to_add_back.append([new_clust_i, clust_j])
               for aa in clust_j:
                    if aa in all_alternates_dict.keys():
                        for alternate in all_alternates_dict[aa]:
                            ic("replacing {} with {}".format(aa, alternate))
                            new_clust_j = [x for x in clust_j if x != aa] + [alternate] 
                            new_clusts_j.append(new_clust_j)
               
               for new_clust_i in new_clusts_i:
                    clusters_to_add_back_list.append([new_clust_i, clust_j])
                    for new_clust_j in new_clusts_j:
                           clusters_to_add_back_list.append([clust_i, new_clust_j])
                           clusters_to_add_back_list.append([new_clust_i, new_clust_j])
               clusters_to_add_back[frozenset([j, j + 1])] = clusters_to_add_back_list

               #ic(cluster_orders[i])
               #ic(remove_both) 
               #ic(cluster_orders[i][j], cluster_orders[i][j + 1])
               if remove_both == True:
                   remove.append(clustorder[j])
               remove.append(clustorder[j + 1])
           remove_dict[seqnum] = list(set(remove))
           
    ic("remove_dict", remove_dict)
    #clusters_filt_dag = []
    #ic(clusters_filt)
    ic("Doing remove")
    reduced_clusters = []
    #too_small_clusters = []
    removed_clustids = list(flatten(list(remove_dict.values())))
    
    removed_clusters = []
    ic("removed clusters", removed_clustids)
   
    for clustid, clust in clustid_to_clust.items():
          new_clust = []
          if clustid in removed_clustids:
                ic("Can remove", clustid, clust)
                for aa in clust:
                    if aa in all_alternates_dict.keys():
                        ic("Alternate found for AA ",aa, all_alternates_dict[aa])
                         # STOPPED HERE
                         # MARK
                         # ADD list of lists of both clustids in edge to try with alternates
          else:
              reduced_clusters.append(clust)
          #else:
          #    too_small_clusters.append(new_clust)

    #ic("minclustsize", minclustsize)
    ic("All alternates_dict", all_alternates_dict) 
    ic("reduced clusters", reduced_clusters)
    #ic("too small clusters" too_small_clusters)

    return(reduced_clusters, clusters_to_add_back)


#@profile
def remove_streakbreakers(hitlist, seqs_aas, seqnums, seqlens, streakmin = 3):
    # Not in use
    # Remove initial RBHs that cross a streak of matches
    # Simplify network for feedback search
    filtered_hitlist = []
    for i in range(len(seqs_aas)):
       seqnum_i = seqnums[i]
       query_prot = [x for x in hitlist if x[0].seqnum == seqnum_i]
       for j in range(len(seqs_aas)):
          seqnum_j = seqnums[j]
          target_prot = [x for x in query_prot if x[1].seqnum == seqnum_j]
         
          # check shy this is happening extra at ends of sequence
          #ic("remove lookbehinds")
          prevmatch = 0
          seq_start = -1
          streak = 0

          no_lookbehinds = []
          for match_state in target_prot:
               #ic(match_state)
               if match_state[1].seqpos <= seq_start:
                     #ic("lookbehind prevented")
                     streak = 0 
                     continue
               no_lookbehinds.append(match_state)

               if match_state[1].seqpos - prevmatch == 1:
                  streak = streak + 1
                  if streak >= streakmin:  
                     seq_start = match_state[1].seqpos
               else:
                  streak = 0
               prevmatch  = match_state[1].seqpos

          #ic("remove lookaheads")
          prevmatch = seqlens[j]
          seq_end = seqlens[j]
          streak = 0

          filtered_target_prot = []
          for match_state in no_lookbehinds[::-1]:
               #ic(match_state, streak, prevmatch)
               if match_state[1].seqpos >= seq_end:
                    #ic("lookahead prevented")
                    streak = 0
                    continue
               filtered_target_prot.append(match_state)
               if prevmatch - match_state[1].seqpos == 1:
                  streak = streak + 1
                  if streak >= streakmin:  
                     seq_end = match_state[1].seqpos
               else:
                  streak = 0
               prevmatch = match_state[1].seqpos
 
          filtered_hitlist = filtered_hitlist + filtered_target_prot
    return(filtered_hitlist) 


#@profile
def get_doubled_seqnums(cluster):
      seqnums = [x.seqnum for x in cluster]

            
      clustcounts = Counter(seqnums)
            #ic(clustcounts)
      to_remove = []
      for key, value in clustcounts.items():
           if value > 1:
               to_remove.append(key)

      return(to_remove)

#@profile
def remove_doubles_by_consistency(cluster, pos_to_clustid, add_back = True):      
    '''
    Keep any doubled amino acids that pass the consistency check  based on previous and next cluster
    Option to keep both if neither are consistent, to send on to further remove_doubles attempts
    '''

    to_remove = get_doubled_seqnums(cluster)
    if len(to_remove) == 0:
          return(cluster)

    cluster_minus_targets = [x for x in cluster if x.seqnum not in to_remove]
    # Can be more than one doubled seqnum per cluster
    # Remove all doubles, then add back in any that are consistent
    to_add_back = []
    for seqnum in to_remove:
        target_aas = [x for x in cluster if x.seqnum == seqnum]
        consistent_ones = []
        for aa in target_aas:
            if consistency_check ( [aa] + cluster_minus_targets, pos_to_clustid ) == True:
                consistent_ones.append(aa)
        # Add back in consistent to cluster     
        if len(consistent_ones) == 1:
            cluster_minus_targets = cluster_minus_targets + [consistent_ones[0]] 
        else:
           if add_back == True:
              to_add_back = to_add_back + target_aas

    if add_back == True:
         ic("Adding back", to_add_back, "to", cluster_minus_targets)
         cluster_minus_targets = cluster_minus_targets + to_add_back

 


    return(cluster_minus_targets)


#@profile
def consistency_check(cluster, pos_to_clustid):
    '''
    For a cluster, see if previous or next amino acids are also all part os same cluster
    '''


    prev_list = []
    next_list = []
    for aa in cluster:
        if aa.prevaa in pos_to_clustid.keys():
            prev_list.append(pos_to_clustid[aa.prevaa])
        if aa.nextaa in pos_to_clustid.keys():
            next_list.append(pos_to_clustid[aa.nextaa])
 
    prevset = list(set(prev_list))
    nextset = list(set(next_list))
    if len(prevset) == 1 or len(nextset) == 1:
        return(True)
 
    else:
        return(False)



#@profile
def remove_doubles_by_scores(clust, index, hidden_states, index_to_aa):

    alternates_dict = {}
    doubled_seqnums = get_doubled_seqnums(clust)
    if doubled_seqnums:
         clust_minus_dub_seqs = [x for x in clust if x.seqnum not in doubled_seqnums] 
         #ic("sequence {} in {}, {} is doubled".format(doubled_seqnums, clustnum, clust))
         for seqnum in doubled_seqnums:
             saved = None
             bestscore = 0
             double_aas = [x for x in clust if x.seqnum == seqnum]     
             #ic(double_aas)
             for aa in double_aas:
                 candidates_w_score = get_set_of_scores(aa, index, hidden_states, index_to_aa)
                 incluster_scores = [x for x in candidates_w_score if x[0] in clust_minus_dub_seqs ]
                 total_incluster_score = sum([x[1] for x in incluster_scores])
                 #ic(total_incluster_score)
                  #ic(incluster_scores)
                 if total_incluster_score > bestscore:
                     saved = aa
                     bestscore = total_incluster_score
             #ic("Adding back {} to {}".format(keeper, clust_minus_dub_seqs))
             if saved:
                 alts = [x for x in double_aas if x != saved]
                 clust_minus_dub_seqs = clust_minus_dub_seqs + [saved]
                 alternates_dict[saved] = alts
#
         return(clust_minus_dub_seqs, alternates_dict)
    else:
         return(clust, alternates_dict) 

#
#
#    return(clustid_to_clust)



#@profile
def remove_doubles_by_graph(cluster, G,  minclustsize = 0, keep_higher_degree = False, keep_higher_score = True, remove_both = False):
            ''' If a cluster contains more 1 amino acid from the same sequence, remove that sequence from cluster'''
           
      
            '''
            If 
            '''
            alternates_dict = {}
            ic("remove_doubles_by_graph")
            ic(cluster)
            to_remove = get_doubled_seqnums(cluster)
            #if len(to_remove) > 0 and check_order_consistency == True:
            # 
            #    cluster = remove_doubles_by_consistency(cluster, pos_to_clustid)
            #    #if new_cluster != cluster:
            #    finished = check_completeness(new_cluster)
            #    if finished == True:
            #         return(cluster)
            #    to_remove = get_doubled_seqnums(cluster)

            #ic(cluster)
            ic(keep_higher_degree, keep_higher_score, to_remove)
            # If there's anything in to_remove, keep the one with highest degree
              

            if len(to_remove) > 0 and keep_higher_score == True:

                 G = G.vs.select(name_in=cluster).subgraph()
                 #ic(G)
                 #rbh_sel = [x for x in rbh_list if x[0] in cluster and x[1] in cluster]
                 #G = igraph.Graph.TupleList(edges=rbh_sel, directed = False)
                 #G = G.simplify() 
                 #ic("edges in cluster", rbh_sel)
                 for seqnum in to_remove:
                     cluster, saved, alts = remove_lower_score(cluster, seqnum, G)
                     alternates_dict[saved] = alts
                 to_remove = get_doubled_seqnums(cluster)
            if len(to_remove) > 0 and keep_higher_degree == True:
            
                 G = G.vs.select(name_in=cluster).subgraph()
                 #ic(G)
                 #rbh_sel = [x for x in rbh_list if x[0] in cluster and x[1] in cluster]
                 #G = igraph.Graph.TupleList(edges=rbh_sel, directed = False)
                 #G = G.simplify() 
                 #ic("edges in cluster", rbh_sel)
                 for seqnum in to_remove:
                    cluster, saved, alts = remove_lower_degree(cluster, seqnum, G)
                    alternates_dict[saved] = alts
  
                 to_remove = get_doubled_seqnums(cluster)
            # Otherwise, remove any aa's from to_remove sequence
            if len(to_remove) > 0 and remove_both == True:
                for x in to_remove:
                   ic("Removing sequence {} from cluster".format(x))
                   #ic(cluster)
                   #ic(seqnums)
                   #ic(clustcounts) 

                cluster = [x for x in cluster if x.seqnum not in to_remove]
            if len(cluster) < minclustsize:
               return([], {})
            else:
                return(cluster, alternates_dict)

#
##@profile
#def resolve_conflicting_clusters(clusters)


#@profile
def remove_lower_score(cluster, seqnum, G):

    target_aas = [x for x in cluster if x.seqnum == seqnum]
            #ic(aas)       
    degrees = []
    edge_sums = {}
    ic(target_aas)
    #ic(G)
    ic(G.vs()['name'])

    aa_idxs = [G.vs.find(name =x) for x in target_aas]
    for aa in target_aas:
         g_new = G.copy()
         query_vs = g_new.vs.find(name = aa)
         target_vs = [x for x in g_new.vs() if x not in aa_idxs]
         #ic("aa_idxs", aa_idxs) 
         #ic("target_vs", target_vs) 
         #ic("query_vs", query_vs) 
         edges = g_new.es.select(_source=query_vs)#   ['weight'])
         edge_sums[aa] = sum(edges['weight'])
    ic(edge_sums)
    ic("dupped aas", target_aas)
             
    highest_score = max(edge_sums, key=edge_sums.get)
    saved = highest_score
    ic("high score", highest_score)
    to_remove = [x for x in target_aas if x != highest_score]
    cluster_filt = [x for x in cluster if x not in to_remove]
    ic("cluster", cluster)
    ic("cluster_filt", cluster_filt)
    alts= to_remove  # Save these for later (if the saved on causes a feedback arc, try the alts)
    return(cluster_filt, saved, alts)
 

#@profile
def remove_lower_degree(cluster, seqnum, G):

    target_aas = [x for x in cluster if x.seqnum == seqnum]
            #ic(aas)       
    degrees = []
    for aa in target_aas:

         degrees.append(G.vs.find(name  = aa).degree())
         # This doesn't 
         #degrees.append(G.degree( aa))
                  # TODO: Get rbh to return scores
                  # get highest score if degree tie
                  # gap_scores.append(G
    ic("dupped aas", target_aas)
             
    highest_degree = target_aas[np.argmax(degrees)]
    ic("high degree", highest_degree)
    to_remove = [x for x in target_aas if x != highest_degree]
    cluster_filt = [x for x in cluster if x not in to_remove]
    ic("cluster", cluster)
    ic("cluster_filt", cluster_filt)
    return(cluster_filt)
   


#@profile
def graph_from_rbh(rbh_list, directed = False):

    weights = [x[2] for x in rbh_list]
    G = igraph.Graph.TupleList(edges=rbh_list, directed = directed)
    G.es['weight'] = weights 
    G = G.simplify(combine_edges = "first")
    return(G)


#@profile
def remove_low_match_prots(numseqs, seqlens, clusters, threshold_min = 0.5): 
    ############## No badly aligning sequences check
    # Remove sequences that have a low proportion of matches from cluster
    # Do another one of these after ordering criteria
    matched_count =  [0] * numseqs
    for pos in flatten(clusters):
        seqnum = get_seqnum(pos)
        matched_count[seqnum] = matched_count[seqnum] + 1
        

    matched_prop = [matched_count[x]/seqlens[x] for x in range(0, numseqs)] 
    poor_seqs = []
    for i in range(0, numseqs):
        if matched_prop[i] < threshold_min:
            #ic("Seq {} is poorly matching, fraction positions matched {}, removing until later".format(i, matched_prop[i]))
            poor_seqs.append(i)

    clusters_tmp = []
    for clust in clusters:
       clust_tmp = []
       for pos in clust:
            if not get_seqnum(pos) in poor_seqs:
               clust_tmp.append(pos)
       clusters_tmp.append(clust_tmp)
 
    clusters  = clusters_tmp
    return(clusters)

 

#@profile
def reshape_flat(hstates_list):

    # Go from (numseqs, seqlen, emb) to (numseqs * seqlen, emb)
    hidden_states = np.reshape(hstates_list, (hstates_list.shape[0]*hstates_list.shape[1], hstates_list.shape[2]))
    return(hidden_states)




#@profile
def split_distances_to_sequence2(D, I, index_to_aa, numseqs, seqlens):
   #{queryaa:{seqindex1:[targetaa1:score1,targetaa2:score2], seqindex2:[targetaa2:score2]} 

   # MAKE this

   ic(index_to_aa)
   query_aa_dict = {}
   for i in range(len(I)):
      #ic(I[i].shape)
      query_aa = index_to_aa[i]
      # Make dictionary, one per sequence
      target_dict = {}
      #for seqindex in seqindexes:
      #    target_dict[seqindex] = []

      for k in range(numseqs):
          target_dict[k] = []
          #target_D_dict[k] = []

      #ic(query_aa, i, D[i]) 
      #ic(query_aa, i, I[i]) 
      for j in range(len(I[i])):
           try:
              target_aa = index_to_aa[I[i][j]] 
           except Exception as E:
               #ic("no aa at",  I[i][j])               
               continue
           seqindex = target_aa.seqindex
           target_dict[seqindex].append([target_aa, D[i][j]]) 
           #ic("repeated", query_aa, target_aa,i,j, D[i][j])

      query_aa_dict[query_aa] = target_dict
      #query_aa_D_dict[query_aa] = target_D_dict
   return(query_aa_dict)
     

      
#
##@profile
#def old():
#      I_tmp.append(query_aa_I_dict)
#      D_tmp.append(query_aa_D_dict)
#      #X_tmp.append(X_query)
#     
#   #ic(padded_seqlen)
#   
#   #ic("X_tmp", X_tmp)
#   #for x in X_tmp:
#   #   ic("X_tmp X", x)
#   #ic(len(X_tmp))
#
#   #I = []
#   #D = []
#   #X = []
#
#   listbreak = 0
#   slices = []
#   ic(seqlens)
#   for seqlen in seqlens:
#      slices.append([listbreak, listbreak + seqlen])
#
#      #I.append(I_tmp[listbreak: listbreak + seqlen])
#      #D.append(D_tmp[listbreak: listbreak + seqlen])
#      #X.append(X_tmp[listbreak: listbreak + seqlen])
#      listbreak = seqlen + listbreak
#
#   ic(slices)
#
#   D = {}
#   I = {}
#   for i in range(len(seqlens)):
#      x = slices[i]
#      D[i] = D_tmp[x[0]:x[1]]
#      I[i] = I_tmp[x[0]:x[1]]
#
#
#   #D =  [D_tmp[x[0]:x[1]] for x in slices]
#   #I =  [I_tmp[x[0]:x[1]] for x in slices]
#
#   #D =  [D_tmp[i:i + padded_seqlen] for i in range(0, len(D_tmp), padded_seqlen)]
#   #I =  [I_tmp[i:i + padded_seqlen] for i in range(0, len(I_tmp), padded_seqlen)]
#
#  
#   return(D, I)
#

#@profile
def get_besthits(I,  minscore = 0.1 ):

   #aa_to_index = {value: key for key, value in index_to_aa.items()}

   hitlist = []
   for aa in I.keys():
      
      for targetseq in I[aa].keys():
          if len(I[aa][targetseq]) > 0 :
              # Top score is always first
              besthit = I[aa][targetseq][0]
              besthit_aa = besthit[0] # AA
              besthit_score = besthit[1] #score to query aa
              # Avoid tie situations
              # Save ambiguous best hits until later
              if len(I[aa][targetseq]) > 1:
                  next_besthit = I[aa][targetseq][1]
                  next_besthit_aa = next_besthit[0] # AA
                  next_besthit_score = next_besthit[1] #score to query aa
    
                  #if besthit_score - next_besthit_score <= 0.002:
                  #    ic("{} has tie between {}:{} and {}:{}".format(aa, besthit_aa, besthit_score, next_besthit_aa, next_besthit_score))
                  #    continue
              #ic(I[aa][targetseq])
              #ic(aa, besthit_aa, besthit_score)
              if besthit_score >= minscore:
                  hitlist.append([aa, besthit_aa, besthit_score])

   return(hitlist) 


#@profile
def get_rbhs(hitlist_top, min_edges = 0):
    '''
    [aa1, aa2, score (higher = better]
    '''

    G_hitlist = igraph.Graph.TupleList(edges=hitlist_top, directed=True) 
    weights = [x[2] for x in hitlist_top]

    rbh_bool = G_hitlist.is_mutual()
    
    hitlist = []
    G_hitlist.es['weight'] = weights 

    G_hitlist.es.select(_is_mutual=False).delete()
    G_hitlist.vs.select(_degree=0).delete()

    

    # What?
    #if min_edges > 0:
        #edge_sel = 2 * min_edges
    #    G_hitlist.vs.select(_degree=min_edges).delete()    

    sources = [G_hitlist.vs[x.source]['name'] for x in G_hitlist.es()]
    targets = [G_hitlist.vs[x.target]['name'] for x in G_hitlist.es()]
    weights = G_hitlist.es()['weight']
 
    ic("len check",  len(sources), len(targets),len(weights))
    hitlist = list(zip(sources,targets, weights))

    return(hitlist)



#@profile
def clustering_to_clusterlist(G, clustering):
    """
    Go from an igraph clustering to a list of clusters [[a,b,c], [d,e]]
    """

    
    # Replace below with this, don't need G as argument             
    #for sub_G in clustering.subgraphs():
    #       connected_set = sub_G.vs()['name']
    #       clusters_list.append(connected_set)


    cluster_ids = clustering.membership
    vertices = G.vs()["name"]
    clusters = list(zip(cluster_ids, vertices))

    clusters_list = []
    for i in range(len(clustering)):
         clusters_list.append([vertices[x] for x in clustering[i]])


    return(clusters_list)


#@profile
def remove_highbetweenness(G, betweenness_cutoff = 0.10):
            n = len(G.vs())
            if n <= 5:
               return(G)
            bet = G.betweenness(directed=True) # experiment with cutoff based on n for speed
            bet_norm = []
 
           #get_new_clustering(G, betweenness_cutoff = 0.15,  apply_walktrap = True)
            correction = ((n - 1) * (n - 2)) / 2
            for x in bet:
                x_norm = x / correction
                #if x_norm > 0.45:
                bet_norm.append(x_norm)
            
                #bet_dict[sub_G.vs["name"]] = norm
            G.vs()['bet_norm'] = bet_norm          
           # #ic("before", sub_G.vs()['name'])
 
            bet_names = list(zip(G.vs()['name'], bet_norm))
            # A node with bet_norm 0.5 is perfectly split between two clusters
            # Only select nodes with normalized betweenness before 0.45
            pruned_vs = G.vs.select([v for v, b in enumerate(bet_norm) if b < betweenness_cutoff]) 
                
            new_G = G.subgraph(pruned_vs)
            return(new_G) 

# Only start with natural, sequential clusters

#@profile
def consistency_clustering(G, minclustsize = 0, dup_thresh = 1):
    '''
    First, naturally consistent
    Second, cluster members prev or next aas fall in same cluster. 

    '''
    # Get naturally disconnected sets
    islands = G.clusters(mode = "weak")
    natural_cluster_list = []
    cluster_list = []
    for sub_G in islands.subgraphs():
        natural_cluster = sub_G.vs()['name']
        ic("Natural connected set", sub_G.vs()['name'])
        min_dupped =  min_dup(natural_cluster, dup_thresh)
        ic(min_dupped, minclustsize)
        if(len(natural_cluster) <= min_dupped):
            if(len(natural_cluster) >= minclustsize):
                 natural_cluster_list.append(natural_cluster)

    pos_to_clustid, clustid_to_clust = get_cluster_dict(natural_cluster_list)
    for natural_cluster in natural_cluster_list:

          # Need to check if duplicated here first
          ic("Checking", natural_cluster)
          if consistency_check(natural_cluster, pos_to_clustid) == True:
              finished = check_completeness(natural_cluster)
              if finished == True:
                  cluster_list.append(natural_cluster)
          else:
             ic("Check if duplicated")
             ic("If duplicated, see if removing one of the aas makes consistent")
             seqnums = [x.seqnum for x in natural_cluster]
             if len(seqnums) < len(natural_cluster):
                  new_cluster = remove_doubles_by_consistency(natural_cluster, pos_to_clustid, add_back = True)
                  finished = check_completeness(new_cluster)
                  if finished == True:
                     cluster_list.append(new_cluster)

    for x in cluster_list:
          ic("natural_cluster", x)

    return(cluster_list)










#@profile
def first_clustering(G,  betweenness_cutoff = .10, ignore_betweenness = False, apply_walktrap = True):
    '''
    Get betweenness centrality
    Each node's betweenness is normalized by dividing by the number of edges that exclude that node. 
    n = number of nodes in disconnected subgraph
    correction = = ((n - 1) * (n - 2)) / 2 
    norm_betweenness = betweenness / correction 
    '''

    #G = igraph.Graph.TupleList(edges=rbh_list, directed=False)
    ic("Start first clustering")


    # Islands in the RBH graph
    
    islands = G.clusters(mode = "weak")
    new_subgraphs = []
    cluster_list = []
    hb_list = []
    all_alternates_dict = {}

    # For each island, evaluate what to do
    for sub_G in islands.subgraphs():
        # Don't remove edges if G = size 2
        if len(sub_G.vs()) < 4:
               betweenness_cutoff = 1
        else:
           betweenness_cutoff = betweenness_cutoff
        #ic("First connected set", sub_G.vs()['name'], apply_walktrap)
        
        # First start with only remove very HB nodes
        new_G = remove_highbetweenness(sub_G, betweenness_cutoff = betweenness_cutoff)
        sub_islands = new_G.clusters(mode = "weak")
        for sub_sub_G in sub_islands.subgraphs():

            # Should do a betweenness here again
            ic("first_clustering: betweenness cutoff", betweenness_cutoff, "apply_walktrap", apply_walktrap)
            new_clusters, alternates_dict = get_new_clustering(sub_sub_G, betweenness_cutoff = betweenness_cutoff,  apply_walktrap = apply_walktrap) 
            if alternates_dict:
                 all_alternates_dict = {**all_alternates_dict, **alternates_dict}

            ic("first_clustering:adding", new_clusters, "to cluster_list")
            cluster_list = cluster_list + new_clusters

    ic("cluster_list at end of first_clustering", cluster_list)
    ic("Alternates at end of first clustering", all_alternates_dict)
    return(cluster_list, all_alternates_dict)



#@profile
def get_new_clustering(G, betweenness_cutoff = 0.10,  apply_walktrap = True):
    ic("get_new_clusteri                 all_alternates_dict = {**all_alternates_dict, **alternates_dict}ng")
    new_clusters = []
    connected_set = G.vs()['name']
   # #ic("after ", sub_connected_set)
    #ic("connected set", connected_set)
    all_alternates_dict = {}
    new_clusters = []
 
    finished = check_completeness(connected_set)
    # Complete if no duplicates
    if finished == True:
        ic("finished connected set", connected_set)
        new_clusters = [connected_set]

    else:
        min_dupped =  min_dup(connected_set, 1.2)
        # Only do walktrap is cluster is overlarge
        ic("min_dupped at start", min_dupped)      
        names = [x['name'] for x in G.vs()]
        ic("names at start", names)
        if (len(connected_set) > min_dupped) and apply_walktrap and len(G.vs()) >= 5:
            # First remove weakly linked aa's then try again
            # Walktrap is last resort
            ic("similarity_jaccard, authority_score")
            hub_scores = G.hub_score()
            names = [x['name'] for x in G.vs()]
            ic(names)
            vx_names = G.vs()
            hub_names = list(zip(names, vx_names, hub_scores))
            
            #high_authority_nodes = [x[0] for x in hub_names if x[2]  > 0.2]
            #ic("high authority_nodes", high_authority_nodes)
            high_authority_nodes_vx = [x[1] for x in hub_names if x[2]  > 0.2]

            low_authority_nodes = [x[0] for x in hub_names if x[2]  <= 0.2]
            ic("removing low authority_nodes", low_authority_nodes)
            low_authority_nodes = []
            if len(low_authority_nodes) > 0:
                
                G = G.subgraph(high_authority_nodes_vx)
                names = [x['name'] for x in G.vs()]
                ic("names prior to new clusters", names)

                min_dupped = min_dup(names, 1.2) 
            if len(names) <= min_dupped:           
                ic("get_new_clustering:new_G", G)
                processed_cluster, alternates_dict =  process_connected_set(names, G, dup_thresh = 1.2, betweenness_cutoff = betweenness_cutoff)
                ic("processed_cluster", processed_cluster)
                #ic("alternates_dict", alternates_dict)
                if alternates_dict:
                   all_alternates_dict = {**all_alternates_dict, **alternates_dict}                   
                new_clusters = new_clusters + processed_cluster
            else:
                ic("applying walktrap")
                # Change these steps to 3??
                # steps = 1 makes clear errors
                ic("len(connected_set, min_duppled", len(connected_set), min_dupped) 
                clustering = G.community_walktrap(steps = 3, weights = 'weight').as_clustering()
                #i = 0
                for sub_G in clustering.subgraphs():
                     sub_connected_set =  sub_G.vs()['name']
                     ic("post cluster subgraph", sub_connected_set)
                     
                     # New clusters may be too large still, try division process w/ betweenness
                     
                     processed_cluster,alternates_dict = process_connected_set(sub_connected_set, sub_G, dup_thresh = 1.2, betweenness_cutoff = betweenness_cutoff) 
                     new_clusters = new_clusters + processed_cluster 
                     if alternates_dict:
                        all_alternates_dict = {**all_alternates_dict, **alternates_dict}                   
        else:
            ic("get_new_clustering:connected_set", connected_set)
            processed_cluster, alternates_dict =  process_connected_set(connected_set, G, dup_thresh = 1.2, betweenness_cutoff = betweenness_cutoff)
            new_clusters = new_clusters + processed_cluster 
            if alternates_dict:
                 all_alternates_dict = {**all_alternates_dict, **alternates_dict}                   
       
    ic("get_new_clustering:all_alternates_dict", all_alternates_dict)
    return(new_clusters, all_alternates_dict)



#@profile
def get_represented_seqs(connected_set):
    represented_seqs = list(set([x.seqnum for x in connected_set]))
    return(represented_seqs)



#@profile
def min_dup(connected_set, dup_thresh):
    represented_seqs = get_represented_seqs(connected_set)
    #ic("tot_represented_seqs", tot_represented_seqs)
    #ic(len(connected_set), len(tot_represented_seqs))
    return(dup_thresh *  len(represented_seqs))

 

#@profile
def process_connected_set(connected_set, G, dup_thresh = 1.2,  betweenness_cutoff = 0.10):
    ''' 
    This will take a connected sets and 
    1) Check if it's very duplicated
    2) If so, remove high betweenness nodes, and check for completeness again
    3) If it's not very duplicated, removed any duplicates by best score
    '''

    ic("process_connected_set: betweenness", betweenness_cutoff, "dup_thresh", dup_thresh)
    new_clusters = []
    all_alternates_dict = {}
    min_dupped =  min_dup(connected_set, dup_thresh)
    if len(connected_set) > min_dupped:
        #ic("cluster too big", connected_set)

        # TRY removing high betweenness and evaluating completeness
        ic("Check for betweenness")
       
        new_G = remove_highbetweenness(G, betweenness_cutoff = 0.10)
        ic("prebet", G.vs()['name'])
        ic("postbet", new_G.vs()['name'])
        #if len(new_Gs) > 1:
        new_islands = new_G.clusters(mode = "weak")
        for sub_G in new_islands.subgraphs():
                alternates_dict = {}
                sub_connected_set = sub_G.vs()['name']
                ic("postbet_island", sub_connected_set)
                sub_min_dupped =  min_dup(sub_connected_set, dup_thresh) 
                
                ic("sub_min_dupped", sub_min_dupped)
                # Actually going to keep all clusters below min dup thresh
                if (len(sub_connected_set) <= sub_min_dupped) or (len(sub_connected_set) <= 5):
                    trimmed_sub_connected_set, alternates_dict = remove_doubles_by_graph(sub_connected_set, sub_G) 
                    #new_clusters.append(trimmed_sub_connected_set)

                else:
                    ic("still about min_dupped applying walktrap")
                    new_walktrap_clusters, alternates_dict = get_new_clustering(sub_G, betweenness_cutoff = betweenness_cutoff,  apply_walktrap = True)
                    for cluster in new_walktrap_clusters:
                            new_clusters.append(cluster)
                    ic(new_clusters)
                    if alternates_dict:
                        all_alternates_dict = {**all_alternates_dict, **alternates_dict}

        #return(new_clusters)
    else:
        trimmed_connected_set, all_alternates_dict = remove_doubles_by_graph(connected_set, G)
        ic("after trimming by removing doubles", trimmed_connected_set)
        new_clusters = [trimmed_connected_set] 
    # If no new clusters, returns []
    return(new_clusters, all_alternates_dict)
 

#@profile
def check_completeness(cluster):

            seqnums = [x.seqnum for x in cluster]
            clustcounts = Counter(seqnums)
            # If any sequence found more than once
            for value in clustcounts.values():
                if value > 1:
                   return(False)
            return(True)
 





#@profile
def get_walktrap(hitlist):
    # UNUSED
    G = igraph.Graph.TupleList(edges=hitlist, directed=True)
    # Remove multiedges and self loops
    #ic("Remove multiedges and self loops")
    G = G.simplify()
    
    ic("start walktrap")
    clustering = G.community_walktrap(steps = 3).as_clustering()
    ic(clustering)
    ic("walktrap done")
    

    clusters_list = clustering_to_clusterlist(G, clustering)
    return(clusters_list)



#@profile
def get_cluster_dict(clusters):
    ''' in use'''
    pos_to_clustid = {}
    clustid_to_clust = {}
    for i in range(len(clusters)):
       clust = clusters[i]
       clustid_to_clust[i] = clust 
       for seq in clust:
              pos_to_clustid[seq] = i

    return(pos_to_clustid, clustid_to_clust)
 


#@profile
def get_cluster_orders(cluster_dict, seqs_aas):
    # This is getting path of each sequence through clusters 
    cluster_orders_dict = {}

    for i in range(len(seqs_aas)):
        seqnum = seqs_aas[i][0].seqnum # Get the setnum of the set of aas
        #ic("SEQNUM",  seqnum)
        cluster_order = []
        
        for j in range(len(seqs_aas[i])):
           key = seqs_aas[i][j]
           #ic("key", key)
           try:
              clust = cluster_dict[key]
              #ic("clust", clust)
              cluster_order.append(clust)
           except Exception as E:
              #ic(E)
              # Not every aa is sorted into a cluster
              continue
        cluster_orders_dict[seqnum] = cluster_order
    return(cluster_orders_dict)



#@profile
def clusters_to_dag(clusters_filt, seqs_aas, gapfilling_attempt, remove_both = True, dag_reached = False, alignment_group = 0, attempt = 0, minclustsize = 1, all_alternates_dict = {}, args = None):
    ######################################3
    # Remove feedback loops in paths through clusters
    # For getting consensus cluster order
  
    ic("status of remove_both", remove_both)
    numseqs = len(seqs_aas)
    pos_to_clustid, clustid_to_clust = get_cluster_dict(clusters_filt)
    cluster_orders_dict = get_cluster_orders(pos_to_clustid, seqs_aas)
    # Also return removed clusters here
    # For each cluster that's removed, try adding it back one at a time an alternate conformation
    clusters_filt_dag_all, clusters_to_add_back = remove_feedback_edges(cluster_orders_dict, clustid_to_clust,  gapfilling_attempt, remove_both, alignment_group = alignment_group, attempt = attempt, all_alternates_dict= all_alternates_dict, args = args)

    ic("clusters_to_dag", all_alternates_dict)
    ic("clusters to add back", clusters_to_add_back)
    # Each potential group of substitutions to make to pairs of clusters

    already_placed = []
    for key, cluster_group_to_add_back in clusters_to_add_back.items():
       skip_first = False
       skip_second = False
       ic("looking at cluster group", key, cluster_group_to_add_back)
         # Meaning that this cluster has already been placed after error correction
       if list(key)[0] in already_placed:
             skip_first = True
             ic(list(key)[0], "already placed")
       if list(key)[1] in already_placed:
             skip_second = True
             ic(list(key)[1], "already placed")

       for cluster_pair_to_add_back in cluster_group_to_add_back:
         if skip_first == True:
             cluster_pair_to_add_back = [cluster_pair_to_add_back[1]]
         if skip_second == True:
             cluster_pair_to_add_back = [cluster_pair_to_add_back[0]]


         trial =  clusters_filt_dag_all + cluster_pair_to_add_back
         ic("TRIAL", trial)
         pos_to_clustid, clustid_to_clust = get_cluster_dict(trial)
         cluster_orders_dict = get_cluster_orders(pos_to_clustid, seqs_aas)
         clusters_filt_dag_trial, clusters_to_add_back_trial = remove_feedback_edges(cluster_orders_dict, clustid_to_clust,  gapfilling_attempt, remove_both, alignment_group = alignment_group, attempt = attempt, all_alternates_dict= all_alternates_dict, args = args)
         if len(clusters_filt_dag_trial) > len(clusters_filt_dag_all):
              clusters_filt_dag_all = clusters_filt_dag_trial
              ic("Alternate worked better")
              ic("Added back", cluster_pair_to_add_back)
              already_placed = already_placed + list(flatten(key))
              ic("Already placed", already_placed)
              break

    too_small = []
    clusters_filt_dag = []
    for clust in clusters_filt_dag_all:
          if len(clust) >= minclustsize:
                clusters_filt_dag.append(clust)
          else:
                if len(clust) > 2:
                   too_small.append(clust) 


    for x in clusters_filt_dag:
       ic("clusters_filt_dag", x)
  
    for x in too_small:
       ic("removed, too small", x)

    ic("Feedback edges removed")

    ic("Get cluster order after feedback removeal")
    
    pos_to_clust_dag, clustid_to_clust_dag = get_cluster_dict(clusters_filt_dag)

    #for x in clustid_to_clust_dag.items():
    #     ic("What", x)

    cluster_orders_dict = get_cluster_orders(pos_to_clust_dag, seqs_aas)

    #ic(cluster_orders_dict)
    dag_or_not_func = graph_from_cluster_orders(list(cluster_orders_dict.values()))[0].simplify().is_dag()
    ic("Dag or Not? from function, ", dag_or_not_func) 

    if dag_or_not_func == True:
          dag_reached = True
    
    else:
          ic("DAG not reached, will try to remove further edges")
          dag_reached = False
    # Not using too_small here
    return(cluster_orders_dict, pos_to_clust_dag, clustid_to_clust_dag, dag_reached)



#@profile
def dag_to_cluster_order(cluster_orders, seqs_aas, pos_to_clust_dag, clustid_to_clust_dag):
    '''
    Cluster orders is a list of lists of each sequence's path through the clustids
    Finds a consensus cluster order using a topological sort
    Requires cluster orders to be a DAG
    '''

    cluster_order = get_topological_sort(cluster_orders) 
    #ic("For each sequence check that the cluster order doesn't conflict with aa order")
    # Check if this ever does anything
    cluster_order = remove_order_conflicts(cluster_order, seqs_aas, pos_to_clust_dag)


    clustid_to_clust = {key:val for key, val in clustid_to_clust_dag.items() if key  in cluster_order}

    cluster_order_dict = {}
    for i in range(len(cluster_order)):
        cluster_order_dict[cluster_order[i]] = i

    clustid_to_clust_inorder = {}
    pos_to_clust_inorder = {}
    cluster_order_inorder = []
    for i in range(len(cluster_order)):
         clustid_to_clust_inorder[i] = clustid_to_clust[cluster_order[i]]    
         cluster_order_inorder.append(i)


    for key in pos_to_clust_dag.keys():
         # Avoid particular situation where a sequence only has one matched amino acid, so  
         # it isn't in the cluster order sequence
         if pos_to_clust_dag[key] in cluster_order_dict.keys():
              pos_to_clust_inorder[key] = cluster_order_dict[pos_to_clust_dag[key]]


    return(cluster_order_inorder, clustid_to_clust_inorder, pos_to_clust_inorder)


# Start with doing groups of semantically similar sequences

# Then combine with a different approach that doesn't allow order changes

# There is currently redundancy in order determination. 

# For topological sort, could limit to just area that was modified potentially

 
# Only do rbh for within same sequence set to start
# Then can do a limited rbh or none for the final alignment
# Is there a layer that capture amino acid properties specifically
# Like a substitute blosum

# Suspect will be a rbh between groups of sequences. 
# Reciprocal best group hit?






#@profile
def get_seq_groups(seqs, seq_names, embedding_dict, logging, exclude, do_clustering, outfile_name, record_dir, seqsim_thresh = 0.75):

    aa_embeddings = embedding_dict['aa_embeddings'] # this is numseqs x padded_seqlen x embedding_dim
    padded_seqlen = aa_embeddings.shape[1]
    ic(aa_embeddings.shape)


    logging.info("Flattening hidden states list")
    hidden_states = np.array(reshape_flat(aa_embeddings)) # This is now numseqs*padded_seqlen x embedding_dim
    # eqnums = list(set([x.seqnum for x in index_to_aa.values()]))
    seqnums = list(range(0, len(seqs)))
    ic(seqnums)
    seqs_aas, seq_to_length = get_seqs_aas(seqs, seqnums)
    index_to_aa, hidden_states, seqnum_to_index, batch_list = remove_maxlen_padding(hidden_states, seqs_aas, padded_seqlen)
    ic(seqnum_to_index)
    ic(batch_list)
    # Hidden_states is now all seqlens * embedding_dim
    pca_plot = True 
    if pca_plot:
       png_hidden_out = "{}/{}.initial.prebatch.png".format(record_dir, outfile_name)
       do_pca_plot(hidden_states, index_to_aa, png_hidden_out, seq_to_length = seq_to_length)
    
    no_batch_correct = False
    if not no_batch_correct:
      
        ic( list(range(len(seqs_aas)))) 
        hidden_states = do_batch_correct(hidden_states, list(range(len(seqs_aas))), batch_list)
           
        if pca_plot:
            png_hidden_out = "{}/{}.initial.postbatch.png".format(record_dir, outfile_name)
            do_pca_plot(hidden_states, index_to_aa, png_hidden_out, seq_to_length = seq_to_length)
    

    # Get 
    sequence_embedding_list = []
    for i in range(len(seqs_aas)):
          seq_indices = seqnum_to_index[i]
          seq_i_aa_embeddings = np.take(hidden_states, seq_indices, axis = 0)
          #ic(seq_i_aa_embeddings.shape)
          seq_i_seq_embedding = np.mean(seq_i_aa_embeddings, axis = 0)
          #ic(seq_i_seq_embedding.shape)
          sequence_embedding_list.append(seq_i_seq_embedding)
    sequence_array = np.array(sequence_embedding_list)
    ic(sequence_array.shape)
    #sequence_array = np.mean(hidden_states_list, axis = 1)

    #else: 
        
    #     sentence_array = np.array(embedding_dict['sequence_embeddings']).astype(np.float32)

    numseqs = len(seqs)

    G, s_index = get_seqsims(sequence_array, k = numseqs)

    to_exclude = []

   
    group_hstates_list = []
    cluster_seqnums_list = []
    cluster_names_list = []
    cluster_seqs_list = []
   

    logging.info("Removing spaces from sequences")
    if do_clustering == True:
        #ic("fastgreedy")
        #ic(G)
    
      repeat = True
      while repeat == True:

        group_hstates_list = []
        cluster_seqnums_list = []
        cluster_names_list = []
        cluster_seqs_list = []
 
        prev_to_exclude = to_exclude
        

    
        ic("GG", G.vs()['name'])
        ic("GG", G.es()['weight'])
        edgelist = []
        weightlist = []
        for edge in G.es():
             if G.vs[edge.target]["name"] not in to_exclude:
                  if G.vs[edge.source]["name"] not in to_exclude:
                     if edge['weight'] >= seqsim_thresh:
                         #if edge.source != edge.target:
                             source = G.vs[edge.source]["name"]
                             target = G.vs[edge.target]["name"]
                        
 
 
                             length_diff_correction = 1 - abs(0.5 - len(seqs[source]) / (len(seqs[source]) + len(seqs[target])))
                             corrected_weight = edge['weight'] * length_diff_correction
                             ic("seqsim: ", source,target, edge['weight'], "l1, l2", len(seqs[source]), len(seqs[target]), corrected_weight)
                             if corrected_weight >= seqsim_thresh:
                             
                                 edgelist.append([ source, target ])
                                 weightlist.append(corrected_weight)
        # Rebuild G
        G = igraph.Graph.TupleList(edges=edgelist, directed=False)
        G.es['weight'] = weightlist

        G = G.simplify(combine_edges=max)
        ic("G", G)
        #seq_clusters = G.community_multilevel(weights = 'weight')
        ## The issue with walktrap is that the seq sim graph is near fully connected
        #ic("multilevel", seq_clusters)
        #seq_clusters = G.community_walktrap(steps = 3, weights = 'weight').as_clustering() 
        #ic("walktrap", seq_clusters)
        #seq_clusters = G.community_fastgreedy(weights = 'weight').as_clustering() 
        #ic("fastgreedy", seq_clusters)

        seq_clusters = G.community_walktrap(steps = 3, weights = 'weight').as_clustering() 
        #for x in seq_clusters.subgraphs():
        #     ic("subgraph", x)      
        if len(seq_clusters.subgraphs()) == len(G.vs()):
        #     #seq_cluster = seq_clusters.vs()['name']
             seq_clusters = G.clusters(mode = "weak") # walktrap can cluster nodes individually. See UBQ
         #                     # If that happens, use original graph
         #    ic("is this happening")

        # Spinglass doesn't work on disconnected graphs
        # Spinglass wins. See  Eg. Extradiol_dioxy 
        #G_weak  = G.clusters(mode = "weak")
        #for sub_G in G_weak.subgraphs():
        #    sub_seq_clusters = sub_G.community_spinglass(weights = 'weight') 
        #        
        #    ic("spinglass", sub_seq_clusters)
        #    for seq_cluster_G in sub_seq_clusters.subgraphs():
        ic("After walktrap", seq_clusters)
        for seq_cluster_G in seq_clusters.subgraphs():
        
                # Do exclusion within clusters
                ic("seq_clusters", seq_cluster_G)
                if exclude == True:
    
                    clust_names = seq_cluster_G.vs()["name"]
                    ic("clust_names", clust_names)
                    cluster_to_exclude = candidate_to_remove(seq_cluster_G, clust_names, z = -5)
                    ic(cluster_to_exclude)
                       
                    #ic('name', to_exclude)
                    to_delete_ids_sub_G = [v.index for v in seq_cluster_G.vs if v['name'] in cluster_to_exclude]
                    #ic('vertix_id', to_delete_ids)
                    seq_cluster_G.delete_vertices(to_delete_ids_sub_G) 
    
                    #to_delete_ids_G = [v.index for v in G.vs if v['name'] in cluster_to_exclude]
                    #G.delete_vertices(to_delete_ids_G)
    
                    ic("to_exclude_pre", to_exclude)
                    to_exclude = to_exclude + cluster_to_exclude
                    to_exclude = list(set(to_exclude))
                    ic("to_exclude_post", to_exclude)
                    if to_exclude:       
                        logging.info("Excluding following sequences: {}".format(",".join([str(x) for x in to_exclude])))
                        ic("Excluding following sequences: {}".format(",".join([str(x) for x in to_exclude])))
    
                hstates = []
                seq_cluster = seq_cluster_G.vs()['name']
                seq_cluster.sort()
                ic(seq_cluster)
                cluster_seqnums_list.append(seq_cluster)
        
                filter_indices = seq_cluster
                group_hstates = np.take(embedding_dict['aa_embeddings'], filter_indices, axis = 0)
                group_hstates_list.append(group_hstates)
                #Aic(group_hstates.shape)
        
                cluster_names = [seq_names[i] for i in filter_indices]
                cluster_names_list.append(cluster_names)
           
                cluster_seq = [seqs[i] for i in filter_indices]
                cluster_seqs_list.append(cluster_seq)
                to_exclude = list(set(to_exclude))
        ic("eq check", to_exclude, prev_to_exclude)
        if set(to_exclude) == set(prev_to_exclude):
           repeat = False
        else:
               cluster_seqs_list = [] 
               cluster_seqnums_list = []
               group_hstates_list = []
               cluster_names_list= []
    else:
         if exclude == True:
            clust_names = G.vs()["name"] 
            to_exclude = candidate_to_remove(G, clust_names, z = -3)
            ic('name', to_exclude)
            to_delete_ids = [v.index for v in G.vs if v['name'] in to_exclude]
            #ic('vertix_id', to_delete_ids)
            G.delete_vertices(to_delete_ids) 
    
            logging.info("Excluding following sequences: {}".format(",".join([str(x) for x in to_exclude])))
    
         else:
           logging.info("Not removing outlier sequences")
           to_exclude = []
 
 
        # #ic([v['name'] for v in G.vs])
         cluster_seqnums_list =  [v['name'] for v in G.vs]
         ic(cluster_seqnums_list, to_exclude)
         cluster_seqnums_list = list(set(cluster_seqnums_list))
         cluster_seqnums_list.sort()
         # Make sure this is removing to_exclude corectly
         cluster_seqs_list = [[seqs[i] for i in cluster_seqnums_list]]
         cluster_names_list = [[seq_names[i] for i in cluster_seqnums_list]]
         group_hstates_list = [np.take(embedding_dict['aa_embeddings'], cluster_seqnums_list, axis = 0)]
         cluster_seqnums_list = [cluster_seqnums_list] 
         to_exclude = list(set(to_exclude))

    ic("seqnum clusters", cluster_seqnums_list)
    ic(cluster_names_list)
    # Anything lost in the processes

    ic("cluster_seqnums_list", cluster_seqnums_list) 
    dropped_seqs = [ x for x in list(range(0,len(seqs))) if x not in flatten(cluster_seqnums_list)]
    ic("Dropped seqs:", dropped_seqs)
    to_exclude = list(set(to_exclude + dropped_seqs))    
    ic("final to exclude", to_exclude)

    return(cluster_seqnums_list, cluster_seqs_list,  cluster_names_list, group_hstates_list, to_exclude)


#@profile
def dedup_clusters(clusters_list, G, minclustsize):
    new_clusters_list = []

    pos_to_clustid, clustid_to_clust = get_cluster_dict(clusters_list)
    for clust in clusters_list:
        if len(clust) > len(get_represented_seqs(clust)):

             resolved = False
             ic("has dups after very first clustering", clust)
             #for pos in clust:
             for otherclust in clusters_list:
               if clusters_list == otherclust:
                   continue
               # Check if removing a smaller cluster resolved duplicates
               if len(set(clust).intersection(set(otherclust))) >= 2:
                    trimmed_clust = [x for x in clust if x not in otherclust]
                    complete = check_completeness(trimmed_clust)
                   
                    if complete: 
                         if trimmed_clust not in new_clusters_list:
                            ic("trimmed cluster", trimmed_clust)
                            new_clusters_list.append(trimmed_clust)
                            resolved = True
             if resolved == False:
                  # Start by trying to resolve with consistency check
                  reduced_clust =  remove_doubles_by_consistency(clust, pos_to_clustid, add_back = True)
                  complete = check_completeness(reduced_clust)
                  if complete:
                      ic("resolved after consistency removal", reduced_clust)
                      new_clusters_list.append(reduced_clust)
                  # Then try by higher score in the original rbh
                  # Potentially replace this with new search "removed_doubles_w_search"
                  else:
                      reduced_clust, alternates_dict =  remove_doubles_by_graph(reduced_clust, G, keep_higher_score = True, remove_both = False)
                      ic(reduced_clust, alternates_dict)
                      complete = check_completeness(reduced_clust)
                      if complete:
                          ic("resolved after graph removal", reduced_clust)
                          new_clusters_list.append(reduced_clust)


        else:
             if clust not in new_clusters_list:
                  new_clusters_list.append(clust)
    return(new_clusters_list)


#@profile
def get_similarity_network(seqs, seq_names, seqnums, hstates_list, logging, minscore1 = 0.5, alignment_group = 0, args = None):
    """
    Required args: record_dir, outfile_name, do_pca, headnorm
    Control for running whole alignment process
    """
    ic(args)
    pca_plot = args.pca_plot # default True
    headnorm = args.headnorm # default False
    record_dir = args.record_dir
    outfile_name = args.outfile_name
    no_batch_correct = args.no_batch_correct
    seqlens = [len(x) for x in seqs]
    ic("seqs", seqs, seqlens)

     
    numseqs = len(seqs)
    padded_seqlen = hstates_list.shape[1]
    embedding_length = hstates_list.shape[2]
    ic("numseqs", numseqs)
    ic("padded_seqlen", padded_seqlen)
    ic(hstates_list.shape)


    logging.info("Flattening hidden states list")
    hidden_states = np.array(reshape_flat(hstates_list))  
    logging.info("embedding_shape: {}".format(hidden_states.shape))


    
    logging.info("Convert index position to amino acid position")

    seqs_aas, seq_to_length = get_seqs_aas(seqs, seqnums)
    index_to_aa, hidden_states, seqnum_to_index, batch_list = remove_maxlen_padding(hidden_states, seqs_aas, padded_seqlen)
    if pca_plot:
        png_hidden_out = "{}/{}.alignment_group{}.prebatch.png".format(record_dir, outfile_name, alignment_group)
        do_pca_plot(hidden_states, index_to_aa, png_hidden_out, seq_to_length = seq_to_length)


    if not no_batch_correct:
        ic( list(range(len(seqs_aas)))) 
        hidden_states = do_batch_correct(hidden_states, list(range(len(seqs_aas))), batch_list)
       
        if pca_plot:
            png_hidden_out = "{}/{}.alignment_group{}.postbatch.png".format(record_dir, outfile_name, alignment_group)
            do_pca_plot(hidden_states, index_to_aa, png_hidden_out, seq_to_length = seq_to_length)

     
    faiss.normalize_L2(hidden_states)
    index= build_index_flat(hidden_states, scoretype = "cosinesim", normalize_l2 = False) # Already normalized
    logging.info("Index built") 

    if pca_plot:
        png_hidden_out = "{}/{}.alignment_group{}.postnorm.png".format(record_dir, outfile_name, alignment_group)
        do_pca_plot(hidden_states, index_to_aa, png_hidden_out, seq_to_length = seq_to_length)


    # Get KNN of eac amino acid
    D1, I1 =  index.search(hidden_states, k = numseqs*20) 

    
 
    logging.info("KNN done")
    
    I2 = split_distances_to_sequence2(D1, I1, index_to_aa, numseqs, seqlens) 
    # I2 is a dictionary of dictionaries of each aa1: {aa1:1.0, aa2:0.8}

    logging.info("Split results into proteins done")

    logging.info("get best hitlist")
    minscore1 = minscore1
 
    hitlist_all = get_besthits(I2, minscore = 0) # High threshold for first clustering 
    
    for x in hitlist_all:
       ic("hitlist_all:", x)
    logging.info("got best hitlist")

    #logging.info("get top hitlist")
  
    logging.info("get reciprocal best hits")

    rbh_list = get_rbhs(hitlist_all)  
    # What was min_edges = 5 about 
    #rbh_list = get_rbhs(hitlist_all, min_edges = 5)  
    logging.info("got reciprocal best hits")
   
    #remove_streaks = False
    #if remove_streaks == True:
    #    logging.info("Remove streak conflict matches")
    #    rbh_list = remove_streakbreakers(rbh_list, seqs_aas, seqnums, seqlens, streakmin = 3)

    for x in rbh_list:
      ic("rbh", x) 
   
    ######################################### Do walktrap clustering
    outnet = "{}/{}.testnet_initial_clustering{}.csv".format(record_dir, outfile_name, alignment_group)
    with open(outnet, "w") as outfile:
          outfile.write("aa1,aa2,score\n")
          # If do reverse first, don't have to do second resort
          for x in rbh_list:
             outstring = "{},{},{}\n".format(x[0], x[1], x[2])        
             outfile.write(outstring)


    ic("Start betweenness calculation to filter cluster-connecting amino acids. Also first round clustering")

    G = graph_from_rbh(rbh_list, directed = False)
  
    # In CBS, how do 10-6-P and 1-6-D end up in same cluster?
    #ic(G.vs.find(name = "0-25-H"))
    #connections = G.es.select(_source = 242)
    #connected_aas = [x.target_vertex['name'] for x in connections]
 

    clusters_list = []
    if len(seqs) > 2:
        minclustsize = int(len(seqs)/2) + 1
        if len(clusters_list) == 0:
            clusters_list, all_alternates_dict = first_clustering(G, betweenness_cutoff = 0.1, ignore_betweenness = False, apply_walktrap = True)

    else:
        minclustsize = 2
        if len(clusters_list) == 0:
            clusters_list, all_alternates_dict = first_clustering(G, betweenness_cutoff = 1, ignore_betweenness = True, apply_walktrap = True)

    #minclustsize = 3

       
    clusters_list = [x for x in clusters_list if len(x) > 1]
    for x in clusters_list:
        ic("First clusters", x)

    new_clusters_list = dedup_clusters(clusters_list, G, minclustsize)
    for x in new_clusters_list:
      ic("Deduplicated first clusters", x)


    clusters_filt = []
    too_small = []
    for clust in new_clusters_list:
          if len(clust) >= minclustsize:
                clusters_filt.append(clust)
          else:
             # This is ever happening?
             if len(clust) > 2:
                too_small.append(clust)
    for x in clusters_filt:
          ic("First clusters with small removed", x)
    for x in too_small:
          ic("collected as too_small", x)
    ic("Getting DAG of cluster orders, removing feedback loops")
    cluster_order, clustid_to_clust, pos_to_clustid, alignment = organize_clusters(clusters_filt, seqs_aas, gapfilling_attempt = 0, minclustsize = minclustsize, all_alternates_dict = all_alternates_dict, seqnames = seq_names, args = args)


    #for key, value in clustid_to_clust.items():
    #    ic("pregapfil", key,value)

    #ic("Need to get new clusters_filt")
    clusters_filt = list(clustid_to_clust.values())  

    for x in clusters_filt:
          ic("First clusters after feedback loop removal", x)
    alignment = make_alignment(cluster_order, seqnums, clustid_to_clust, seq_names)
    #ic(alignment_ic(alignment, seq_names)[0])

    #logging.info("\n{}".format(alignment))


    if len(seqnums) > 2:
       if len(seqnums) < 5:
            minclustsize = len(seqnums) - 1
       else:
            minclustsize = 4
    else:
       minclustsize = 2

    ignore_betweenness = False
    minscore = 0.5
    betweenness_cutoff = 0.30
    history_unassigned = {'onebefore':[], 'twobefore':[], 'threebefore':[]}
    ic("HERE")
    ic(alignment)
    most_complete_alignment = alignment
    too_small = [] 
    rbh_dict = {}
    match_dict = {}
    ############## CONTROL LOOP ###################
    for gapfilling_attempt in range(0, 200):
        gapfilling_attempt = gapfilling_attempt + 1
        ic("Align this is gapfilling attempt ", gapfilling_attempt)
        logging.info("gapfilling_attempt {}".format(gapfilling_attempt))
        if gapfilling_attempt > 6 and minclustsize > 2 and gapfilling_attempt % 2 == 1:
                minclustsize = minclustsize - 1
        ic("This is the minclustsize", minclustsize)
         
        unassigned = get_unassigned_aas(seqs_aas, pos_to_clustid, too_small)
        for x in unassigned:
           ic("unassign", x)

        if len(unassigned) == 0:
            ic("Alignment complete after gapfilling attempt {}".format(gapfilling_attempt - 1))
     
            alignment = make_alignment(cluster_order, seqnums, clustid_to_clust, seq_names)
            return(alignment, index, hidden_states, index_to_aa)

        if ( unassigned == history_unassigned['threebefore'] or  unassigned == history_unassigned['twobefore'] ) and gapfilling_attempt > 10:
            if minscore > 0.1:
                minscore = 0.1
                ic("reducing minscore to {} at gapfilling attempt {}".format(minscore, gapfilling_attempt))
            ################ Final stage #############3
            else: 
                alignment = make_alignment(cluster_order, seqnums, clustid_to_clust, seq_names)
                ic("current {}, previous record {}".format(alignment.numassigned,most_complete_alignment.numassigned))
                if alignment.numassigned < most_complete_alignment.numassigned:
                    ic("Replacing current alignment with previous more complete alignment")
                    alignment = most_complete_alignment
                else:
                    ic("Currently alignment is the most complete so far")
                ic("Align by placing remaining amino acids")
                cluster_order, clustid_to_clust, pos_to_clustid, alignment = fill_in_hopeless2(unassigned, seqs_aas, cluster_order, clustid_to_clust, pos_to_clustid, index, hidden_states, gapfilling_attempt, args = args)
                unassigned = get_unassigned_aas(seqs_aas, pos_to_clustid)
                ic('This unassigned should be empty', unassigned)
                alignment = make_alignment(cluster_order, seqnums, clustid_to_clust, seq_names)

                return(alignment,   index, hidden_states, index_to_aa)
 
        history_unassigned['threebefore'] = history_unassigned['twobefore']
        history_unassigned['twobefore'] = history_unassigned['onebefore']
        history_unassigned['onebefore'] = unassigned
        
        apply_walktrap = False

        # Do one or two rounds of clustering between guideposts
        if gapfilling_attempt in list(range(1, 100,2)):#  or gapfilling_attempt in [1, 2, 3, 4]:

            ic("Align by clustering within guideposts")
            # Don't allow modification of previous guideposts
            if gapfilling_attempt > 4:
                 apply_walktrap = True

            if gapfilling_attempt > 3:
              if gapfilling_attempt < 15: 
                # This removes 'stranded' amino acids, where neither the previous or next amino acid are placed adjacent. 
                # If a good cluster, will be added right back
               cluster_order, clustid_to_clust = address_stranded3(alignment)
               alignment = make_alignment(cluster_order, seqnums, clustid_to_clust, seq_names) 
               clusterlist = list(clustid_to_clust.values())
               new_clusterlist = []
               pos_to_clustid, clustid_to_clust = get_cluster_dict(clusterlist)
               unassigned = get_unassigned_aas(seqs_aas, pos_to_clustid)

            # Return too_small from this
            cluster_order, clustid_to_clust, pos_to_clustid, alignment, too_small, rbh_dict, all_new_rbh = fill_in_unassigned_w_clustering(unassigned, seqs_aas, cluster_order, clustid_to_clust, pos_to_clustid, I2,  gapfilling_attempt, minscore = minscore ,minclustsize = minclustsize, ignore_betweenness = ignore_betweenness, betweenness_cutoff = betweenness_cutoff, apply_walktrap = apply_walktrap, rbh_dict = rbh_dict, seqnames = seq_names, args = args)
 
            outnet = "{}/{}.testnet_clustering_group_{}_gapfilling_{:04}.csv".format(record_dir, outfile_name, alignment_group, gapfilling_attempt)
            with open(outnet, "w") as outfile:
                  outfile.write("aa1,aa2,score\n")
                  # If do reverse first, don't have to do second resort
                  for x in all_new_rbh:
                     outstring = "{},{},{}\n".format(x[0], x[1], x[2])        
                     outfile.write(outstring)


 
            for x in too_small:
               ic("collected as too_small after clustering", x)        

            for key,value in clustid_to_clust.items():
                 ic(key, value)

        else:
            
            ic("Align by best match (looser)")
            logging.info("Add aa's to existing clusters")
            if gapfilling_attempt > 3:
              if gapfilling_attempt < 15: 
                # This removes 'stranded' amino acids. 
                # If a good cluster, will be added right back
               #cluster_order, clustid_to_clust = address_stranded3(alignment)
               alignment = make_alignment(cluster_order, seqnums, clustid_to_clust, seq_names) 
               clusterlist = list(clustid_to_clust.values())
               new_clusterlist = []
               pos_to_clustid, clustid_to_clust = get_cluster_dict(clusterlist)
               unassigned = get_unassigned_aas(seqs_aas, pos_to_clustid)
            cluster_order, clustid_to_clust, pos_to_clustid, alignment, match_dict = fill_in_unassigned_w_search(unassigned, seqs_aas, cluster_order, clustid_to_clust, pos_to_clustid, index, hidden_states,  index_to_aa, gapfilling_attempt, minclustsize = minclustsize, remove_both = True, match_dict= match_dict, seqnames = seq_names, args = args)
            too_small = []
            #cluster_order_merge, clustid_to_clust_merge, pos_to_clustid_merge, alignment)


        # THINGS TO DO AT THE END OF GAPFILLING CYCLE
        if alignment.numassigned > most_complete_alignment.numassigned:
            ic("This is the most complete alignment so far")
            most_complete_alignment = alignment
   
        outaln = "{}/{}.alignment_clustering_group_{}_gapfilling_{:04}.aln".format(record_dir, outfile_name, alignment_group, gapfilling_attempt)
        with open(outaln, "w") as outfile:
            outfile.write(alignment.format_aln("clustal"))

           
            #ic(alignment_ic(alignment, seq_names)[0])
    return( alignment,  index, hidden_states,  index_to_aa)   


    

   


#@profile
def format_gaps(unassigned, highest_clustnum):
    # Do this when getting ranges in the first place. Add more to list
    # what is the different between this and np.inf and -inf as used before?
    
    output_unassigned = []
    for gap in unassigned:
        starting_clustid =  gap[0]
        ending_clustid = gap[2]

        if not ending_clustid and ending_clustid != 0:
            ending_clustid = highest_clustnum + 1 # capture the last cluster
         # If unassigned sequence extends before the sequence
        if not starting_clustid and starting_clustid != 0:
           starting_clustid = -1
        output_unassigned.append([starting_clustid, gap[1], ending_clustid])
    return(output_unassigned)

      

#@profile
def get_set_of_scores(gap_aa, index, hidden_states, index_to_aa):

    candidates = get_looser_scores(gap_aa, index, hidden_states)    
    candidates_aa = []
    for score in candidates:
        try:
           target_aa = index_to_aa[score[1]]
        except Exception as E:
           # Not all indices correspond to an aa.
           continue
        candidates_aa.append([target_aa, score[0]])
    return(candidates_aa)
      



#@profile
def get_best_of_matches(clustid_to_clust, matches):
    for clustid in clustid_to_clust.keys():
         potential_matches = [x for x in matches if x[1] == clustid]
         
         if potential_matches :
             match_seqnums = [x[0].seqnum for x in potential_matches]
             match_seqnums = list(set(match_seqnums))
             for seqnum in match_seqnums:
                 potential_matches_seqnum = [x for x in potential_matches if x[0].seqnum == seqnum]
                 #ic("seqnum: {}, matches {}".format(seqnum, potential_matches_seqnum)) 
  
                 current_bestscore = 0
                 current_bestmatch = ""
                 for match in potential_matches_seqnum:
                     if match[2] > current_bestscore:
                           current_bestscore = match[2]
                           current_bestmatch = match[0]
        

                 newclust = clustid_to_clust[clustid] + [current_bestmatch]
                 #ic("Updating {} from {} to {}".format(clustid, clustid_to_clust[clustid], newclust))
                 clustid_to_clust[clustid] = newclust
    #ic(clustid_to_clust)
    return(clustid_to_clust)

 

#@profile
def get_best_matches(starting_clustid, ending_clustid, gap_aa, clustid_to_clust, candidates_w_score):
     # Don't update the cluster here, send it back. 
     # candidate_w_score = zipped tuple [aa, score]
    scores = []
    current_best_score = 0
    current_best_match = ""
    match_found = False
    #ic(starting_clustid, ending_clustid)
    #ic(gap_aa)
    #ic(clustid_to_clust)
    for cand in range(starting_clustid + 1, ending_clustid):
         ic("candidate", gap_aa, cand,  clustid_to_clust[cand], "bestscore", current_best_score, current_best_match)
     
         candidate_aas =  clustid_to_clust[cand]
         incluster_scores = [x for x in candidates_w_score if x[0] in candidate_aas]
         ic("incluster scores", incluster_scores)
         total_incluster_score = sum([x[1] for x in incluster_scores]) / len(incluster_scores) # Take the mean score within the cluster. Or median?
         ic("totla_inclucster", total_incluster_score)
         if total_incluster_score > current_best_score:
            if total_incluster_score > 0.5: # Bad matches being added (ex. 0.3)
              current_best_score = total_incluster_score
              current_best_match = cand
              match_found = True


    if match_found: 
        ic("Match found!", current_best_score, current_best_match, clustid_to_clust[current_best_match]) 
        #old = clustid_to_clust[current_best_match]
        #new = old + [gap_aa]
        #clustid_to_clust[current_best_match] = new
        output = [gap_aa, current_best_match, current_best_score]
        #ic("Updating cluster {} from \n{}\nto\n{}".format(current_best_match, old, new)) 
        #match_score = [gap_aa, current_best_score, current_best_match]

    else:

         #ic("no match found in (existing clusters")    
         output = []
    return(output)


#@profile
def removeSublist(lst):
    #https://www.geeksforgeeks.org/python-remove-sublists-that-are-present-in-another-sublist/
    curr_res = []
    result = []
    for ele in sorted(map(set, lst), key = len, reverse = True):
        if not any(ele <= req for req in curr_res):
            curr_res.append(ele)
            result.append(list(ele))
          
    return result
      




#@profile
def fill_in_unassigned_w_search(unassigned, seqs_aas, cluster_order, clustid_to_clust, pos_to_clustid, index, hidden_states,  index_to_aa, gapfilling_attempt, minclustsize = 1, remove_both = True, match_dict = {}, seqnames = [], args = None):

    '''
    Try group based assignment, this time using new search for each unassigned
    Decision between old cluster and new cluster?
    
    '''
    #seqnums = [x[0].seqnum for x in seqs_aas]
    clusters = list(clustid_to_clust.values())
    matches = []
    #ic("unassigned")
    #for x in unassigned:
       #ic("unassigned", x)

    match_scores = []
    unassigned = format_gaps(unassigned, max(clustid_to_clust.keys())) 


    for gap in unassigned:

        starting_clustid =  gap[0]
        ending_clustid = gap[2]
        ic(gap)
        if starting_clustid in clustid_to_clust.keys():
              # If before start of clusts, will be -1
              starting_clust =  clustid_to_clust[starting_clustid]
        else:
              starting_clust = []
        if ending_clustid  in clustid_to_clust.keys():
              ending_clust =  clustid_to_clust[ending_clustid]
        else:
              ending_clust = []

        ic("already searched", gap[1], starting_clust, ending_clust)
        ic("already searched", gap[1] + starting_clust + ending_clust)
        already_searched = frozenset(gap[1] + starting_clust + ending_clust)
        ic("already searched", frozenset(gap[1] + starting_clust + ending_clust))
        if already_searched in match_dict.keys():
            ic("Matches pulled from cache")
            matches = matches + match_dict[already_searched]
        else:
            gap_matches = []
            for gap_aa in gap[1]:
                candidates_aa = get_set_of_scores(gap_aa, index, hidden_states, index_to_aa)
    
                # For each clustid_to_clust, it should be checked for consistency. 
                output = get_best_matches(starting_clustid, ending_clustid, gap_aa, clustid_to_clust, candidates_aa)
                if output:
                   gap_matches.append(output)
            matches = matches + gap_matches
            match_dict[already_searched] = gap_matches

    for x in matches:
        ic("match", x)

    clustid_to_clust = get_best_of_matches(clustid_to_clust, matches)

    clusterlist = list(clustid_to_clust.values())

    new_clusterlist = []
    pos_to_clustid, clustid_to_clust = get_cluster_dict(clusterlist)
    all_alternates_dict  =  {}
    for clustnum, clust in clustid_to_clust.items():
         to_remove = get_doubled_seqnums(clust)
         if len(to_remove) > 0:
              clust = remove_doubles_by_consistency(clust, pos_to_clustid)
              to_remove = get_doubled_seqnums(clust)
         if len(to_remove) > 0:
              clust, alternates_dict = remove_doubles_by_scores(clust, index, hidden_states, index_to_aa)
              if alternates_dict:
                   all_alternates_dict = {**all_alternates_dict, **alternates_dict}
              to_remove = get_doubled_seqnums(clust)
         if len(to_remove) == 0:
            new_clusterlist.append(clust)     

    for x in new_clusterlist:
        ic("Clusters from best match", x)
   
    cluster_order, clustid_to_clust, pos_to_clustid, alignment = organize_clusters(new_clusterlist, seqs_aas, gapfilling_attempt, minclustsize, all_alternates_dict = all_alternates_dict, seqnames = seqnames, args = args)
    return(cluster_order, clustid_to_clust, pos_to_clustid, alignment, match_dict)


#@profile
def organize_clusters(clusterlist, seqs_aas, gapfilling_attempt,  minclustsize = 1, all_alternates_dict = {}, seqnames = [], args = None):
    ic("All alternates dict at start of organize_clusters", all_alternates_dict)
    seqnums = [x[0].seqnum for x in seqs_aas]
    cluster_orders_dict, pos_to_clust, clustid_to_clust, dag_reached = clusters_to_dag(clusterlist, seqs_aas, remove_both = True, gapfilling_attempt = gapfilling_attempt, minclustsize = minclustsize, all_alternates_dict = all_alternates_dict, args = args)
 
    dag_attempts = 1
    while dag_reached == False:
         
          ic("call point 3")
          clusters_filt = list(clustid_to_clust.values())
          if len(clusters_filt) < 2:
               ic("Dag not reached, no edges left to remove".format(count))
               return(1)
          cluster_orders_dict, pos_to_clust, clustid_to_clust,  dag_reached = clusters_to_dag(clusters_filt, seqs_aas, remove_both = True, gapfilling_attempt = gapfilling_attempt, minclustsize = minclustsize, all_alternates_dict = all_alternates_dict, args = args)          
          dag_attempts = dag_attempts + 1
 
    cluster_order, clustid_to_clust, pos_to_clustid =  dag_to_cluster_order(list(cluster_orders_dict.values()), seqs_aas, pos_to_clust, clustid_to_clust)
    alignment = make_alignment(cluster_order, seqnums, clustid_to_clust, seqnames) 

    return(cluster_order, clustid_to_clust, pos_to_clustid, alignment)


    #while dag_reached == False:

    #   if len(clusters_merged) < 2:
    #       #ic("Dag not reached, no edges left to remove".format(count))
    #       return(1)
    #   #cluster_orders_dag, pos_to_clust_dag, clustid_to_clust_dag, clusters_filt, dag_reached = clusters_to_dag(clusters_filt, seqs_aas)
    #   ic("call point 5")
    #   
    #   cluster_orders_merge, pos_to_clust_merge, clustid_to_clust_merge,  dag_reached = clusters_to_dag(clusters_merged, seqs_aas, remove_both = True, minclustsize= minclustsize, write_ordernet = True)
    #   clusters_merged = list(clustid_to_clust_merge.values())


    ##ic("Dag found, getting cluster order with topological sort of merged clusters")
    #cluster_order_merge, clustid_to_clust_merge, pos_to_clustid_merge =  dag_to_cluster_order(cluster_orders_merge, seqs_aas, pos_to_clust_merge, clustid_to_clust_merge)
 
    ##ic("First gap filling alignment")
    #alignment = make_alignment(cluster_order_merge, seqnums, clustid_to_clust_merge)
    ##ic(alignment_ic(alignment, seq_names)[0])


#@profile
def get_targets(gap, seqs_aas, cluster_order, pos_to_clustid):
        
        starting_clustid = gap[0]
        ending_clustid = gap[2] 
        gap_seqaas = gap[1]
        #gap_seqnum = list(set([x.seqnum for x in gap_seqaas]))[0]
        target_aas_list = get_ranges(seqs_aas, cluster_order, starting_clustid, ending_clustid, pos_to_clustid)
        #if "3-2-Y" in gap_seqaas:
        #      ic("HERE")

        edgelist = []
        target_aas = list(flatten(target_aas_list))
        for query in gap_seqaas:
              for target in target_aas:
                  edgelist.append([query, target])

        return(edgelist)
        #target_aas = list(set(target_aas + gap_seqaas))


#@profile
def address_unassigned_aas(scope_aas, neighbors, I2, minscore = 0.5, ignore_betweenness = False,  betweenness_cutoff = 0.3, minsclustsize = 2, apply_walktrap = True, rbh_dict = {}):

        #ic("neighbors", neighbors)
        #ic("address_unassigned_aas:rbh_dict", rbh_dict)
        # Avoid repeats of same rbh calculation
        ic("address_unassigned_aas:scope_aas", scope_aas)
        ic("rbh_dict.keys()", rbh_dict.keys())
        if True: #not frozenset(scope_aas) in rbh_dict.keys():
            limited_I2 = {}
            # Suspect that this is slow
            for key in I2.keys():
               if key in scope_aas:
                   limited_I2[key] = I2[key].copy() 
            #ic("limited")
            ic("address_unassigned_aas:new_rbh_minscore", minscore)
            for query_aa in limited_I2.keys():             
                 # These keys 
                 for seq in limited_I2[query_aa].keys():
                       #ic("scope", scope_aas) 
                       #ic("neighbors", neighbors[query_aa])
                       
                       #limited_I2[query_aa][seq] = [x for x in limited_I2[query_aa][seq] if x[0] in scope_aas]
                       limited_I2[query_aa][seq] = [x for x in limited_I2[query_aa][seq] if x[0] in neighbors[query_aa]]
            ic("limited_I2", limited_I2)
            # Get reciprocal best hits in a limited range
            new_hitlist = get_besthits(limited_I2, minscore)
      
            new_rbh = get_rbhs(new_hitlist)
            ic("new_rbh from get_rbh", new_rbh[0:5])
            rbh_dict[frozenset(scope_aas)] = new_rbh
        else:
            ic("address_unassigned_aas RBH pulled from cache")
            new_rbh = rbh_dict[frozenset(scope_aas)]   
        for x in new_rbh:
             ic("address_unassigned_aas:new_rbh", x)
        G = graph_from_rbh(new_rbh) 
        new_clusters,all_alternates_dict  = first_clustering(G, betweenness_cutoff = betweenness_cutoff,  ignore_betweenness = ignore_betweenness, apply_walktrap = apply_walktrap ) 
     
        for x in new_clusters:
            ic("address_unassigned_aas:new_clusters", x)
 
        #new_clusters = removeSublist(new_clusters)
        clustered_aas = list(flatten(new_clusters))

        return(new_clusters, new_rbh, rbh_dict, all_alternates_dict)


       




#@profile
def fill_in_unassigned_w_clustering(unassigned, seqs_aas, cluster_order, clustid_to_clust, pos_to_clustid, I2, gapfilling_attempt,  minscore = 0.1, minclustsize = 2, ignore_betweenness = False, betweenness_cutoff = 0.3, apply_walktrap = True, rbh_dict = {}, seqnames = [], args = None ):        
    '''
    Run the same original clustering, ??? allows overwritting of previous clusters
    
    No, do network to find ranges for allxall search. 
    If a new member of an rbh cluster has a large unassigned range, check if has higher rbh t o sequence?
    '''
    all_alternates_dict = {}
    clusters_filt = list(clustid_to_clust.values())
    ic("fill_in_unassigned_w_clustering: TESTING OUT CLUSTERS_FILT")
    for x in clusters_filt:
        ic("fill_in_unassigned_w_clustering: preassignment clusters_filt", x)
    # extra arguments?
    edgelist = []
    ic("fill_in_unassigned_w_clustering:unassigned", unassigned)
    for gap in unassigned:
        ic("fill_in_unassigned_w_clustering", gap)
        gap_edgelist = get_targets(gap, seqs_aas, cluster_order, pos_to_clustid)
        edgelist = edgelist + gap_edgelist
        #ic("gap_edgelist", gap_edgelist)
    edgelist_G = igraph.Graph.TupleList(edges=edgelist, directed=False)
    edgelist_G = edgelist_G.simplify() # Remove duplicate edges
    islands = edgelist_G.clusters(mode = "weak") 

    new_clusters_from_rbh = []
    all_new_rbh  = []
    # Neighbors = everything in the subgraph
    # Something about this not working 0-10-Y, 4-8-Y
    # Make sure gap is also identical, not just the vertexes?
    for sub_G in islands.subgraphs():
       neighbors = {}
       for vertex in sub_G.vs():
           vertex_neighbors = sub_G.neighbors(vertex)
           neighbors[vertex['name']] = sub_G.vs[vertex_neighbors]["name"] + [vertex['name']]
          
       newer_clusters, newer_rbh, rbh_dict, alternates_dict = address_unassigned_aas(sub_G.vs()['name'], neighbors, I2, minscore = 0.5, ignore_betweenness = False,  betweenness_cutoff = 0.3, minsclustsize = 2, apply_walktrap = apply_walktrap, rbh_dict = rbh_dict)
       ic(newer_clusters[0:5])
       ic(newer_rbh[0:5])
       all_alternates_dict = {**all_alternates_dict, **alternates_dict}
       new_clusters_from_rbh  = new_clusters_from_rbh + newer_clusters
       all_new_rbh = all_new_rbh + newer_rbh
    new_clusters = []
    too_small = []
    

    for clust in new_clusters_from_rbh:
          if len(clust) >= minclustsize:
                new_clusters.append(clust)
          else:
             # This is never happening?
             if len(clust) > 1:
                too_small.append(clust)

    for x in new_clusters:
        ic("All new clusters", x)
    ic("or here")
    #ic("New clusters:", new_clusters)
    # Very important to removeSublist here
    # Is it anymore?
    new_clusters = removeSublist(new_clusters)
    # Need to remove overlaps from clusters
    # Get amino acids in more than one cluster, remove them. 

    #ic("New clusters after sublist removal",  new_clusters) 
    aa_counter = {}
    new_clusters_flat  = flatten(new_clusters) 
    #ic("flat_clusters", new_clusters_flat)
    aa_counts = Counter(new_clusters_flat)
    dupped_aas = {key for key, val in aa_counts.items() if val != 1}
    ic("dupped aas", dupped_aas)

    # From doubled aas from clusters list of lists
    new_clusters = [[aa for aa in clust if aa not in dupped_aas] for clust in new_clusters]
        

    # If this makes clusters too small remove them
 
    new_clusters = [clust for clust in new_clusters if len(clust) >= minclustsize]

    ic("WHAT is minclustsize", minclustsize)
    for x in new_clusters:
        ic("All new new clusters", x)

    #ic("New clusters after overlap removal",  new_clusters) 
    # Due to additional walktrap, there's always a change that a new cluster won't be entirely consistent with previous clusters. 
    # In this section, remove any members of a new cluster that would bridge between previous clusters and cause over collapse
    #ic(pos_to_clustid)
    new_clusters_filt = []
    for clust in new_clusters:
         clustids = []
         posids = []
         new_additions = []
         for pos in clust:      
            #ic(pos)
            if pos in pos_to_clustid.keys():
               clustid = pos_to_clustid[pos]
               clustids.append(clustid)
               posids.append(pos)
               ic(pos, clustid)
            else:
                # Position wasn't previously clustered
                #ic("new_additions", clust,pos)
                new_additions.append(pos)
         ic("new_additions", new_additions)
         #ic("posids", posids)                  
         if len(list(set(clustids))) > 1:
            #ic("new cluster contains component of multiple previous clusters. Keeping largest matched cluster")
            clustcounts = Counter(clustids)
            largest_clust = max(clustcounts, key=clustcounts.get)   
            sel_pos = [posids[x] for x in range(len(posids)) if clustids[x] == largest_clust]
            #ic("Split cluster catch", clustcounts, largest_clust, posids, clustids, sel_pos)
            new_clust = sel_pos + new_additions
                
         else:
            new_clusters_filt.append(clust)             


    new_clusters_filt = removeSublist(new_clusters_filt)

    clusters_new = remove_overlap_with_old_clusters(new_clusters_filt, clusters_filt)
    clusters_merged = clusters_new + clusters_filt

    for x in clusters_merged:
       ic("clusters_merged", x)
    cluster_order, clustid_to_clust, pos_to_clustid, alignment = organize_clusters(clusters_merged, seqs_aas, gapfilling_attempt, minclustsize, all_alternates_dict = all_alternates_dict, seqnames = seqnames, args = args)
    return(cluster_order, clustid_to_clust, pos_to_clustid, alignment, too_small, rbh_dict, all_new_rbh)


 



#@profile
def fill_in_hopeless2(unassigned,  seqs_aas, cluster_order, clustid_to_clust, pos_to_clustid, index, hidden_states, gapfilling_attempt, args = None):
    '''
    For amino acids with no matches at all, add in as singleton clusters
    '''
    seqnums = [x[0].seqnum for x in seqs_aas]
    clusters_filt = list(clustid_to_clust.values())
    for gap in unassigned:
        #ic("GAP", gap)
        starting_clustid = gap[0]
        ending_clustid = gap[2]
       
        gap_seqaas = gap[1]
        gap_seqnum = list(set([x.seqnum for x in gap_seqaas]))[0]
         
        target_seqs_list = get_ranges(seqs_aas, cluster_order, starting_clustid, ending_clustid, pos_to_clustid) 

 
        #ic(target_seqs_list)
        target_seqs = list(flatten(target_seqs_list))
        target_seqs = [x for x in target_seqs if not x.seqnum == gap_seqnum]
 
        for aa in gap_seqaas:
               clusters_filt.append([aa])
    
    cluster_order, clustid_to_clust, pos_to_clustid, alignment = organize_clusters(clusters_filt, seqs_aas, gapfilling_attempt, minclustsize = 1, args = args)
    return(cluster_order, clustid_to_clust, pos_to_clustid, alignment)

   



#@profile
def get_align_args():

    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--in", dest = "fasta_path", type = str, required = True,
                        help="Path to fasta")
    
    parser.add_argument("-e", "--emb", dest = "embedding_path", type = str, required = False,
                        help="Path to embeddings")

    parser.add_argument("-o", "--outfile", dest = "out_path", type = str, required = True,
                        help="Path to outfile")

    parser.add_argument("-nb", "--no_batch_correct", dest = "no_batch_correct", action = "store_true", 
                        help="If added, don't do batch correction on sequences")
    parser.add_argument("-sl", "--seqlimit", dest = "seqlimit", type = int, required = False,
                        help="Limit to n sequences. For testing")

    parser.add_argument("-ex", "--exclude", dest = "exclude", action = "store_true",
                        help="Exclude outlier sequences from initial alignment process")


    parser.add_argument("-fx", "--fully_exclude", dest = "fully_exclude", action = "store_true",
                        help="Additionally exclude outlier sequences from final alignment")

    parser.add_argument("-l", "--layers", dest = "layers", nargs="+", type = int,
                        help="Which layers (of 30 in protbert) to select")
    parser.add_argument("-hd", "--heads", dest = "heads", type = str,
                        help="File will one head identifier per line, format layer1_head3")

    parser.add_argument("-st", "--seqsimthresh", dest = "seqsimthresh",  type = float, required = False, default = 0,
                        help="Similarity threshold for clustering sequences")


    parser.add_argument("-m", "--model", dest = "model_name",  type = str, required = True,
                        help="Model name or path to local model")

    parser.add_argument("-p", "--pca_plot", dest = "pca_plot",  action = "store_true", required = False, 
                        help="If flagged, output 2D pca plot of amino acid clusters")

    parser.add_argument("-l2", "--headnorm", dest = "headnorm",  action = "store_true", required = False, 
                        help="Take L2 normalization of each head")
    args = parser.parse_args()

    return(args)


#@profile
def  do_pca_plot(hidden_states, index_to_aa, outfile, clustid_to_clust = None, seq_to_length = None):

        #filter down the hidden states, TODO

        
        seqnums = list(set([x.seqnum for x in index_to_aa.values()])) 
        ic("seqnums", seqnums)
        if clustid_to_clust:
            aa_to_clustid = {} 

  
            ic(clustid_to_clust)
            for clustid, aas in clustid_to_clust.items(): 
               for aa in aas: 
                  aa_to_clustid[aa] = clustid
                
            clustid_to_color = {}
            for key in clustid_to_clust.keys():
               ic(key)
               clustid_to_color[key] = (random.random(), random.random(),random.random())

        elif seq_to_length:
            seqnum_to_color = {}
           
            for key in seqnums:
                seqnum_to_color[key] = (random.random(), random.random(),random.random())

        indexes = list(index_to_aa.keys())
        hidden_states_aas = hidden_states[indexes, :] 

        d1 = hidden_states.shape[1]
        target = 128

        pca = faiss.PCAMatrix(d1, target)

        pca.train(np.array(hidden_states))
 
        #pkl_pca_in = "/scratch/gpfs/cmcwhite/qfo_2020/qfo_sample5000.fasta.aa.128dim.pcamatrix.pkl"
        #with open(pkl_pca_in, "rb") as f:
        #    cache_pca = pickle.load(f)
        #    pcamatrix = cache_pca['pcamatrix']
        #    bias = cache_pca['bias']


        bias = faiss.vector_to_array(pca.b)
        pcamatrix = faiss.vector_to_array(pca.A).reshape(pca.d_out, pca.d_in)

        reduced = np.array(hidden_states_aas) @ pcamatrix.T + bias
        ic(reduced.shape)

        colorlist = []

        labellist  = []
        aalist = []   
        seqlist = []
        poslist = []
        for i in range(len(hidden_states)):
            if i in index_to_aa.keys(): 
              aa = index_to_aa[i]
              aalist.append(aa.seqaa)
              seqlist.append(aa.seqnum)
              poslist.append(aa.seqpos)

              if clustid_to_clust:
                  clustid = aa_to_clustid[aa] 
                  ic(clustid)

           
                  color = clustid_to_color[clustid]
                  labellist.append(clustid)
              elif seqlens:
                  color = seqnum_to_color[aa.seqnum]
                  labellist.append(aa.seqnum)
                  
              else:
                  color = "blue"
                  labellist.append(0)
             #ic(color)
              colorlist.append(color)
        label_arr = np.array(labellist)
        color_arr = np.array(colorlist)
       
  
        for dim1 in [1,2,3]:
          for dim2 in [1,2,3]:
            if dim1 == dim2:
                continue

            plt.figure()           
            if clustid_to_clust: 
                for iclust in clustid_to_clust.keys(): 
                   plt.scatter(reduced[:,dim1-1][label_arr == iclust], reduced[:,dim2-1][label_arr == iclust], c = color_arr[label_arr == iclust], alpha = 0.8, label = iclust)
            if seq_to_length:
                for iclust in seq_to_length.keys(): 
                   plt.scatter(reduced[:,dim1-1][label_arr == iclust], reduced[:,dim2-1][label_arr == iclust], c = color_arr[label_arr == iclust], alpha = 0.8, label = iclust)
            plt.legend()
            plt.xlabel('component {}'.format(dim1))
            plt.ylabel('component {}'.format(dim2))


            plt.savefig("{}.pca{}{}.png".format(outfile,dim1,dim2))
            plt.clf()
        pcasave= pd.DataFrame(reduced[:,[0,1,2,3,4,5,6]])
        pcasave['clustid'] = labellist
        pcasave['color'] = colorlist
        pcasave['seq'] = seqlist
        pcasave['pos'] = poslist
        pcasave['aa'] = aalist

        ic(pcasave)
        pcasave.to_csv("{}.pca.csv".format(outfile), index = False) 

if __name__ == '__main__':

    args = get_align_args()

    ic("args", args)
    fasta_path = args.fasta_path
    embedding_path = args.embedding_path
    outfile = args.out_path
    exclude = args.exclude
    fully_exclude = args.fully_exclude
    layers = args.layers
    heads = args.heads
    model_name = args.model_name
    pca_plot = args.pca_plot
    seqlimit = args.seqlimit
    headnorm = args.headnorm
    seqsim_thresh  = args.seqsimthresh 
    # Keep to demonstrate effect of clustering or not
    do_clustering = True
 
    outfile_path =  os.path.dirname(outfile) # get place to store output file
    outfile_name =   os.path.splitext(os.path.basename(outfile))[0] # get outfile without extension
    if outfile_path:
       record_dir = "{}/alignment_files_{}".format(outfile_path,outfile_name)
    else:
       record_dir = "alignment_files_{}".format(outfile_name)
    if os.path.exists(record_dir):
        shutil.rmtree(record_dir, ignore_errors=True)
    os.mkdir(record_dir)

    args.outfile_name  = outfile_name
    args.record_dir = record_dir

    ic("Params!", args)
    logname = "{}/{}.align.log".format(record_dir, outfile_name)
    #ic("logging at ", logname)
    log_format = "%(asctime)s::%(levelname)s::"\
             "%(filename)s::%(lineno)d::%(message)s"
    logging.basicConfig(filename=logname, level='DEBUG', format=log_format)

    #logging.info("Check for torch")
    #logging.info(torch.cuda.is_available())

    #model_name = 'prot_bert_bfd'


    if heads is not None:
       with open(heads, "r") as f:
         headnames = f.readlines()
         ic(headnames)
         headnames = [x.replace("\n", "") for x in headnames]

         ic(headnames)
    else:
       headnames = None
    logging.info("Check for torch")
    logging.info(torch.cuda.is_available())

    # Padding value Very important for this one
    # Glyco_hydro_18_D2.vie.3seqs.fasta
    # padding = 5 or 9 does NOT work
    # padding = 10 and 11 does
    padding = 10
    minscore1 = 0.5

    logging.info("model: {}".format(model_name))
    logging.info("fasta: {}".format(fasta_path))
    logging.info("padding: {}".format(padding))
    logging.info("first score thresholds: {}".format(minscore1))
   
    seq_names, seqs, seqs_spaced = parse_fasta_for_embed(fasta_path, padding = padding)
    if seqlimit:
       seq_names = seq_names[0:seqlimit]
       seqs = seqs[0:seqlimit]
       seqs_spaced = seqs_spaced[0:seqlimit]

 
    ic("Sequences", seqs)    
    if embedding_path:
       with open(embedding_path, "rb") as f:
             embedding_dict = pickle.load(f)

    else:
        seqlens = [len(x) for x in seqs]
        embedding_dict = get_embeddings(seqs_spaced,
                                    model_name,
                                    seqlens = seqlens,
                                    get_sequence_embeddings = True,
                                    get_aa_embeddings = True,
                                    layers = layers,  
                                    padding = padding,
                                    heads = headnames)

    # First do batch correction
    


    # Get groups of internally consistent sequences 
    ic("seqsim_thresh", seqsim_thresh)
    cluster_seqnums_list, cluster_seqs_list,  cluster_names_list, cluster_hstates_list, to_exclude = get_seq_groups(seqs ,seq_names, embedding_dict, logging, exclude, do_clustering, outfile_name, record_dir, seqsim_thresh = seqsim_thresh)

    aln_fasta_list = []
    excluded_records = []
    for excluded_seqnum in to_exclude:
         
         excluded_record = SeqRecord(Seq(seqs[excluded_seqnum]), id=seq_names[excluded_seqnum], description = '')
         excluded_records.append(excluded_record)
         # Option to keep poor matches out
         if fully_exclude != True:
            aln_fasta_list.append([">{}\n{}\n".format(seq_names[excluded_seqnum], seqs[excluded_seqnum])])
   

    with open("{}/{}.excluded.fasta".format(record_dir, outfile_name), "w") as output_handle:
        SeqIO.write(excluded_records, output_handle, "fasta")

    alignments = []
    hidden_states_list = []
    index_to_aas_list = []

    # For each sequence group, do a sub alignment
    for i in range(len(cluster_names_list)):
        group_seqs = cluster_seqs_list[i]

             

        group_seqnums = cluster_seqnums_list[i]
        group_names = cluster_names_list[i]
        group_embeddings = cluster_hstates_list[i] 
        ic("group seqnames", group_names, group_seqnums)

        group_seqs_out = "{}/{}.alignment_group{}.fasta".format(record_dir, outfile_name, i)
        group_records = []

        for j in range(len(group_seqs)):
             group_records.append(SeqRecord(Seq(group_seqs[j]), id=group_names[j], description = ''))
 
        with open(group_seqs_out, "w") as output_handle:
            SeqIO.write(group_records, output_handle, "fasta")

        if len(group_names) ==  1:
             aln_fasta_list.append([">{}\n{}\n".format(group_names[0], group_seqs[0])])


        else:
            # Main function
            alignment, index, hidden_states, index_to_aa = get_similarity_network(group_seqs, group_names, group_seqnums, group_embeddings, logging, minscore1 = minscore1, alignment_group = i, args = args)
            alignments.append(alignment)
            index_to_aas_list.append(index_to_aa)
            hidden_states_list.append(hidden_states)
    
            cluster_order, clustid_to_clust = clusts_from_alignment(alignment)
    
            ic("attempt squish")       
            for rep in range(0,10):
                 prevclust = alignment
                 cluster_order, clustid_to_clust = squish_clusters2(alignment, index, hidden_states, index_to_aa)                
                 alignment = make_alignment(cluster_order, group_seqnums, clustid_to_clust, group_names)
                 if alignment.alignment == prevclust:
                        break
            alignment = make_alignment(cluster_order, group_seqnums, clustid_to_clust, group_names)
             
            # Too much unpredictable behavior, need to include scores if doing this  
            #cluster_order, clustid_to_clust = squish_clusters_longrange2(alignment)
    
            # Need all the embeddings from the sequence
            # Need clustid_to_clust
            
            if pca_plot: 
                png_align_out = "{}/{}.alignment_group{}.fasta.png".format(record_dir, outfile_name,  i)
                do_pca_plot(hidden_states, index_to_aa, png_align_out, clustid_to_clust = clustid_to_clust)
    
    
    
            #str_alignment = obj_aln_to_str(alignment)
          
            aln_fasta_list_group = []
            #for k in range(len(str_alignment)):
            #       aln_fasta_list_group.append(">{}\n{}\n".format(group_names[k], str_alignment[k]))    
            #       aln_fasta_list_group.append(">{}\n{}\n".format(group_names[k], str_alignment[k]))    
    
            fasta_align_i = alignment.format_aln("fasta")
            clustal_align_i = alignment.format_aln("clustal")   #alignment_ic(alignment, group_names)
    
            aln_fasta_list.append([">" + x for x in fasta_align_i.split(">")][1:]) # fasta_align_i.split("\n")) 
    
    
            fasta_align_out = "{}/{}.alignment_group{}.fasta.aln".format(record_dir, outfile_name, i)
            #fasta_align_i = alignments_i[1]
            with open(fasta_align_out, "w") as o:
                  o.write(fasta_align_i)
    
            clustal_align_out = "{}/{}.alignment_group{}.clustal.aln".format(record_dir, outfile_name, i)
            i#clustal_align_i = alignments_i[0]
            with open(clustal_align_out, "w") as o:
                  o.write(clustal_align_i)

            # If nothing to merge
            if len(cluster_names_list) == 1 and (( len(excluded_records) == 0 or fully_exclude == True )) :
                with open(outfile, "w") as o:
                      o.write(clustal_align_i)
                sys.exit()
           
         

            
    
   
    consolidator = "mafft"
    if consolidator == "mafft":
      if len(cluster_names_list) > 1 or ( len(excluded_records) > 0 or fully_exclude == False ) :
    
        seq_count = 1
    
        


        ic("aln_fasta_list", aln_fasta_list)
        with open("{}/{}.all_fastas_aln.fasta".format(record_dir, outfile_name,), "w") as o:
    
            with open("{}/{}.key_table.txt".format(record_dir, outfile_name), "w") as tb:
                for k in range(len(aln_fasta_list)):
                  
                   for s in range(len(aln_fasta_list[k])):
                        #ic(aln_fasta_list[k][s])
                        o.write("{}\n".format(aln_fasta_list[k][s]))
                        tb.write("{} ".format(seq_count))
                        seq_count = seq_count + 1
                   tb.write("\n")
        

        try:
            
            #os.system("mafft --clustalout --merge key_table.txt --auto all_fastas_aln.fasta > {}".format(outfile))
            os.system("singularity exec /scratch/gpfs/cmcwhite/mafft_7.475.sif mafft --clustalout --merge {}/{}.key_table.txt --auto {}/{}.all_fastas_aln.fasta > {}".format(record_dir, outfile_name, record_dir, outfile_name, outfile))
               
            os.system("cat {}".format(outfile))
        except Exception as E:
            ic("Not doing mafft merge") 
    
    #if consolidator == "embeddings":
    ic("is this happening")
    for i in range(len(alignments)):
                    mylist = []
                    cluster_order, clustid_to_clust = clusts_from_alignment(alignments[i])
                    #ic(cluster_order)
                    #ic(clustid_to_clust)
                    for key, value in clustid_to_clust.items():
                          clustid_embeddings = []
                          indexes = [x.index for x in value]
                          #ic("indexes", indexes)
                          clustid_embeddings = np.take(hidden_states_list[i], indexes, 0)
                          #ic("clustid_embeddings", clustid_embeddings) 
                          #mean_embedding = clustid_embeddings.mean(axis=0)
                          #for i in range(len(clustid_embeddings)):
                          clustid_embeddings = normalize(clustid_embeddings, axis =1, norm = "l2")
                          if len(indexes) > 1:
                              cosim = cosine_similarity(clustid_embeddings)
                              upper = cosim[np.triu_indices(cosim.shape[0], k = 1)]
                              #ic(upper)
                              mean_cosim = np.mean(upper)
                          else:
                             mean_cosim = 0
                          ic(key, mean_cosim, len(indexes))
                          #ic("mean embedding", mean_embedding)
                          #new_array = np.append(new_array, mean_embedding, axis=1)






#@profile
def consolidate_w_clustering(clusters_dict, seqs_aas_dict):
    return(0) 





#
##@profile
#def squish_clusters(cluster_order, clustid_to_clust, index,hidden_states, full_cov_numseq, index_to_aa):
#    
#    '''
#    There are cases where adjacent clusters should be one cluster. 
#    If any quality scores, squish them together(tetris style)
#    XA-X  ->  XAX
#    X-AX  ->  XAX
#    XA-X  ->  XAX
#    Start with doing this at the end
#    With checks for unassigned aa's could do earlier
#    '''
#
#    removed_clustids = [] 
#    for i in range(len(cluster_order)-1):
#
#       c1 = clustid_to_clust[cluster_order[i]]
#       # skip cluster that was 2nd half of previous squish
#       if len(c1) == 0:
#         continue
#       c2 = clustid_to_clust[cluster_order[i + 1]]
#       c1_seqnums = [x.seqnum for x in c1]
#       c2_seqnums = [x.seqnum for x in c2]
#       seqnum_overlap = set(c1_seqnums).intersection(set(c2_seqnums))
#       #if len(list(seqnum_overlap)) < target_n:
#           #continue
#       # Can't merge if two clusters already have same sequences represented
#       if len(seqnum_overlap) > 0:
#          continue            
#       else:
#
#          #combo = c1 + c2
#          ## Only allow full coverage for now
#          #if len(combo) < target_n:
#          #     continue
#          intra_clust_hits= []
#          for aa1 in c1:
#            candidates = get_looser_scores(aa1, index, hidden_states)
#
#            for candidate in candidates:
#                try:
#                   score = candidate[0]
#                   candidate_index = candidate[1]
#                   target_aa = index_to_aa[candidate_index]
#                   if target_aa in c2:
#                       if score > 0.001:
#                           intra_clust_hits.append([aa1,aa2,score] )
#
#                except Exception as E:
#                   # Not all indices correspond to an aa.
#                   continue
#
#          #ic("c1", c1)
#          #ic("c2", c2)
#          combo = c1 + c2
#          scores = [x[2] for x in intra_clust_hits if x is not None]
#          # Ad hoc, get ones where multiple acceptable hits to second column
#          if len(scores) > (0.5 * len(c1) * len(c2)):
#              #ic("An acceptable squish")
#              removed_clustids.append(cluster_order[i + 1])
#              clustid_to_clust[cluster_order[i]] = combo
#              clustid_to_clust[cluster_order[i + 1]] = []
#              # If full tetris effect. 
#              # If complete, doesn't matter
#              # Change? don't worry with score?
#              #elif len(combo) == full_cov_numseq:
#                 #removed_clustids.append(cluster_order[i + 1])
#                 #clustid_to_clust[cluster_order[i]] = combo
#                 #clustid_to_clust[cluster_order[i + 1]] = []
# 
#                        
#
#    #ic("Old cluster order", cluster_order)
#    cluster_order = [x for x in cluster_order if x not in removed_clustids]
#    #ic("New cluster order", cluster_order)
#
#    return(cluster_order, clustid_to_clust)
#
#
##@profile
#def remove_doubles2(cluster, rbh_list, numseqs, minclustsize):
#    """
#    Will need to resolve ties with scores
#    """
#    seqcounts = [0] * numseqs # Will each one replicated like with [[]] * n?
#    for pos in cluster:
#       seqnum = get_seqnum(pos)
#       #ic(seq, seqnum)
#       seqcounts[seqnum] = seqcounts[seqnum] + 1
#    #doubled = [i for i in range(len(seqcounts)) if seqcounts[i] > 1]
#
#    G = igraph.Graph.TupleList(edges=rbh_list, directed = False)
#    G = G.simplify() 
# 
#    # To do: check if doing extra access hashing other places
#    for seqnum in range(len(seqcounts)):
#        if seqcounts[seqnum] > 1:
#            aas = [x for x in cluster if get_seqnum(x) == seqnum]
#            #ic(aas)       
#            degrees = []
#            for aa in aas: 
#                  degrees.append(G.degree(aa))
#                  # TODO: Get rbh to return scores
#                  # get highest score if degree tie
#                  # gap_scores.append(G
#            
#            #ic(degrees)
#            highest_degree = aas[np.argmax(degrees)]
#            to_remove = [x for x in aas if x != highest_degree]
#            cluster = [x for x in cluster if x not in to_remove]
#
#
#    if len(cluster) < minclustsize:
#         return([])
#
#    else:
#         return(cluster)
#
#
#
#
##@profile
#def remove_doubles_old(cluster, numseqs, minclustsize = 3):
#    """
#    If a cluster has two aas from the same sequence, remove from cluster
#    Also removes clusters smaller than a minimum size (default: 3)
#    Parameters:
#       clusters (list): [aa1, aa2, aa3, aa4]
#                       with aa format of sX-X-N
#       numseqs (int): Total number of sequence in alignment
#       minclustsize (int):
#    Returns:
#       filtered cluster_list 
#    Could do this with cluster orders instead
#    """
#    #clusters_filt = []
#    #for i in range(len(clusters_list)): 
#    seqcounts = [0] * numseqs # Will each one replicated like with [[]] * n?
#    for pos in cluster:
#       seqnum = get_seqnum(pos)
#       #ic(seq, seqnum)
#       seqcounts[seqnum] = seqcounts[seqnum] + 1
#    remove_list = [i for i in range(len(seqcounts)) if seqcounts[i] > 1]
#    clust = []
#    for pos in cluster:
#       seqnum =  get_seqnum(pos)
#       if seqnum in remove_list:
#          #ic("{} removed from cluster {}".format(seq, i))
#          continue
#       else:
#          clust.append(pos)
#    if len(clust) < minclustsize:
#         return([])
#
#    else:
#         return(clust)
#
#
#
#
##@profile
#def load_model_old(model_name):
#    logging.info("Load tokenizer")
#    #ic("load tokenizer")
#    tokenizer = AutoTokenizer.from_pretrained(model_name)
#
#    logging.info("Load model")
#    #ic("load model")
#    model = AutoModel.from_pretrained(model_name, output_hidden_states=True)
#
#    return(model, tokenizer)


#
##@profile
#def split_distances_to_sequence(D, I, seqnums, index_to_aa, numseqs, padded_seqlen):
#   I_tmp = []
#   D_tmp = []
#   #X_tmp = []
#   #ic(D.shape)
#   #ic(I.shape)
#   # For each amino acid...
#   for i in range(len(I)):
#      #ic(i)
#      # Make empty list of lists, one per sequence
#      I_query =  [[] for i in range(numseqs)]
#      D_query = [[] for i in range(numseqs)]
#      #X_query = [[] for i in range(numseqs)]
#     
#      # Split each amino acid's  nearest neighbors into sequences (list for seq1, list for seq2, etc)
#      for j in range(len(I[i])):
#           try:
#              aa = index_to_aa[I[i][j]]
#
#              seqnum = aa.seqnum
#              seqnum_index = seqnums.index(seqnum)
#              I_query[seqnum_index].append(aa) 
#              D_query[seqnum_index].append(D[i][j])
#              #X_query[seqnum_index].append(I[i][j])
#           except Exception as E:
#               continue
#
#      I_tmp.append(I_query)
#      D_tmp.append(D_query)
#      #X_tmp.append(X_query)
#   #ic("X_tmp", X_tmp)
#   #ic(padded_seqlen)
#   #for x in X_tmp:
#   #   ic("X_tmp X", x)
#   # Split amino acid matches into sequences 
#   D =  [D_tmp[i:i + padded_seqlen] for i in range(0, len(D_tmp), padded_seqlen)]
#   I =  [I_tmp[i:i + padded_seqlen] for i in range(0, len(I_tmp), padded_seqlen)]
#   #X =  [X_tmp[i:i + padded_seqlen] for i in range(0, len(X_tmp), padded_seqlen)]
#   #ic("X_final", X)
#   #for x in X:
#   #   ic("X final x",X)
#   #ic(D.shape)
#   #ic(I.shape)
#  
#  
#   return(D, I)
#
#
#
#
#
##@profile
#def get_besthits_old(D, I, seqnums, index_to_aa, padded_seqlen, minscore = 0.1 ):
#
#   aa_to_index = {value: key for key, value in index_to_aa.items()}
#
#   hitlist = []
#
#   #
#      #          query_seqnum = query_id.seqnum
#      #         query_seqnum_ind = seqnums.index(query_seqnum)
#      #         seqpos = query_id.seqpos
#      #         ind = I2[query_seqnum_ind][seqpos]
#      #         dist = D2[query_seqnum_ind][seqpos]
#   #ic(seqnums)
#   #ic(padded_seqlen)
#   #ic(len(D))
#   #ic(len(D[0][0]))
#   #ic(len(I[0][0]))
#   #ic(index_to_aa)
#   for query_i in range(len(D)):
#      query_seq = seqnums[query_i]
#      for query_aa in range(len(D[query_i])):
#           # Non-sequence padding isn't in dictionary
#           try:
#              query_id = index_to_aa[query_i * padded_seqlen + query_aa] 
#
#           except Exception as E:
#              #ic("exception", query_i, padded_seqlen, query_aa)
#              continue
#           for target_i in range(len(D[query_i][query_aa])):
#               target_seq = seqnums[target_i]
#               #ic(target_seq, target_i, "seq, i")
#               scores = D[query_i][query_aa][target_i]
#               if len(scores) == 0:
#                  continue
#               ids = I[query_i][query_aa][target_i]
#               #if query_seq in [4]:
#                    #ic(query_id)
#                    #ic("scores", scores)
#                    #ic("ids", ids)
#               bestscore = scores[0]
#               bestmatch_id = ids[0]
#
#               if bestscore >= minscore:
#                  hitlist.append([query_id, bestmatch_id, bestscore])
#   #for x in hitlist:
#   #    #ic("errorcheck", x)
#   return(hitlist) 
#
#
#
#    
##        for seq in target_seqs_list:
##           for query_id in seq:
##               query_seqnum = query_id.seqnum
##               query_seqnum_ind = seqnums.index(query_seqnum)
##               seqpos = query_id.seqpos
##               ind = I2[query_seqnum_ind][seqpos]
##               dist = D2[query_seqnum_ind][seqpos]    
##               for j in range(len(ind)):
#                   ids = ind[j]
#                   #ic(query_id)
#                   #scores = dist[j]
#                   good_indices = [x for x in range(len(ids)) if ids[x] in target_seqs]
#                   ids_target = [ids[g] for g in good_indices]
#                   scores_target = [dist[j][g] for g in good_indices]
#                   if len(ids_target) > 0:
#                       bestscore = scores_target[0]
#                       bestmatch_id = ids_target[0]
#                       if query_seqnum == bestmatch_id.seqnum:
#                            continue
#                       if bestmatch_id in target_seqs:
#                           if bestscore >= minscore:
#                              new_hitlist.append([query_id, bestmatch_id, bestscore])#, pos_to_clustid[bestmatch_id]])
#

#
##@profile
#def doubles_in_clust(clust):
#    seen = []
#    #doubled = []
#    doubled_seqnums = []
#    for pos in clust:
#       if pos.seqnum not in seen:
#          seen.append(pos.seqnum)
#       else:
#          #doubled.append(pos)
#          doubled_seqnums.append(pos.seqnum)
#    doubled_seqnums = list(set(doubled_seqnums))
#    return(doubled_seqnums)
#
## maybe for 
#
#
##@profile
#def remove_doubles_deprecated(clustid_to_clust, index, hidden_states, index_to_aa):
#    for clustnum, clust in clustid_to_clust.items():
#        doubled_seqnums = doubles_in_clust(clust)
#        if doubled_seqnums:
#    #ic(clustid_to_clust) 
#
#    return(cluster_order2, clustid_to_clust)
#
##
#
#
##@profile
#def fill_in_unassigned_w_clustering_old(unassigned, seqs_aas, cluster_order, clustid_to_clust, pos_to_clustid, I2,  minscore = 0.1, minclustsize = 2, ignore_betweenness = False, betweenness_cutoff = 0.3, apply_walktrap = True ):        
#    '''
#    Run the same original clustering, ??? allows overwritting of previous clusters
#    
#    So there's a situation where the gap range contains a different amino acid that itself has a different gap range, with a much higher score. 
#    Maybe do larger first?
#    No, do network to find ranges for allxall search. 
#    If a new member of an rbh cluster has a large unassigned range, check if has higher rbh t o sequence?
#    '''
#    clusters_filt = list(clustid_to_clust.values())
#    ic("TESTING OUT CLUSTERS_FILT")
#    for x in clusters_filt:
#        ic("preassignment clusters_filt", x)
#    # extra arguments?
#    new_clusters = []
#  
#    newer_rbhs = []
#    for gap in unassigned:
#        newer_clusters, newer_rbh = address_unassigned(gap, seqs_aas, pos_to_clustid, cluster_order, clustid_to_clust,  I2,  minscore = minscore, ignore_betweenness = ignore_betweenness, betweenness_cutoff = betweenness_cutoff, minclustsize = minclustsize, apply_walktrap = apply_walktrap)
#
#        new_clusters  = new_clusters + newer_clusters
#        newer_rbhs = newer_rbhs + newer_rbh
#
#    unique_rbhs = [list(x) for x in set(tuple(x) for x in newer_rbhs)]
#
#    G2 = graph_from_rbh(unique_rbhs)
#    new_clusters_list = dedup_clusters(new_clusters, G2, minclustsize)
#
#    # Going to do an additional step
#    # If a cluster has doubles
#    # See if there is overlap with another cluster
#    # See if removing the overlap amino acids resolves the doubles
#    #
#    too_small = []
#    new_clusters = []
#    for clust in new_clusters_list:
#          if len(clust) >= minclustsize:
#                new_clusters.append(clust)
#          else:
#             # This is neever happening?
#             if len(clust) > 1:
#                too_small.append(clust)
#
#
#
#    # Remove small clusters
#    #new_clusters = [x for x in clusters_list if len(x) >= minclustsize]
#
#    for x in new_clusters:
#        ic("All new clusters", x)
#
#    #ic("New clusters:", new_clusters)
#    # Very important to removeSublist here
#    new_clusters = removeSublist(new_clusters)
#    # Need to remove overlaps from clusters
#    # Get amino acids in more than one cluster, remove them. 
#
#    #ic("New clusters after sublist removal",  new_clusters) 
#    aa_counter = {}
#    new_clusters_flat  = flatten(new_clusters) 
#    #ic("flat_clusters", new_clusters_flat)
#    aa_counts = Counter(new_clusters_flat)
#    dupped_aas = {key for key, val in aa_counts.items() if val != 1}
#    ic("dupped aas", dupped_aas)
#
#    # From doubled aas from clusters list of lists
#    new_clusters = [[aa for aa in clust if aa not in dupped_aas] for clust in new_clusters]
#        
#
#    # If this makes clusters too small remove them
# 
#    new_clusters = [clust for clust in new_clusters if len(clust) >= minclustsize]
#
#    ic("WHAT is minclustsize", minclustsize)
#    for x in new_clusters:
#        ic("All new new clusters", x)
#
#    #ic("New clusters after overlap removal",  new_clusters) 
#    # Due to additional walktrap, there's always a change that a new cluster won't be entirely consistent with previous clusters. 
#    # In this section, remove any members of a new cluster that would bridge between previous clusters and cause over collapse
#    #ic(pos_to_clustid)
#    new_clusters_filt = []
#    for clust in new_clusters:
#         clustids = []
#         posids = []
#         new_additions = []
#         for pos in clust:      
#            #ic(pos)
#            if pos in pos_to_clustid.keys():
#               clustid = pos_to_clustid[pos]
#               clustids.append(clustid)
#               posids.append(pos)
#               ic(pos, clustid)
#            else:
#                # Position wasn't previously clustered
#                ic("new_additions", clust,pos)
#                new_additions.append(pos)
#         #ic("posids", posids)                  
#         if len(list(set(clustids))) > 1:
#            #ic("new cluster contains component of multiple previous clusters. Keeping largest matched cluster")
#            clustcounts = Counter(clustids)
#            largest_clust = max(clustcounts, key=clustcounts.get)   
#            sel_pos = [posids[x] for x in range(len(posids)) if clustids[x] == largest_clust]
#            #ic("Split cluster catch", clustcounts, largest_clust, posids, clustids, sel_pos)
#            new_clust = sel_pos + new_additions
#                
#         else:
#            new_clusters_filt.append(clust)             
#
#    # T0o much merging happening
#    # See s4-0-I, s4-1-L in cluster 19 of 0-60 ribo
#
#    new_clusters_filt = removeSublist(new_clusters_filt)
#
#    clusters_new = remove_overlap_with_old_clusters(new_clusters_filt, clusters_filt)
#    clusters_merged = clusters_new + clusters_filt
#
#    for x in clusters_merged:
#       ic("clusters_merged", x)
#    cluster_order, clustid_to_clust, pos_to_clustid, alignment = organize_clusters(clusters_merged, seqs_aas, minclustsize)
#    return(cluster_order, clustid_to_clust, pos_to_clustid, alignment, too_small)


#
##@profile
#def address_stranded2(alignment):
#    cluster_order, clustid_to_clust = clusts_from_alignment(alignment)
#    to_remove =[]
#    clustered_aas = list(flatten(clustid_to_clust.values())) 
#    new_cluster_order = []
#    new_clustid_to_clust = {}
#    ic(clustered_aas)
#    for clustid in cluster_order:
#         clust = clustid_to_clust[clustid]
#         ic("check for stranding", clust)
#         prevaa_clustered = [x for x in clust if x.prevaa in clustered_aas]
#         nextaa_clustered = [x for x in clust if x.nextaa in clustered_aas]
#         ic(prevaa_clustered, nextaa_clustered)
#         if len(prevaa_clustered) > 0 or len(nextaa_clustered) > 0:
#             new_cluster_order.append(clustid)
#             new_clustid_to_clust[clustid] = clust
#         else:
#             ic("Found stranding, Removing stranded clust", clust) 
#    return(new_cluster_order, new_clustid_to_clust)
#
#
#
##@profile
#def address_stranded(alignment):
#    cluster_order, clustid_to_clust = clusts_from_alignment(alignment)
#    to_remove =[]
#    maxkey =  max(clustid_to_clust.keys())
#    new_clustid_to_clust = {}
#    new_cluster_order = []
#    for clustid in cluster_order:
#
#       clust = clustid_to_clust[clustid]
##    for clustid, clust in clustid_to_clust.items():
#       firstgap = False
#       secondgap = False
#       newclust = []
#       for aa in clust:
#           firstgap = False
#           secondgap = False
#
#           if clustid == 0:
#             firstgap = True
#           if clustid == maxkey:
#             secondgap = True
#           
#           if clustid > 0:
#               if aa.seqnum not in [x.seqnum for x in clustid_to_clust[clustid - 1]]:
#                  firstgap = True
#
#           if clustid < maxkey:
#               if aa.seqnum not in [x.seqnum for x in clustid_to_clust[clustid + 1]]:
#                  secondgap = True
#           
#           if firstgap == True and secondgap == True:
#                 to_remove.append(aa)
#
#           else:
#              newclust.append(aa)
#       if len(newclust) > 0:
#          new_clustid_to_clust[clustid] = newclust 
#          new_cluster_order.append(clustid)
#
#
#
#    ic("stranded", to_remove)
#    return(new_cluster_order, new_clustid_to_clust)
##    adict = {}
##    for i in range(len(alignment)):
##       seq = alignment[i]
##       current_streak = 0
##       prev_streak = 0
##       #ic(seq)
##       last_i = -1 
##       last_pos = seq[0]
##       for j in range(len(seq)):
##            clustid = j
##            pos = seq[j]
##            #ic("lastpos", last_pos)
##            #ic(pos, current_streak, prev_streak)
##            if pos == "-":
##               current_streak = current_streak + 1
##            else:
##               if current_streak > 0 and last_i ==0 :
##                  if last_pos != "-":
##                      #ic("last_pos", last_pos)
##                      adict[last_pos] = [0, current_streak]
##               if prev_streak > 0 & current_streak > 0:
##                  adict[last_pos] = [prev_streak, current_streak]
##               prev_streak = current_streak
##               current_streak = 0                      
##               last_pos = pos
##               last_i = i
##               #firstpos  = False  
##
##    ic("stranded") 
##    ic(adict)
# 
#    
#
#    #for i in range(len(alignment[0])):
#       
#
#
#    # Start of "in a gap" 
#    # To address where first character placed then gap
#    
#    #for i in range(len(cluster_order) - 1):
#    #  c1 = clustid_to_clust[cluster_order[i]]
#      # contains gap
#    #  if len(c1) < full_cov_numseq:
#            # if contains gap, see which sequence has gap
#
#    #        #ic("x")            
    #return(0)

## This is in the goal of finding sequences that poorly match before aligning
## SEQSIM
#
##@profile
#def graph_from_distindex(index, dist, seqnames, seqsim_thresh = 0):  
#
#    edges = []
#    weights = []
#    complete = []
#    for i in range(len(index)):
#       #complete.append(i)
#       for j in range(len(index[i])):
#          #if j in complete:
#          #    continue
#          # Index should not return negative
#          weight = dist[i,j]
#          if weight < 0:
#             #ic("Index {}, {} returned negative similarity, replacing with 0.001".format(i,j))
#             weight = 0.001
#          edge = (i, index[i, j])
#          #if edge not in order_edges:
#          # Break up highly connected networks, simplify clustering
#          if weight >= seqsim_thresh:
#              edges.append(edge)
#              weights.append(weight)
#
#    with open("seqsim.txt", "w") as outfile:
#        for i in range(len(edges)):
#          ic("seqsim ", edges[i], weights[i])
#          outfile.write("{},{},{}".format(seqnames[edges[i]], seqnames[edges[i]], weights[i]))      
# 
#    G = igraph.Graph.TupleList(edges=edges, directed=False)
#    G.es['weight'] = weights
#    return(G)
##
### If removing a protein leads to less of a drop in total edgeweight that other proteins
##
#
##@profile
#def remove_order_conflicts2(cluster_order, seqs_aas,numseqs, pos_to_clustid):
#    """ 
#    After topological sort,
#    remove any clusters that conflict with sequence order 
#    """
#    #ic("pos_to_clustid", pos_to_clustid)   
#    #ic("cluster-order remove_order_conflict", cluster_order)  
#    clusters_w_order_conflict= []
#    for i in range(numseqs): 
#        prev_cluster = 0
#        for j in range(len(seqs_aas[i])):
#           key = seqs_aas[i][j]
#           try:
#               clust = pos_to_clustid[key]
#           except Exception as E:
#               continue
#
#           order_index = cluster_order.index(clust)
#           #ic(key, clust, order_index)
#           if order_index < prev_cluster:
#                clusters_w_order_conflict.append(clust)
#                #ic("order_violation", order_index, clust)
#           prev_cluster  = order_index
#    #ic(cluster_order)
#    #ic(clusters_w_order_conflict)
#    cluster_order = [x for x in cluster_order if x not in clusters_w_order_conflict]
#    return(cluster_order)
#def fill_in_unassigned2(unassigned, seqs, seqs_aas, G, clustid_to_clust):
#    '''
#    Try group based assignment
#    Decision between old cluster and new cluster?
#    
#    '''
#    clusters = list(clustid_to_clust.values())
#
#    #ic("unassigned")
#    #for x in unassigned:
#       #ic("unassigned", x)
#    numclusts = []
#    #new_clusters = []
#    matches = []
#    unassigned = format_gaps(unassigned, max(clustid_to_clust.keys())) 
#
#    for gap in unassigned:
#
#        starting_clustid =  gap[0]
#        ending_clustid = gap[2]
#        #if gap[1][0].seqnum in to_exclude:
#        #    #ic(gap[1], gap[1][0].seqnum, "exclude", to_exclude)
#        #    continue
#        for gap_aa in gap[1]:
#            gap_aa_cluster_max = []
#            scores = []
#             
#            unassigned_index = G.vs.find(name = gap_aa).index
#            #ic("unassigned index", unassigned_index) 
#            connections = G.es.select(_source = unassigned_index)
#            scores = connections['weight']
#      
#            connected_index  = [x.target for x in connections]
#            #ic(connected_index)
#            connected_aas = [x.target_vertex['name'] for x in connections]
#            # Source vs target seems random, potentially make graph directed and unsimplified
#            if gap_aa in connected_aas:
#                connected_aas = [x.source_vertex['name'] for x in connections]
#
#            #ic("unassigned", gap_aa, starting_clustid, ending_clustid)
#            #ic(connected_aas)
#            #ic(scores)  
#            both = list(zip(connected_aas, scores))
#            #ic(both)
#            output = get_best_matches(starting_clustid, ending_clustid, gap_aa, clustid_to_clust, both)
#            if output:
#               matches.append(output)
#
#        
#    clustid_to_clust = get_best_of_matches(clustid_to_clust, matches)
#
#    return(clustid_to_clust)
#            # Get all edges from gap aa
#            # Then sort out to groups 
#
#    # For each cluster, only add in the top scoring from each sequence
#
##@profile
#def address_unassigned(gap, seqs_aas, pos_to_clustid, cluster_order, clustid_to_clust,  I2, minscore = 0.5, ignore_betweenness = False, betweenness_cutoff = 0.30, minclustsize = 2, apply_walktrap = True):
#        # This is allowed to disturb prior clusters
#        # Get a range of sequences between guidepost (guidepost inclusive?)
#        new_clusters = []
#        starting_clustid = gap[0]
#        ending_clustid = gap[2] 
#        gap_seqaas = gap[1]
#        #gap_seqnum = list(set([x.seqnum for x in gap_seqaas]))[0]
#        target_aas_list = get_ranges(seqs_aas, cluster_order, starting_clustid, ending_clustid, pos_to_clustid)
#        #if "3-2-Y" in gap_seqaas:
#        #      ic("HERE")
#
#        target_aas = list(flatten(target_aas_list))
#        target_aas = list(set(target_aas + gap_seqaas))
#
#        #ic("For each of the unassigned seqs, get their top hits from the previously computed distances/indices")
#         
#        ic("gap", gap_seqaas)
#        ic("targets", target_aas)
#        # Only get query_aas that are in the target_seqs
#        #ic(I2['3-2-Y'])
#        #ic("I2 in loop")
#        #for key, value in I2.items():
#        #     ic(key, value) 
# 
#        limited_I2 = {}
#        #tmp_I2= copy.deepcopy(I2)#  {key: value for key, value in I2.items()}
#        # Suspect that this is slow
#        for key in I2.keys():
#           if key in target_aas:
#               limited_I2[key] = I2[key].copy()#copy.deepcopy(I2[key]) 
#        ic("limited")
#        ic("new_rbh_minscore", minscore)
#        #for key, value in tmp_I2.items():
#        #     ic(key, value) 
#        
#        #limited_I2 = { your_key: tmp_I2[your_key] for your_key in target_aas }
#        
#        #for key in tmp_I2.keys():
#        #   if key in target_aas:
#        #       limited_I2[key] = tmp_I2[key]
#          
#        #for key, value in limited_I2.items():
#        #     ic(key, value) 
#        #ic('hey') 
#        for query_aa in limited_I2.keys():
#             
#             # These keys 
#             for seq in limited_I2[query_aa].keys():
#                   
#                   limited_I2[query_aa][seq] = [x for x in limited_I2[query_aa][seq] if x[0] in target_aas]
#
#        #ic("Postfilter limited I2", limited_I2)
#
#       
#        # Get reciprocal best hits in a limited range
#        new_hitlist = get_besthits(limited_I2, minscore)
#
#        #for x in new_hitlist:
#        #   ic("new_hitlist", x)
# 
# 
#        new_rbh = get_rbhs(new_hitlist)
#        #ic("query_aa", query_aa)
#        for x in new_rbh:
#             ic("new_rbh", x)
#        G = graph_from_rbh(new_rbh) 
#        new_clusters  = first_clustering(G, betweenness_cutoff = betweenness_cutoff,  ignore_betweenness = ignore_betweenness, apply_walktrap = apply_walktrap ) 
#     
#        for x in new_clusters:
#            ic("new_clusters", x)
# 
#        #new_clusters = removeSublist(new_clusters)
#        clustered_aas = list(flatten(new_clusters))
#
#
#        #unmatched = [x for x in gap_seqaas if not x in clustered_aas]     
#        #hopelessly_unmatched  = []
#        #This means that's it's the last cycle
#        #if ignore_betweenness == True:
#        #  
#           # If no reciprocal best hits
#           
#           #for aa in unmatched: 
#           #      #ic("its index, ", aa.index)
#           #      hopelessly_unmatched.append(aa)
#                 #new_clusters.append([aa])
#
#        return(new_clusters, new_rbh)
#
#def get_seq_groups2(seqs, seq_names, embedding_dict, logging, exclude, do_clustering, seqsim_thresh= 0.75):
#    numseqs = len(seqs)
#
#    
#    #hstates_list, sentence_embeddings = get_hidden_states(seqs, model, tokenizer, layers, return_sentence = True)
#    #logging.info("Hidden states complete")
#    #ic("end hidden states")
#
#    #if padding:
#    #    logging.info("Removing {} characters of neutral padding X".format(padding))
#    #    hstates_list = hstates_list[:,padding:-padding,:]
#
#    to_exclude = []
#
#    G = get_seqsims(seqs, embedding_dict, seqsim_thresh = seqsim_thresh)
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
#        #ic("fastgreedy")
#        #ic(G)
#    
#      #repeat = True
#      #
#      #while repeat == True:
#      d = sentence_array.shape[1]
#      for k in range(1, 20):
#         kmeans = faiss.Kmeans(d = d, k = k, niter = 20)
#         kmeans.train(sentence_array)
#    
#   
#         D, I = kmeans.index.search(sentence_array, 1) 
#         ic("D", D)
#         ic("I", I)
#         clusters = I.squeeze()
#         labels = list(zip(G.vs()['name'], clusters))
#         #for x in labels:
#         #    ic("labels", x[0], x[1])
#
#
#         group_hstates_list = []
#         cluster_seqnums_list = []
#         cluster_names_list = []
#         cluster_seqs_list = []
# 
#         prev_to_exclude = to_exclude
#        
#         means = []
#         for clustid in list(set(clusters)):
#             ic("eval clust", clustid)
#             clust_seqs = [x[0] for x in labels if x[1] == clustid] 
#             ic("clust_seqs", clust_seqs)
#             #ic("labels from loop", labels)
#             #for lab in labels:
#             #     ic("labels", lab, lab[0], lab[1], clustid) 
#             #     if lab[1] == clustid:
#             # 
#              #         ic("yes")
#             #ic("GG", G.vs()['name'])
#             #ic("GG", G.es()['weight'])
#             #edgelist = []
#             weightlist = []
#             for edge in G.es():
#                  #ic(edge, edge['weight'])
#                  #ic(G.vs[edge.target]["name"], G.vs[edge.source]["name"])
#                  if G.vs[edge.target]["name"] in clust_seqs:
#                       if G.vs[edge.source]["name"] in clust_seqs:
#                          weightlist.append(edge['weight'])
#                          ic(G.vs[edge.target]["name"], G.vs[edge.source]["name"], edge['weight'])
#             ic(weightlist)
#             ic("clust {} mean {}".format(clustid, np.mean(weightlist)))
#             means.append(np.mean(weightlist))
#         ic("k {} overall mean {}".format(clustid, np.mean(means)))    
#
#      #return(0)
#
    #hstates_heads = hstates_list.reshape(numseqs, padded_seqlen, -1, 64)

    #if headnorm == True:
    #    hstates_heads = hstates_list.reshape(-1, 64)
    #    ic(hstates_heads.shape)
   # 
   #     hstates_heads = normalize(hstates_heads, axis =1, norm = "l2")
   # 
   # 
   #     hstates_list = hstates_heads.reshape(numseqs, padded_seqlen, embedding_length)
   #     ic(hstates_list.shape)

