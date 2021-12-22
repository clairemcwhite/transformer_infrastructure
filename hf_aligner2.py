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
#from Bio.Seq import Seq
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

class AA:
   def __init__(self):
       self.seqnum = ""
       self.seqindex = ""
       self.seqpos = ""
       self.seqaa = ""
       self.index = ""
       self.clustid = ""

   #__str__ and __repr__ are for pretty #printing
   def __str__(self):
        return("{}-{}-{}".format(self.seqnum, self.seqpos, self.seqaa))

   def __repr__(self):
    return str(self)
 

# This is in the goal of finding sequences that poorly match before aligning
# SEQSIM
def graph_from_distindex(index, dist):  

    edges = []
    weights = []
    for i in range(len(index)):
       for j in range(len(index[i])):
          # Index should not return negative
          weight = dist[i,j]
          if weight < 0:
             #print("Index {}, {} returned negative similarity, replacing with 0.001".format(i,j))
             weight = 0.001
          edge = (i, index[i, j])
          #if edge not in order_edges:
          # Break up highly connected networks, simplify clustering
          if weight > 0.7:
              edges.append(edge)
              weights.append(weight)

     
    for i in range(len(edges)):
      print("seqsim ", edges[i], weights[i])
       
    G = igraph.Graph.TupleList(edges=edges, directed=False)
    G.es['weight'] = weights
    return(G)

# If removing a protein leads to less of a drop in total edgeweight that other proteins

def candidate_to_remove(G, v_names,z = -3):


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

def graph_from_cluster_orders(cluster_orders_lol):

    order_edges = []
    for order in cluster_orders_lol:
       for i in range(len(order) - 1):
          edge = (order[i], order[i + 1])
          #if edge not in order_edges:
          order_edges.append(edge)
          
          #print(edge)

    G_order = igraph.Graph.TupleList(edges=order_edges, directed=True)
    return(G_order, order_edges)

def get_topological_sort(cluster_orders_lol):
    #print("start topological sort")
    cluster_orders_nonempty = [x for x in cluster_orders_lol if len(x) > 0]
    #with open("tester2.txt", "w") as f:
    #   for x in cluster_orders_nonempty:
    #      f.write("{}\n". format(x))

    
    dag_or_not = graph_from_cluster_orders(cluster_orders_nonempty)[0].simplify().is_dag()
    # 
    

    #print ("Dag or Not?, dag check immediately before topogical sort", dag_or_not)
  
    G_order = graph_from_cluster_orders(cluster_orders_nonempty)[0]
    G_order = G_order.simplify()

    topo_sort_indices = G_order.topological_sorting()
    cluster_order = []

    # Note: this is in vertex indices. Need to convert to name to get clustid
    for i in topo_sort_indices:
       cluster_order.append(G_order.vs[i]['name'])
    return(cluster_order) #, clustid_to_clust_dag)

def remove_order_conflicts(cluster_order, seqs_aas, pos_to_clustid):
   #print("remove_order_conflicts, before: ", cluster_order)
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
              #print("Order violation", posid, clustid)
              bad_clustids.append(clustid)
   cluster_order =  [x for x in cluster_order if x not in bad_clustids]
   return(cluster_order)
def remove_order_conflicts2(cluster_order, seqs_aas,numseqs, pos_to_clustid):
    """ 
    After topological sort,
    remove any clusters that conflict with sequence order 
    """
    #print("pos_to_clustid", pos_to_clustid)   
    #print("cluster-order remove_order_conflict", cluster_order)  
    clusters_w_order_conflict= []
    for i in range(numseqs): 
        prev_cluster = 0
        for j in range(len(seqs_aas[i])):
           key = seqs_aas[i][j]
           try:
               clust = pos_to_clustid[key]
           except Exception as E:
               continue

           order_index = cluster_order.index(clust)
           #print(key, clust, order_index)
           if order_index < prev_cluster:
                clusters_w_order_conflict.append(clust)
                #print("order_violation", order_index, clust)
           prev_cluster  = order_index
    #print(cluster_order)
    #print(clusters_w_order_conflict)
    cluster_order = [x for x in cluster_order if x not in clusters_w_order_conflict]
    return(cluster_order)
 
def make_alignment(cluster_order, seqnums, clustid_to_clust):
    # Set up a bunch of vectors of "-"
    # Replace with matches
    # cluster_order = list in the order that clusters go
    print("Alignment clusters")
    for clustid, clust in clustid_to_clust.items():
        print(clustid, clust)

    numseqs = len(seqnums)
    alignment =  [["-"] * len(cluster_order) for i in range(numseqs)]
    #print(cluster_order)
   # #print("test cluster order", cluster_order)
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
                  alignment[seqnum_index][order] = c_dict[seqnum]
               except Exception as E:
                   continue
    alignment_str = ""
    print("Alignment")

    str_alignment = obj_aln_to_str(alignment)
    for row_str in str_alignment: 
       print("Align: ", row_str[0:150])
        
    return(alignment)

def obj_aln_to_str(alignment):
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

def alignment_print(alignment, seq_names):
       
        records = []
        #alignment = ["".join(x) for x in alignment]
        alignment = obj_aln_to_str(alignment)
              
        for i in range(len(alignment)):
             #print(seq_names[i], alignment[i])
             #print(alignment[i], seq_names[i])
             records.append(SeqRecord(Seq(alignment[i]), id=seq_names[i]))
        align = MultipleSeqAlignment(records)
        clustal_form = format(align, 'clustal')
        fasta_form = format(align, 'fasta')
        return(clustal_form, fasta_form)


def get_ranges(seqs_aas, cluster_order, starting_clustid, ending_clustid, pos_to_clustid):
    #print("start get ranges")
  
    #print(cluster_order, starting_clustid, ending_clustid) 
    print('start, end', starting_clustid, ending_clustid)
    # if not x evaluates to true if x is zero 
    # If unassigned sequence goes to the end of the sequence
    if not ending_clustid and ending_clustid != 0:
       ending_clustid = np.inf    
    # If unassigned sequence extends before the sequence
    if not starting_clustid and starting_clustid != 0:
       starting_clustid = -np.inf   
 
    # cluster_order must be zero:n
    # Add assertion
    pos_lists = []
    for x in seqs_aas:
            pos_list = []
            startfound = False

            # If no starting clustid, add sequence until hit ending_clustid
            if starting_clustid == -np.inf:
                 startfound = True
                
            prevclust = "" 
            for pos in x:
                try: 
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
                        #print(pos_clust, starting_clustid, ending_clustid)
                except Exception as E:
                        #print(startfound, "exception", pos, prevclust, starting_clustid, ending_clustid)
                        if startfound == True or prevclust == cluster_order[-1]:
                           if prevclust:
                               if prevclust >= starting_clustid and prevclust <= ending_clustid:    
                                   pos_list.append(pos)
                           else:
                              pos_list.append(pos)
                         


            pos_lists.append(pos_list)
    return(pos_lists)





def get_unassigned_aas(seqs_aas, pos_to_clustid):
    ''' 
    Get amino acids that aren't in a sequence
    '''
    #print(pos_to_clustid)
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
  
           try:
              # Read to first cluster hit
              clust = pos_to_clustid[key]
              prevclust = clust
           # If it's not in a clust, it's unsorted
           except Exception as E:
              unsorted = []
              unsorted.append(key)
              for k in range(j + 1, len(seqs_aas[i])):
                  key = seqs_aas[i][k]
                  try:
                     nextclust = pos_to_clustid[key]
                     #print(nextclust)
                     break
                  # Go until you hit next clust or end of seq
                  except Exception as E:
                     unsorted.append(key)
                     last_unsorted = k
              unassigned.append([prevclust, unsorted, nextclust, i])
              nextclust = []
              prevclust = []
    return(unassigned)






def address_unassigned(gap, seqs, seqs_aas, seqnums, pos_to_clustid, cluster_order, clustid_to_clust, numseqs, I2, minscore = 0.1, ignore_betweenness = False, betweenness_cutoff = 0.10, minclustsize = 2, apply_walktrap = True):
        # This is allowed to disturb prior clusters
        # Get a range of sequences between guidepost (guidepost inclusive?)
        new_clusters = []
        starting_clustid = gap[0]
        ending_clustid = gap[2] 
        gap_seqaas = gap[1]
        #gap_seqnum = list(set([x.seqnum for x in gap_seqaas]))[0]
        target_aas_list = get_ranges(seqs_aas, cluster_order, starting_clustid, ending_clustid, pos_to_clustid)
        #if "3-2-Y" in gap_seqaas:
        #      print("HERE")

        target_aas = list(flatten(target_aas_list))
        target_aas = list(set(target_aas + gap_seqaas))

        #print("For each of the unassigned seqs, get their top hits from the previously computed distances/indices")
         
        print("gap", gap_seqaas)
        print("targets", target_aas)
        # Only get query_aas that are in the target_seqs
        #print(I2['3-2-Y'])
        #print("I2 in loop")
        #for key, value in I2.items():
        #     print(key, value) 
 
        limited_I2 = {}
        #tmp_I2= copy.deepcopy(I2)#  {key: value for key, value in I2.items()}
        # Suspect that this is slow
        for key in I2.keys():
           if key in target_aas:
               limited_I2[key] = I2[key].copy()#copy.deepcopy(I2[key]) 
        print("limited")
        #for key, value in tmp_I2.items():
        #     print(key, value) 
        
        #limited_I2 = { your_key: tmp_I2[your_key] for your_key in target_aas }
        
        #for key in tmp_I2.keys():
        #   if key in target_aas:
        #       limited_I2[key] = tmp_I2[key]
          
        #for key, value in limited_I2.items():
        #     print(key, value) 
        #print('hey') 
        for query_aa in limited_I2.keys():
             
             # These keys 
             for seq in limited_I2[query_aa].keys():
                   
                   limited_I2[query_aa][seq] = [x for x in limited_I2[query_aa][seq] if x[0] in target_aas]

        #print("Postfilter limited I2", limited_I2)

        

        # Get reciprocal best hits in a limited range
        new_hitlist = get_besthits(limited_I2)

        #for x in new_hitlist:
        #   print("new_hitlist", x)
 
 
        new_rbh = get_rbhs(new_hitlist)
        #print("query_aa", query_aa)
        #for x in new_rbh:
        #     print("new_rbh", x)
        G = graph_from_rbh(new_rbh) 
        new_clusters  = first_clustering(G, betweenness_cutoff = betweenness_cutoff, minclustsize = minclustsize,  ignore_betweenness = ignore_betweenness, apply_walktrap = apply_walktrap ) 
     
        for x in new_clusters:
            print("new_clusters", x)
 
        #new_clusters = removeSublist(new_clusters)
        clustered_aas = list(flatten(new_clusters))


        #unmatched = [x for x in gap_seqaas if not x in clustered_aas]     
        #hopelessly_unmatched  = []
        #This means that's it's the last cycle
        #if ignore_betweenness == True:
        #  
           # If no reciprocal best hits
           
           #for aa in unmatched: 
           #      #print("its index, ", aa.index)
           #      hopelessly_unmatched.append(aa)
                 #new_clusters.append([aa])

        return(new_clusters, new_rbh)

def get_looser_scores(aa, index, hidden_states):
     '''Get all scores with a particular amino acid''' 
     hidden_state_aa = np.take(hidden_states, [aa.index], axis = 0)
     # Search the total number of amino acids
     n_aa = hidden_states.shape[0]
     #index.nprobe = 100
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
        #print(D.shape)
        #print(aa1.index)
        #print(aa2.index)
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




def address_isolated_aas(unassigned_aa, cohort_aas, D, I, minscore):
    '''
    Maybe overwrite score? 
    Or match to cluster with higher degree
    '''
    #print("Address isolated aas")
    connections = []
    for cohort_aa in cohort_aas:
        score = get_particular_score(unassigned_aa, cohort_aa, D, I)
        #print(unassigned_aa, cohort_aa, score)
 
    return(cluster)


def clusts_from_alignment(alignment):
   # Pass alignment object around. 
   # Contains both cluster order and clustid_to_clust info
   clustid_to_clust = {}
   align_length = len(alignment[0])

   cluster_order = range(0, align_length)
   for i in cluster_order:
       clust = [x[i] for x in alignment if not x[i] == "-"]

       clustid_to_clust[i] = clust 

   return(cluster_order, clustid_to_clust)


def address_stranded(alignment):
    cluster_order, clustid_to_clust = clusts_from_alignment(alignment)
    to_remove =[]
    maxkey =  max(clustid_to_clust.keys())
    new_clustid_to_clust = {}
    new_cluster_order = []
    for clustid in cluster_order:

       clust = clustid_to_clust[clustid]
#    for clustid, clust in clustid_to_clust.items():
       firstgap = False
       secondgap = False
       newclust = []
       for aa in clust:
           firstgap = False
           secondgap = False

           if clustid == 0:
             firstgap = True
           if clustid == maxkey:
             secondgap = True
           
           if clustid > 0:
               if aa.seqnum not in [x.seqnum for x in clustid_to_clust[clustid - 1]]:
                  firstgap = True

           if clustid < maxkey:
               if aa.seqnum not in [x.seqnum for x in clustid_to_clust[clustid + 1]]:
                  secondgap = True
           
           if firstgap == True and secondgap == True:
                 to_remove.append(aa)

           else:
              newclust.append(aa)
       if len(newclust) > 0:
          new_clustid_to_clust[clustid] = newclust 
          new_cluster_order.append(clustid)



    print("stranded", to_remove)
    return(new_cluster_order, new_clustid_to_clust)
#    adict = {}
#    for i in range(len(alignment)):
#       seq = alignment[i]
#       current_streak = 0
#       prev_streak = 0
#       #print(seq)
#       last_i = -1 
#       last_pos = seq[0]
#       for j in range(len(seq)):
#            clustid = j
#            pos = seq[j]
#            #print("lastpos", last_pos)
#            #print(pos, current_streak, prev_streak)
#            if pos == "-":
#               current_streak = current_streak + 1
#            else:
#               if current_streak > 0 and last_i ==0 :
#                  if last_pos != "-":
#                      #print("last_pos", last_pos)
#                      adict[last_pos] = [0, current_streak]
#               if prev_streak > 0 & current_streak > 0:
#                  adict[last_pos] = [prev_streak, current_streak]
#               prev_streak = current_streak
#               current_streak = 0                      
#               last_pos = pos
#               last_i = i
#               #firstpos  = False  
#
#    print("stranded") 
#    print(adict)
 
    

    #for i in range(len(alignment[0])):
       


    # Start of "in a gap" 
    # To address where first character placed then gap
    
    #for i in range(len(cluster_order) - 1):
    #  c1 = clustid_to_clust[cluster_order[i]]
      # contains gap
    #  if len(c1) < full_cov_numseq:
            # if contains gap, see which sequence has gap

    #        #print("x")            
    #return(0)

def squish_clusters_longrange2(alignment):

    cluster_order, clustid_to_clust = clusts_from_alignment(alignment)
    clustid_to_clust = {} # build up
    merged = []
    # Don't do it this way? what is sequence has not assigned amino acids? Will have been placed at this point though
    all_seqnums = list(set([x.seqnum for x in list(flatten(alignment)) if not x == "-"]))
    #print("All seqnums", all_seqnums)
    #print(cluster_order)
    cluster_order2 = []
    for i in cluster_order:
       if i in merged:
           continue

       clust = [x[i] for x in alignment]
       clust_aas_i = [x for x in clust if not x == "-"]
       clust_seqnums_i = [x.seqnum for x in clust_aas_i]
       clust_gaps_i = [x for x in all_seqnums if x not in clust_seqnums_i]
       #print("order {}, cluster {}".format(i, clust_aas_i)) 


       if "-" in clust:
           #print("clust {}: {} has gaps".format(i, clust))
           avail_gaps = {}
           dist_blocks = {}
           bound_found = {}
           for seq in clust_seqnums_i:
                 avail_gaps[seq] = 0
                 dist_blocks[seq] = 1
                 bound_found[seq] = False

           max_clustid = len(cluster_order)
           present_gaps = len(clust_gaps_i)
           all_block_aas = []

           # scan forward counting stretches and gaps
           all_block_aas = []
           for j in range(i + 1, len(cluster_order)):
                  
                  if not False in bound_found.values():
                      #print("All boundaries found")
                      break

                   
                  clust_aas_j = [x[j] for x in alignment if not x[j] == "-"] 
                  #print("clust_aas_i", clust_aas_i)
                  #print("clust_aas_j", clust_aas_j)
                  clust_seqnums_j = [x.seqnum for x in clust_aas_j]
                  clust_gaps_j = [x for x in all_seqnums if x not in clust_seqnums_j]
                  # Could do squish at this point by testing if clust_gaps_i == clust_seqnums_j
                  for seq in clust_seqnums_i:
                        
                        if seq in clust_seqnums_j:
                           if not bound_found[seq]:
                               dist_blocks[seq] = dist_blocks[seq] + 1 
                               all_block_aas = all_block_aas  + [x for x in clust_aas_j if x.seqnum == seq]
                        if seq in clust_gaps_j:
                               avail_gaps[seq] = avail_gaps[seq] + 1     
                               present_gaps = present_gaps + 1
                               bound_found[seq] = True
                        #print(dist_blocks, avail_gaps, bound_found)
                   
           #print(clust_aas_i, dist_blocks, avail_gaps, present_gaps)
           if not 0 in avail_gaps.values():
             #print("There are currently {} gaps".format(present_gaps))
             #print("avail_gaps = {}".format(avail_gaps))
             #print("We can attempt a merge at distance {}".format(min(avail_gaps.values())))     
             shift_dist = min(avail_gaps.values())
             block_range = max(dist_blocks.values())
             #print("block range: {}".format(block_range))
             #if block_range > 5: # FOR TESTING
             #   continue
             potential_new_clusts = [] 
             new_gap_count = 0
 
             tot_displaced = 0
             current_clust = clust_aas_i 
           
             current_gaps = len(alignment) - len(current_clust)
             for b in range(0, block_range):
                if (i + b + shift_dist) >= len(cluster_order):
                     #print("Is this happening?")
                     break
                target_clust = [x[i + b + shift_dist] for x in alignment if not x[i + b + shift_dist] == "-"]
                current_gaps = current_gaps + len(alignment) - len(target_clust)
                #if b == block_range:
                #    current_gaps = current_gaps + numseq - len(target_clust)

                displaced = [x for x in target_clust if x in all_block_aas]
                #print("{} would be displaced from cluster {}".format(displaced, target_clust))
                
                target_minus_displaced = [x for x in target_clust if x not in all_block_aas]
                potential_new_clust = current_clust + target_minus_displaced
                current_clust = displaced
                num_displaced = len(displaced)
                tot_displaced = tot_displaced + num_displaced
                potential_new_clusts.append(potential_new_clust)
                new_gap_count = new_gap_count + len(alignment) - len(potential_new_clust)
             #print("Gap found in cluster {}".format(cluster_order[i]))              
             #print("Current gaps: {}, displaced: {}, new gaps: {}".format(current_gaps, tot_displaced, new_gap_count))
             # Not very scientific
             #if tot_displaced < current_gaps and tot_displaced < 10 and new_gap_count < 5 or tot_displaced == 0 and new_gap_count * tot_displaced < current_gaps: 
             #if tot_displaced < len(all_seqnums) and new_gap;;_count < len(all_seqnums) and 
             if (new_gap_count + 1)* tot_displaced < current_gaps:
                      print("from original gap cluster {}".format(cluster_order[i]))
                      print("merging forward with shift dist {}".format(shift_dist))
                      
                      merged.append(i)
                      #cluster_order2.append(i)

                      for z1 in range(0, block_range):
                            cluster_order2.append(i + z1 + shift_dist)
                            merged.append(i + z1 + shift_dist)
                            
                            clustid_to_clust[i + z1 + shift_dist] = potential_new_clusts[z1]

             else:
                #print("Not good merge {}, {}".format(i, clust_aas_i)) 
                clustid_to_clust[i] = clust_aas_i
                cluster_order2.append(i)   
                #print("clustid_to_clust1", clustid_to_clust)
                #print("cluster_order2", cluster_order2)
           else:       
              #print("No available space {}, {}".format(i, clust_aas_i)) 
              clustid_to_clust[i] = clust_aas_i
              cluster_order2.append(i)   
              #print("clustid_to_clust2", clustid_to_clust)
              #print("cluster_order2", cluster_order2)
       else:      
         #print("No gaps normal {}, {}".format(i, clust_aas_i)) 
         clustid_to_clust[i] = clust_aas_i
         cluster_order2.append(i)   
         #print("clustid_to_clust3", clustid_to_clust)
         #print("cluster_order2", cluster_order2)
                  #if j >= max_clustid + dist_block:
                  #     avail_gaps.append(avail_gap)
                  #     break
                  #print(aa, cluster_seqnums)
                  #present_gaps = present_gaps + numseq - len(cluster_seqnums)


           #print("for aa {}, block = {}, and gap = {}".format(aa, dist_block, avail_gap))

          
    #print(cluster_order2)
    #print(clustid_to_clust) 

    return(cluster_order2, clustid_to_clust)




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
    print("attempt squish")
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
                   #print("target_aa", target_aa)
                   if target_aa in c2:
                       if score > 0:
                           intra_clust_hits.append(score )
 
                #except Exception as E:
                #   # Not all indices correspond to an aa, yes they do
                #   continue
          #print(intra_clust_hits)
          #print("c1", c1)
          #print("c2", c2)
          combo = c1 + c2
          #scores = [x[2] for x in intra_clust_hits if x is not None]
          candidate_merge_list.append([cluster_order[i], cluster_order[i + 1], sum(intra_clust_hits)])
          #print("candidate merge list", candidate_merge_list) 
 
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
    for squish in [1,2, 3]:                   
   
        # Start with scores between adjacent clusters
        # Want to merge the higher score when there's a choice
        #print(edges)        
        G = igraph.Graph.TupleList(edges=edges, directed=False)
        G.es['weight'] = weights
        islands = G.clusters(mode = "weak")
        edges = []
        weights = []
        for sub_G in islands.subgraphs():
            n = len(sub_G.vs())
    
            #print(sub_G)
            #print(n)
            
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
               #print(max_weight)
    
               maybe = sub_edges.select(weight_eq = max_weight)
    
               #print(vertex)
               for e in maybe:
                  highest_edge = [x['name'] for x in sub_G.vs() if x.index  in e.tuple]
                  #print(highest_edge, max_weight)
                  #if max_weight > node_highest[highest_edge[0]]:
                  #      node_highest[highest_edge[0]] = max_weight
                  if highest_edge not in edges:
                      edges.append(highest_edge)
                      weights.append(max_weight)
    
    
                  #print(highest_edge)
                  #print(node_highest)
                  #if highest_edge not in to_merge:
                  #   to_merge.append(highest_edge)
     
    
    #print("to_merge", to_merge)
    
    for c in to_merge: 
              #c = [cluster1, cluster2]
              removed_clustids.append(c[1])
              clustid_to_clust[c[0]] =   clustid_to_clust[c[0]] + clustid_to_clust[c[1]]
              clustid_to_clust[c[1]] = []

    #print("Old cluster order", cluster_order)
    cluster_order = [x for x in cluster_order if x not in removed_clustids]
        #ifor vs in sub_G.vs():
            
      
    return(cluster_order, clustid_to_clust)

 
 
def remove_overlap_with_old_clusters(new_clusters, prior_clusters):
    '''
    Discard any new clusters that contain elements of old clusters
    Only modify new clusters in best match-to-cluster process
    '''
    
    aas_in_prior_clusters = list(flatten(prior_clusters))
    #print("aas in prior", aas_in_prior_clusters)
      
    final_new_clusters = []
    for n in new_clusters:
        #for p in prior_clusters:
        overlap =  list(set(aas_in_prior_clusters).intersection(set(n)))
        if len(overlap) > 0:
             #print("prior", p)
             #print("new with overlap of old ", n)
             continue
        elif n in final_new_clusters:
             continue
        else:
             final_new_clusters.append(n)

    return(final_new_clusters)
    
 

      


def remove_feedback_edges(cluster_orders_dict, clustid_to_clust, remove_both = True, alignment_group = 0, attempt = 0, write_ordernet = False, minclustsize = 1):
    """
    Remove both improves quality of initial alignment by remove both aas that are found out of order
    For final refinement, only remove the first one that occurs out of order

    """
    G_order, order_edges = graph_from_cluster_orders(list(cluster_orders_dict.values()))
    weights = [1] * len(G_order.es)

    if write_ordernet == True:
 
       outnet = "testnet_prefb_{}_attempt{}.csv".format(alignment_group, attempt)
       print("outnet", outnet)
       with open(outnet, "w") as outfile:
          outfile.write("c1,c2\n")
          # If do reverse first, don't have to do second resort
          for x in order_edges:
             outstring = "{},{}\n".format(x[0], x[1])        
             outfile.write(outstring)



    # Remove multiedges and self loops
    #print(G_order)
    G_order.es['weight'] = weights
    G_order = G_order.simplify(combine_edges=sum)

    dag_or_not = G_order.is_dag()
    #print ("Dag or Not before remove_feedback?, ", dag_or_not)



    # The edges to remove to make a directed acyclical graph
    # Corresponds to "look backs"
    # With weight, fas, with try to remove lighter edges
    # Feedback arc sets are edges that point backward in directed graph
    
    fas = G_order.feedback_arc_set(weights = 'weight')

  
    print("feedback arc set")
    for x in fas:
        print("arc", x) 

    #i = 0
    to_remove = []
    for edge in G_order.es():
        #print(edge)
        #print(edge.index)
        #print("i", i)
        #return(0)
        source_vertex = G_order.vs[edge.source]["name"]
        target_vertex = G_order.vs[edge.target]["name"]
        if edge.index in fas:
            print(edge.index, source_vertex, target_vertex)
            to_remove.append([source_vertex, target_vertex])
    #print("to_remove", to_remove)   
    # REMOVE all places where seqnum is inferred from order
    remove_dict = {}
   
    #print("cluster_orders_dict", cluster_orders_dict)
    if remove_both == True: 
        to_remove_flat = list(flatten(to_remove))
    else:
        to_remove_flat = [x[0] for x in to_remove]
    #print("to_remove 2", to_remove_flat)
      
    
 
    for seqnum, clustorder in cluster_orders_dict.items():
      remove_dict[seqnum] = []
      remove = []
      if len(clustorder) == 1:
          if clustorder[0] in to_remove_flat: 
              remove_dict[seqnum] = [clustorder[0]]
      print(clustorder)
      for j in range(len(clustorder) - 1):
          

           if [clustorder[j], clustorder[j +1]] in to_remove:
               #print(cluster_orders[i])
               #print(remove_both) 
               #print(cluster_orders[i][j], cluster_orders[i][j + 1])
               if remove_both == True:
                   remove.append(clustorder[j])
               remove.append(clustorder[j + 1])
           remove_dict[seqnum] = list(set(remove))
           
    print("remove_dict", remove_dict)
    #clusters_filt_dag = []
    #print(clusters_filt)
    print("Doing remove")
    reduced_clusters = []
    for clustid, clust in clustid_to_clust.items():
          new_clust = []
          for aa in clust:
              seqnum = aa.seqnum
              remove_from = remove_dict[seqnum]
              if clustid in remove_from:
                  print("removing seq {} from clust {}".format( seqnum, clustid)) 
              else:
                  new_clust.append(aa)
          if len(new_clust) >= minclustsize:
              reduced_clusters.append(new_clust)
    print("minclustsize", minclustsize)
    
    print("reduced clusters", reduced_clusters)
    #for i in range(len(clusters_filt)):
      #    clust = []
         #for aa in clusters_filt[i]:
            #print(aa)
          #  seqnum = aa.seqnum
            #print(aa.seqnum)
            
           # remove_from = remove_dict[seqnum] 
           # if i in remove_from:
           #     print("removing ", i, seqnum) 
           # else:
           #    clust.append(aa)
         #clusters_filt_dag.append(clust)
    #print("remove feedback")
    #dag_or_not = graph_from_cluster_orders(cluster_orders_dag).is_dag()
    #print ("Dag or Not?, ", dag_or_not)

    #for x in clusters_filt_dag:
    #       #print(x)

    return(reduced_clusters)

def remove_streakbreakers(hitlist, seqs_aas, seqnums, seqlens, streakmin = 3):
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
          #print("remove lookbehinds")
          prevmatch = 0
          seq_start = -1
          streak = 0

          no_lookbehinds = []
          for match_state in target_prot:
               #print(match_state)
               if match_state[1].seqpos <= seq_start:
                     #print("lookbehind prevented")
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

          #print("remove lookaheads")
          prevmatch = seqlens[j]
          seq_end = seqlens[j]
          streak = 0

          filtered_target_prot = []
          for match_state in no_lookbehinds[::-1]:
               #print(match_state, streak, prevmatch)
               if match_state[1].seqpos >= seq_end:
                    #print("lookahead prevented")
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

def remove_doubles(cluster, G, minclustsize = 0, keep_higher_degree = False, check_order_consistency = False, keep_higher_score = False):
            ''' If a cluster contains more 1 amino acid from the same sequence, remove that sequence from cluster'''
           
      
            '''
            If 
            '''
            #print("cluster", cluster)
            seqnums = [x.seqnum for x in cluster]

            
            clustcounts = Counter(seqnums)
            #print(clustcounts)
            to_remove = []
            for key, value in clustcounts.items():
                if value > 1:
                   to_remove.append(key)
            print(cluster)
            print(keep_higher_degree, keep_higher_score, to_remove)
            # If there's anything in to_remove, keep the one with highest degree

            if len(to_remove) > 0 and keep_higher_score == True:

                 G = G.vs.select(name_in=cluster).subgraph()
                 #print(G)
                 #rbh_sel = [x for x in rbh_list if x[0] in cluster and x[1] in cluster]
                 #G = igraph.Graph.TupleList(edges=rbh_sel, directed = False)
                 #G = G.simplify() 
                 #print("edges in cluster", rbh_sel)
                 for seqnum in to_remove:
                     cluster = remove_lower_score(cluster, seqnum, G)

            elif len(to_remove) > 0 and keep_higher_degree == True:

                 G = G.vs.select(name_in=cluster).subgraph()
                 #print(G)
                 #rbh_sel = [x for x in rbh_list if x[0] in cluster and x[1] in cluster]
                 #G = igraph.Graph.TupleList(edges=rbh_sel, directed = False)
                 #G = G.simplify() 
                 #print("edges in cluster", rbh_sel)
                 for seqnum in to_remove:
                    cluster = remove_lower_degree(cluster, seqnum, G)
            # Otherwise, remove any aa's from to_remove sequence
            else:
                for x in to_remove:
                   print("Removing sequence {} from cluster".format(x))
                   #print(cluster)
                   #print(seqnums)
                   #print(clustcounts) 

                cluster = [x for x in cluster if x.seqnum not in to_remove]
            if len(cluster) < minclustsize:
               return([])
            else:
                return(cluster)

#def resolve_conflicting_clusters(clusters)

def remove_lower_score(cluster, seqnum, G):

    target_aas = [x for x in cluster if x.seqnum == seqnum]
            #print(aas)       
    degrees = []
    edge_sums = {}
    print(target_aas)
    #print(G)
    print(G.vs()['name'])

    aa_idxs = [G.vs.find(name =x) for x in target_aas]
    for aa in target_aas:
         g_new = G.copy()
         query_vs = g_new.vs.find(name = aa)
         target_vs = [x for x in g_new.vs() if x not in aa_idxs]
         #print("aa_idxs", aa_idxs) 
         #print("target_vs", target_vs) 
         #print("query_vs", query_vs) 
         edges = g_new.es.select(_source=query_vs)#   ['weight'])
         edge_sums[aa] = sum(edges['weight'])
         #G.es.select(_source=aa)
         #print("edges", aa, G)
         #degrees.append(G.vs.find(name  = aa).degree())
         # This doesn't 
         #degrees.append(G.degree( aa))
                  # TODO: Get rbh to return scores
                  # get highest score if degree tie
                  # gap_scores.append(G
    print(edge_sums)
    print("dupped aas", target_aas)
             
    highest_score = max(edge_sums, key=edge_sums.get)
    print("high score", highest_score)
    to_remove = [x for x in target_aas if x != highest_score]
    cluster_filt = [x for x in cluster if x not in to_remove]
    print("cluster", cluster)
    print("cluster_filt", cluster_filt)
    return(cluster_filt)
 
def remove_lower_degree(cluster, seqnum, G):

    target_aas = [x for x in cluster if x.seqnum == seqnum]
            #print(aas)       
    degrees = []
    for aa in target_aas:

         degrees.append(G.vs.find(name  = aa).degree())
         # This doesn't 
         #degrees.append(G.degree( aa))
                  # TODO: Get rbh to return scores
                  # get highest score if degree tie
                  # gap_scores.append(G
    print("dupped aas", target_aas)
             
    highest_degree = target_aas[np.argmax(degrees)]
    print("high degree", highest_degree)
    to_remove = [x for x in target_aas if x != highest_degree]
    cluster_filt = [x for x in cluster if x not in to_remove]
    print("cluster", cluster)
    print("cluster_filt", cluster_filt)
    return(cluster_filt)
   

def graph_from_rbh(rbh_list, directed = False):

    weights = [x[2] for x in rbh_list]
    G = igraph.Graph.TupleList(edges=rbh_list, directed = directed)
    G.es['weight'] = weights 
    G = G.simplify(combine_edges = "first")
    return(G)

def doubles_in_clust(clust):
    seen = []
    #doubled = []
    doubled_seqnums = []
    for pos in clust:
       if pos.seqnum not in seen:
          seen.append(pos.seqnum)
       else:
          #doubled.append(pos)
          doubled_seqnums.append(pos.seqnum)
    doubled_seqnums = list(set(doubled_seqnums))
    return(doubled_seqnums)

# maybe for 

def remove_doubles3(clustid_to_clust, index, hidden_states, index_to_aa):
    
    for clustnum, clust in clustid_to_clust.items():
        doubled_seqnums = doubles_in_clust(clust)
        if doubled_seqnums:
             clust_minus_dub_seqs = [x for x in clust if x.seqnum not in doubled_seqnums] 
             #print("sequence {} in {}, {} is doubled".format(doubled_seqnums, clustnum, clust))
             for seqnum in doubled_seqnums:
                 bestscore = 0
                 double_aas = [x for x in clust if x.seqnum == seqnum]     
                 for aa in double_aas:
                     candidates_w_score = get_set_of_scores(aa, index, hidden_states, index_to_aa)
                     incluster_scores = [x for x in candidates_w_score if x[0] in clust_minus_dub_seqs ]
                     total_incluster_score = sum([x[1] for x in incluster_scores])
                     if total_incluster_score > bestscore:
                         keeper = aa
                         bestscore = total_incluster_score
                 #print("Adding back {} to {}".format(keeper, clust_minus_dub_seqs))
                 clust_minus_dub_seqs = clust_minus_dub_seqs + [keeper]
             clustid_to_clust[clustnum] = clust_minus_dub_seqs


    return(clustid_to_clust)


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
            #print("Seq {} is poorly matching, fraction positions matched {}, removing until later".format(i, matched_prop[i]))
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

 
def reshape_flat(hstates_list):

    # Go from (numseqs, seqlen, emb) to (numseqs * seqlen, emb)
    hidden_states = np.reshape(hstates_list, (hstates_list.shape[0]*hstates_list.shape[1], hstates_list.shape[2]))
    return(hidden_states)



def split_distances_to_sequence2(D, I, index_to_aa, numseqs, seqlens):
   #{queryaa:{seqindex1:[targetaa1:score1,targetaa2:score2], seqindex2:[targetaa2:score2]} 

   # MAKE this

   print(index_to_aa)
   query_aa_dict = {}
   for i in range(len(I)):
      #print(I[i].shape)
      query_aa = index_to_aa[i]
      # Make dictionary, one per sequence
      target_dict = {}
      #for seqindex in seqindexes:
      #    target_dict[seqindex] = []

      for k in range(numseqs):
          target_dict[k] = []
          #target_D_dict[k] = []

      #print(query_aa, i, D[i]) 
      #print(query_aa, i, I[i]) 
      for j in range(len(I[i])):
           try:
              target_aa = index_to_aa[I[i][j]] 
           except Exception as E:
               #print("no aa at",  I[i][j])               
               continue
           seqindex = target_aa.seqindex
           target_dict[seqindex].append([target_aa, D[i][j]]) 
           #print("repeated", query_aa, target_aa,i,j, D[i][j])

      query_aa_dict[query_aa] = target_dict
      #query_aa_D_dict[query_aa] = target_D_dict
   return(query_aa_dict)
     

      
#def old():
#      I_tmp.append(query_aa_I_dict)
#      D_tmp.append(query_aa_D_dict)
#      #X_tmp.append(X_query)
#     
#   #print(padded_seqlen)
#   
#   #print("X_tmp", X_tmp)
#   #for x in X_tmp:
#   #   print("X_tmp X", x)
#   #print(len(X_tmp))
#
#   #I = []
#   #D = []
#   #X = []
#
#   listbreak = 0
#   slices = []
#   print(seqlens)
#   for seqlen in seqlens:
#      slices.append([listbreak, listbreak + seqlen])
#
#      #I.append(I_tmp[listbreak: listbreak + seqlen])
#      #D.append(D_tmp[listbreak: listbreak + seqlen])
#      #X.append(X_tmp[listbreak: listbreak + seqlen])
#      listbreak = seqlen + listbreak
#
#   print(slices)
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
    
                  if besthit_score - next_besthit_score <= 0.002:
                      print("{} has tie between {}:{} and {}:{}".format(aa, besthit_aa, besthit_score, next_besthit_aa, next_besthit_score))
                      continue
              #print(I[aa][targetseq])
              #print(aa, besthit_aa, besthit_score)
              if besthit_score >= minscore:
                  hitlist.append([aa, besthit_aa, besthit_score])

   return(hitlist) 

def get_rbhs(hitlist_top):
    '''
    [aa1, aa2, score (higher = better]
    '''

    G_hitlist = igraph.Graph.TupleList(edges=hitlist_top, directed=True) 
    weights = [x[2] for x in hitlist_top]

    rbh_bool = G_hitlist.is_mutual()
    
    hitlist = []
    G_hitlist.es['weight'] = weights 
    for i in range(len(G_hitlist.es())):
        if rbh_bool[i] == True:
           source_vertex = G_hitlist.vs[G_hitlist.es()[i].source]["name"]
           target_vertex = G_hitlist.vs[G_hitlist.es()[i].target]["name"]
           score = G_hitlist.es()[i]['weight']
           hitlist.append([source_vertex, target_vertex, score])
    return(hitlist)


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
           # #print("before", sub_G.vs()['name'])
 
            #bet_names = list(zip(sub_G.vs()['name'], bet_norm))
            # A node with bet_norm 0.5 is perfectly split between two clusters
            # Only select nodes with normalized betweenness before 0.45
            pruned_vs = G.vs.select([v for v, b in enumerate(bet_norm) if b < betweenness_cutoff]) 
                
            new_G = G.subgraph(pruned_vs)
            return(new_G) 

def first_clustering(G,  betweenness_cutoff = .10, minclustsize = 0, ignore_betweenness = False, apply_walktrap = True):
    '''
    Get betweenness centrality
    Each node's betweenness is normalized by dividing by the number of edges that exclude that node. 
    n = number of nodes in disconnected subgraph
    correction = = ((n - 1) * (n - 2)) / 2 
    norm_betweenness = betweenness / correction 
    '''

    #G = igraph.Graph.TupleList(edges=rbh_list, directed=False)
    print("Start first clustering")

    # Remove multiedges and self loops
    #print("Remove multiedges and self loops")
    #G = G.simplify()

    # Could estimate this with walktrap clustering with more steps if speed is an issue  

    # Initial islands in the RBH graph
    
    islands = G.clusters(mode = "weak")
    new_subgraphs = []
    cluster_list = []
    hb_list = []

    # For each island, evaluate what to do
    for sub_G in islands.subgraphs():
        # Don't remove edges if G = size 2
        if len(sub_G.vs()) < 4:
               betweenness_cutoff = 1
        else:
           betweenness_cutoff = betweenness_cutoff
        print("First connected set", sub_G.vs()['name'])
        
        # First start with only remove very HB nodes
        new_G = remove_highbetweenness(sub_G, betweenness_cutoff = 0.1)
        sub_islands = new_G.clusters(mode = "weak")
        for sub_sub_G in sub_islands.subgraphs():

            # Should do a betweenness here again


            new_clusters = get_new_clustering(sub_sub_G, betweenness_cutoff = betweenness_cutoff,  apply_walktrap = apply_walktrap) 
            cluster_list = cluster_list + new_clusters

        # Need? a first pass where where remove very hb nodes
        #new_G = remove_highbetweenness(sub_G, betweenness_cutoff = 0.45)
    
            # It's not necesarrily a connected set since the hb nodes were removed
        #new_set = new_G.vs()['name']
        #hb_list = hb_list + [x for x in sub_G.vs['name'] if x not in new_set]
       

        #for sub_su


    #print("excuse me, what'sthe minclustsize", minclustsize)
    #for x in cluster_list:
    #    print("Pre size filt", x)
    # Do minclustsize later
    #cluster_list = [x for x in cluster_list if len(x) >= minclustsize]
    #for x in cluster_list:
    #   print("cluster_list", x)
    #    #print(sub_G)
    #    if ignore_betweenness == False:
    #        n = len(sub_G.vs())
    #        # Remove small subgraphs
    #        # ?? Marker 
    #        if n < 3:
    #            if n < minclustsize:
    #               continue
    #            else:
    #              # #print("here1", sub_G.vs()['name'])
    #               cluster_list.append(sub_G.vs()['name'])
    #               #print("New cluster from first clustering", sub_G.vs()['name'])
    #               continue

 
    #        new_G = remove_highbetweenness(sub_G, betweenness_cutoff = betweenness_cutoff)
    #
    #        # It's not necesarrily a connected set since the hb nodes were removed
    #        connected_set = new_G.vs()['name']
    #        print("connected set", connected_set)
    #        hb_list = hb_list + [x for x in sub_G.vs['name'] if x not in connected_set]

    #    else:
    #        #print("We are ignoring betweenness now")
    #        connected_set = sub_G.vs()['name']
    #        if len(connected_set) < minclustsize:
    #            continue


    #    if len(connected_set) < minclustsize:
    #       #print(connected_set, len(connected_set))
    #       continue

    #    #if ignore_betweenness == False:
    #    sub_islands = new_G.clusters(mode = "weak")
    #        
    #    for sub_sub_G in sub_islands.subgraphs():
    #         print("sub_sub_G", sub_sub_G.vs()['name'])
    #         sub_sub_connected_sets = get_new_clustering(sub_sub_G, apply_walktrap = apply_walktrap)
    #         print("sub_sub_connected_sets", sub_sub_connected_sets)
    #         if sub_sub_connected_sets is not None:

                   #cluster_list = cluster_list + sub_sub_connected_sets

        #else:
        #    #print("Dealing with connected set, ignore betweenness")
        #    # Potentially break apply_walktrap False to a second step
        #   
        #    if len(sub_G.vs()['name']) > 2: 
        #      sub_sub_connected_set = get_new_clustering(sub_G, minclustsize, apply_walktrap = False)
        #      #print("sub_sub_connected_set", sub_sub_connected_set)
        #      cluster_list= cluster_list  + sub_sub_connected_set
        #    else:
        #      cluster_list.append(sub_G.vs()['name'])             
                        
        #print(cluster_list) 
    #for x in cluster_list:
        #print("First clustering cluster", x)

    #cluster_list = removeSublist(cluster_list)
    print("cluster_list", cluster_list)
    return(cluster_list)


def get_new_clustering(G, betweenness_cutoff = 0.10,  apply_walktrap = True):

    new_clusters = []
    connected_set = G.vs()['name']
   # #print("after ", sub_connected_set)
    print("connected set", connected_set)

    new_clusters = []
 
    finished = check_completeness(connected_set)
    # Complete if no duplicates
    if finished == True:
        print("finished connected set", connected_set)
        new_clusters = [connected_set]

    else:
        
        min_dupped =  min_dup(connected_set, 1.2)
        # Only do walktrap is cluster is overlarge
        if (len(connected_set) > min_dupped) and apply_walktrap and len(G.vs()) >= 5:
            print("applying walktrap")
            # Change these steps to 3??
            
            clustering = G.community_walktrap(steps = 1, weights = 'weight').as_clustering()
            #i = 0
            for sub_G in clustering.subgraphs():
                 sub_connected_set =  sub_G.vs()['name']
                 print("post cluster subgraph", sub_connected_set)
                 
                 # New clusters may be too large still, try division process w/ betweenness
                 
                 new_clusters = new_clusters + process_connected_set(sub_connected_set, sub_G, dup_thresh = 1.2, betweenness_cutoff = betweenness_cutoff) 

        else:

            new_clusters = new_clusters + process_connected_set(connected_set, G, dup_thresh = 1.2, betweenness_cutoff = betweenness_cutoff)
        

    return(new_clusters)


def get_represented_seqs(connected_set):
    represented_seqs = list(set([x.seqnum for x in connected_set]))
    return(represented_seqs)


def min_dup(connected_set, dup_thresh):
    represented_seqs = get_represented_seqs(connected_set)
    #print("tot_represented_seqs", tot_represented_seqs)
    #print(len(connected_set), len(tot_represented_seqs))
    return(dup_thresh *  len(represented_seqs))

 
def process_connected_set(connected_set, G, dup_thresh = 1.2,  betweenness_cutoff = 0.10):
    ''' 
    This will take a connected sets and 
    1) Check if it's very duplicated
    2) If so, remove high betweenness nodes, and check for completeness again
    3) If it's not very duplicated, removed any duplicates by best score
    '''

    new_clusters = []
    min_dupped =  min_dup(connected_set, dup_thresh)
    if len(connected_set) > min_dupped:
        print("cluster too big", connected_set)

        # TRY removing high betweenness and evaluating completeness
        print("Check for betweenness")
       
        new_G = remove_highbetweenness(G, betweenness_cutoff = 0.10)
        print("prebet", G.vs()['name'])
        print("postbet", new_G.vs()['name'])
        #if len(new_Gs) > 1:
        new_islands = new_G.clusters(mode = "weak")
        for sub_G in new_islands.subgraphs():
                sub_connected_set = sub_G.vs()['name']
                print("postbet_island", sub_connected_set)
                sub_min_dupped =  min_dup(sub_connected_set, dup_thresh) 
                

                # Actually going to keep all clusters below min dup thresh
                if (len(sub_connected_set) <= sub_min_dupped) or (len(sub_connected_set) <= 5):
                #    sub_connected_set = remove_doubles(sub_connected_set, sub_G, keep_higher_score = True)
                   


                #sub_finished = check_completeness(sub_connected_set)
                ## Complete if no duplicates
                #if sub_finished == True:
                #    print("finished sub after hbremoval", sub_connected_set)
                    new_clusters.append(sub_connected_set)
  


        #return(new_clusters)
    else:
        #trimmed_connected_set = remove_doubles(connected_set, G, keep_higher_score = True)
        #print("after trimming by removing doubles", trimmed_connected_set)
        new_clusters = [connected_set] 
    # If no new clusters, returns []
    return(new_clusters)
 
def check_completeness(cluster):

            seqnums = [x.seqnum for x in cluster]
            clustcounts = Counter(seqnums)
            # If any sequence found more than once
            for value in clustcounts.values():
                if value > 1:
                   return(False)
            return(True)
 




def get_walktrap(hitlist):
    G = igraph.Graph.TupleList(edges=hitlist, directed=True)
    # Remove multiedges and self loops
    #print("Remove multiedges and self loops")
    G = G.simplify()
    
    print("start walktrap")
    clustering = G.community_walktrap(steps = 1).as_clustering()
    print(clustering)
    print("walktrap done")
    

    clusters_list = clustering_to_clusterlist(G, clustering)
    return(clusters_list)


def get_cluster_dict(clusters, seqs):

    pos_to_clustid = {}
    clustid_to_clust = {}
    for i in range(len(clusters)):
       clust = clusters[i]
       clustid_to_clust[i] = clust 
       for seq in clust:
              pos_to_clustid[seq] = i

    return(pos_to_clustid, clustid_to_clust)
 
def get_cluster_orders(cluster_dict, seqs_aas):
    # This is getting path of each sequence through clusters 
    cluster_orders_dict = {}

    for i in range(len(seqs_aas)):
        seqnum = seqs_aas[i][0].seqnum # Get the setnum of the set of aas
        #print("SEQNUM",  seqnum)
        cluster_order = []
        
        for j in range(len(seqs_aas[i])):
           key = seqs_aas[i][j]
           #print("key", key)
           try:
              clust = cluster_dict[key]
              #print("clust", clust)
              cluster_order.append(clust)
           except Exception as E:
              #print(E)
              # Not every aa is sorted into a cluster
              continue
        cluster_orders_dict[seqnum] = cluster_order
    return(cluster_orders_dict)

#cluster_orders_dag, pos_to_clust_dag, dag_reached = clusters_to_dag(clusters_filt, seqs_aas)

def clusters_to_dag(clusters_filt, seqs_aas, remove_both = True, dag_reached = False, alignment_group = 0, attempt = 0, write_ordernet = False, minclustsize = 1):
    ######################################3
    # Remove feedback loops in paths through clusters
    # For getting consensus cluster order
  
    #print("status of remove_both", remove_both)
    numseqs = len(seqs_aas)
    #for x in clusters_filt:
    #     #print(x)
    pos_to_clustid, clustid_to_clust = get_cluster_dict(clusters_filt, seqs_aas)
    #print("clustid_to_clust pre dag", clustid_to_clust)
    #print("test1")
    #print(pos_to_clustid)
    #print("test2")
    #print(clustid_to_clust)
    cluster_orders_dict = get_cluster_orders(pos_to_clustid, seqs_aas)
    #print(cluster_orders_dict)
    #print("test3")

    print("clusters_to_dag minclustsize", minclustsize)
    #print("Find directed acyclic graph")   
    clusters_filt_dag = remove_feedback_edges(cluster_orders_dict, clustid_to_clust, remove_both, alignment_group = alignment_group, attempt = attempt, write_ordernet = write_ordernet, minclustsize= minclustsize)


    clusters_filt_dag = [x for x in clusters_filt_dag if len(x) >= minclustsize]
    #for x in clusters_filt_dag:
    #   print("clusters_filt_dag", x)
    #print("Feedback edges removed")

    #print("Get cluster order after feedback removeal")
    

    pos_to_clust_dag, clustid_to_clust_dag = get_cluster_dict(clusters_filt_dag, seqs_aas)

    cluster_orders_dict = get_cluster_orders(pos_to_clust_dag, seqs_aas)

    #print(cluster_orders_dict)
    dag_or_not_func = graph_from_cluster_orders(list(cluster_orders_dict.values()))[0].simplify().is_dag()
    print("Dag or Not? from function, ", dag_or_not_func) 

    if dag_or_not_func == True:
          dag_reached = True
    
    else:
          #print("DAG not reached, will try to remove further edges")
          dag_reached = False
    clusters_filt = list(clustid_to_clust_dag.values())
    return(cluster_orders_dict, pos_to_clust_dag, clustid_to_clust_dag, clusters_filt, dag_reached)




def dag_to_cluster_order(cluster_orders_dict, seqs_aas, pos_to_clust_dag, clustid_to_clust_dag):

    #pos_to_clust_dag out of sync with clustid_to_clust_dag
    # Describe the diff between input variables
    #print("calling topo_sort from function")

    #with open("tester1.txt", "w") as f:
    #   for x in cluster_orders_dict.values():
    #      f.write("{}\n".format(x))
    

    #print("cluster orders dict", cluster_orders_dict)
    cluster_order = get_topological_sort(list(cluster_orders_dict.values())) 
    #print("For each sequence check that the cluster order doesn't conflict with aa order")
    #print("cluster_order", cluster_order)
    cluster_order = remove_order_conflicts(cluster_order, seqs_aas, pos_to_clust_dag)


    clustid_to_clust = {key:val for key, val in clustid_to_clust_dag.items() if key  in cluster_order}

    # Another dag check

    #print("cluster_order", cluster_order)
    #print("renumber")
    cluster_order_dict = {}
    for i in range(len(cluster_order)):
        cluster_order_dict[cluster_order[i]] = i

    clustid_to_clust_inorder = {}
    pos_to_clust_inorder = {}
    cluster_order_inorder = []
    for i in range(len(cluster_order)):
         clustid_to_clust_inorder[i] = clustid_to_clust[cluster_order[i]]    
         cluster_order_inorder.append(i)

    #print("clustid_to_clust_inorder", clustid_to_clust_inorder)
    #print("cluster_order_inorder", cluster_order_inorder)
    #print("cluster_order_dict", cluster_order_dict)

    for key in pos_to_clust_dag.keys():
         #print("key", key)
         #print("pos_to_clust_dag[key]", pos_to_clust_dag[key])
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





def get_seq_groups(seqs, seq_names, embedding_dict, logging, exclude, do_clustering):
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

    print(sentence_array.shape)
    if sentence_array.shape[1] > 1024:
       sentence_array = sentence_array[:,:1024]
    print(sentence_array.shape)

    #print("sentence_array", sentence_array)
    s_index = build_index_flat(sentence_array)
    s_distance, s_index2 = s_index.search(sentence_array, k = k_select)

    #print(s_distance) 
    #print(s_index2)
    G = graph_from_distindex(s_index2, s_distance)
    #print(G)
    G = G.simplify(combine_edges = "first")  # symmetrical, doesn't matter
    #print(G)
    #print("not excluding?", exclude)
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
    
      repeat = True
      while repeat == True:

        group_hstates_list = []
        cluster_seqnums_list = []
        cluster_names_list = []
        cluster_seqs_list = []
 
        prev_to_exclude = to_exclude
        

        print("GG", G.vs()['name'])
        print("GG", G.es()['weight'])
        #seq_clusters = G.community_fastgreedy(weights = 'weight').as_clustering() 
        #print(seq_clusters)
           # This has about same output as fastgreedy
            #print("multilevel")
        seq_clusters = G.community_multilevel(weights = 'weight')
        #seq_clusters = G.community_walktrap(steps = 2, weights = 'weight').as_clustering() 
        print(G)       
        print(seq_clusters)
        for seq_cluster_G in seq_clusters.subgraphs():
            # Do exclusion within clusters
            print("seq_clusters", seq_cluster_G)
            if exclude == True:

                clust_names = seq_cluster_G.vs()["name"]
                print("clust_names", clust_names)
                cluster_to_exclude = candidate_to_remove(seq_cluster_G, clust_names, z = -3)
                print(cluster_to_exclude)
                   
                #print('name', to_exclude)
                to_delete_ids_sub_G = [v.index for v in seq_cluster_G.vs if v['name'] in cluster_to_exclude]
                #print('vertix_id', to_delete_ids)
                seq_cluster_G.delete_vertices(to_delete_ids_sub_G) 

                to_delete_ids_G = [v.index for v in G.vs if v['name'] in cluster_to_exclude]
                G.delete_vertices(to_delete_ids_G)
                print("to_exclude_pre", to_exclude)
                to_exclude = to_exclude + cluster_to_exclude
                to_exclude = list(set(to_exclude))
                print("to_exclude_post", to_exclude)
                if to_exclude:       
                    logging.info("Excluding following sequences: {}".format(",".join([str(x) for x in to_exclude])))
                    print("Excluding following sequences: {}".format(",".join([str(x) for x in to_exclude])))

            hstates = []
            seq_cluster = seq_cluster_G.vs()['name']
            seq_cluster.sort()
            print(seq_cluster)
            cluster_seqnums_list.append(seq_cluster)
    
            filter_indices = seq_cluster
            group_hstates = np.take(embedding_dict['aa_embeddings'], filter_indices, axis = 0)
            group_hstates_list.append(group_hstates)
            #Aprint(group_hstates.shape)
    
            cluster_names = [seq_names[i] for i in filter_indices]
            cluster_names_list.append(cluster_names)
       
            cluster_seq = [seqs[i] for i in filter_indices]
            cluster_seqs_list.append(cluster_seq)
            to_exclude = list(set(to_exclude))
        print("eq check", to_exclude, prev_to_exclude)
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
            print('name', to_exclude)
            to_delete_ids = [v.index for v in G.vs if v['name'] in to_exclude]
            #print('vertix_id', to_delete_ids)
            G.delete_vertices(to_delete_ids) 
    
            logging.info("Excluding following sequences: {}".format(",".join([str(x) for x in to_exclude])))
    
         else:
           logging.info("Not removing outlier sequences")
           to_exclude = []
 
 
        # #print([v['name'] for v in G.vs])
         cluster_seqnums_list =  [v['name'] for v in G.vs]
         print(cluster_seqnums_list, to_exclude)
         cluster_seqnums_list = list(set(cluster_seqnums_list))
         cluster_seqnums_list.sort()
         # Make sure this is removing to_exclude corectly
         cluster_seqs_list = [[seqs[i] for i in cluster_seqnums_list]]
         cluster_names_list = [[seq_names[i] for i in cluster_seqnums_list]]
         group_hstates_list = [np.take(embedding_dict['aa_embeddings'], cluster_seqnums_list, axis = 0)]
         cluster_seqnums_list = [cluster_seqnums_list] 
         to_exclude = list(set(to_exclude))

    print("senum clusters", cluster_seqnums_list)
    print(cluster_names_list)
    return(cluster_seqnums_list, cluster_seqs_list,  cluster_names_list, group_hstates_list, to_exclude)

def dedup_clusters(clusters_list, G, minclustsize):
    new_clusters_list = []
    for clust in clusters_list:
        if len(clust) > len(get_represented_seqs(clust)):

             resolved = False
             print("has dups after very first clustering", clust)
             #for pos in clust:
             for otherclust in clusters_list:
               if clusters_list == otherclust:
                   continue
               # Check if removing a smaller cluster resolved duplicates
               if len(set(clust).intersection(set(otherclust))) >= 2:
                    trimmed_clust = [x for x in clust if x not in otherclust]
                    clust_complete = check_completeness(trimmed_clust)
                   
                    if clust_complete: 
                         if trimmed_clust not in new_clusters_list:
                            new_clusters_list.append(trimmed_clust)
                            resolved = True
             if resolved == False:
                  reduced_clust =  remove_doubles(clust, G, keep_higher_score = True)
                  complete = check_completeness(reduced_clust)
                  if complete:
                      new_clusters_list.append(reduced_clust)

 

        else:
             if clust not in new_clusters_list:
                  new_clusters_list.append(clust)
    return(new_clusters_list)

def get_similarity_network(seqs, seq_names, seqnums, hstates_list, logging, padding = 5, minscore1 = 0.5, alignment_group = 0, headnorm = False):
    """
    Control for running whole alignment process
    seqs should be spaced
    padding tells amount of padding to remove from seqs
    model = prot_bert_bfd
    """

    seqlens = [len(x) for x in seqs]
    print("seqs", seqs, seqlens)

    padded_seqlen = hstates_list.shape[1]
     
    numseqs = len(seqs)
    print("numseqs", numseqs)
    print("padded_seqlen", padded_seqlen)
    print(hstates_list.shape)

    lsts = []

    
    #return(0)

    #hstates_list.normalize(axis = 2, norm = "l2")
    embedding_length = hstates_list.shape[2]
    #hstates_heads = hstates_list.reshape(numseqs, padded_seqlen, -1, 64)

    if headnorm == True:
        hstates_heads = hstates_list.reshape(-1, 64)
        print(hstates_heads.shape)
    
        hstates_heads = normalize(hstates_heads, axis =1, norm = "l2")
    

        hstates_list = hstates_heads.reshape(numseqs, padded_seqlen, embedding_length)
        print(hstates_list.shape)

#    for head in range(0, hstates_list.shape[2], 64):
#   
#      lst = []
#      print(head, head + 64)
#      b = np.take(hstates_list, indices= range(head, head + 64), axis = 2)
#    
#      print(b.shape)
#
##    #print(b)
#    # For prot
#      protlist = []
#      for i in range(len(b)):
#        # for aa
#        #for j in range(len(b[i])):
#        #  # for aa
#        #  for k in range(len(b[i])):
#        #      print("vect1", b[i][j])
#        #      print("vect2", b[i][k])
#        #      protlist.append(cosine_similarity(b[i][j], b[i][k]))
#        #norm_b = normalize(b[i], axis =2, norm = "l2")
#        #norm_b = np.linalg.norm(b[i], axis=1)
#        #print(norm_b)
#        #print(b[i].shape)
#        A = cosine_similarity(b[i], dense_output= False)
#        B =  A[np.triu_indices(3, k = 1)]
#        protlist.append(np.mean(B))
#      print(protlist)
# #           
  
#    #means = np.mean(b, axis = 1)
#    #print(variances)
#    #rint(np.var(means, axis = 1))
#       
#   #for s in hstates_list:
#   #     print(s.shape)
#        



   # Drop X's from here
    #print(hstates_list.shape)
    # Remove first and last X padding

    # After encoding, remove spaces from sequences
    #for seq in seqs:
    #   hidden_states = get_hidden_states([seq], model, tokenizer, layers)
    #   hidden_states_list.append(hidden_states)


    # Build index from all amino acids 
    #d = hidden_states[0].shape[1]

    # Go from (numseqs, seqlen, emb) to (numseqs * seqlen, emb)
    #return(0)
    logging.info("Flattening hidden states list")
    hidden_states = np.array(reshape_flat(hstates_list))  
    logging.info("embedding_shape: {}".format(hidden_states.shape))


    
    logging.info("Convert index position to amino acid position")
    #index_to_aa = {}


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
    

    # Initial index to remove padding from input embeddings
    index_to_aa = {}
    aa_indices = []
    for i in range(len(seqs_aas)):
        for j in range(padded_seqlen):
           if j >= seqlens[i]:
             continue 
           
           aa = seqs_aas[i][j]
           index_num = i * padded_seqlen + j
           #aa.index = index_num           
           index_to_aa[index_num] = aa
           aa_indices.append(index_num)         


    for key, value in index_to_aa.items():
          print(key, value)


    # Remove padding from aa embeddings
    hidden_states = np.take(hidden_states, list(index_to_aa.keys()), 0)

    index_to_aa = {}
    count_index = 0
    for i in range(len(seqs_aas)):
       for j in range(0, seqlens[i]):
           aa = seqs_aas[i][j]
           aa.index = count_index
           aa.seqindex = i
           index_to_aa[count_index] = aa
           count_index = count_index + 1
           
    
    print("shape", hidden_states.shape)
    logging.info("Build index")     

    # TODO: Add check for when voronoi is needed
    #index= build_index_voronoi(hidden_states, seqlens)
    #print("ncells", int(max(seqlens)/10))
    #index.nprobe =  int(max(seqlens)/10)
    #index.nprobe = 3
    faiss.normalize_L2(hidden_states)
    index= build_index_flat(hidden_states)
    logging.info("Search index2") 

    #print("36")
    #miniquery = np.take(hidden_states, [36], 0)
    #D,I = index.search(miniquery, k = numseqs*100)
    #print(I)
    #print("424")
    #miniquery = np.take(hidden_states, [424], 0)
    
    #D,I = index.search(miniquery, k = numseqs*100)
    #print(I)
    #print("miniquery_done")
    #return(0)
    # CHANGE BACK
    D1, I1 =  index.search(hidden_states, k = numseqs*10) 
    print("SHAPE", hidden_states.shape)


    #a, test1_D = index.search(hidden_states[36], k = 500)
    #print(test1_I)
    #print(test1_D)
    logging.info("Search index2 done")

    #print("Split results into proteins") 
    logging.info("Split results into proteins2") 
    I2 = split_distances_to_sequence2(D1, I1, index_to_aa, numseqs, seqlens) 
    logging.info("Split results into proteins done")

    #for aa in I2.keys():
    #     print("key", aa)
    #     for seq in I2[aa].keys():
    #          print(aa, seq, I2[aa][seq])

    #print(I2)
    logging.info("get best hitlist")
    #print("get best hitlist")
    minscore1 = 0.01
 
    #return(0)
    hitlist_all = get_besthits(I2, minscore = minscore1)
    
    #hitlist_all = get_besthits(D2, I2, seqnums, index_to_aa, padded_seqlen, minscore = minscore1)
    for x in hitlist_all:
       print("hitlist_all:", x)
    logging.info("got best hitlist")

    #logging.info("get top hitlist")
  
    #hitlist_top = [ x for x in hitlist_all if x[2] >= minscore1]
    #logging.info("got top hitlist")
    #for x in hitlist_top:
    #   print("hitlist_top:", x)


    #print("hitlist_top")
    #for x in hitlist_top:
    #      print("hitlist_top", x)
    #logging.info("Get reciprocal best hits")
    #print("Get reciprocal best hits")
    logging.info("get reciprocal best hits")
    rbh_list = get_rbhs(hitlist_all) 
    logging.info("got reciprocal best hits")
   
    #for x in rbh_list:
    #    print("rbh:", x)
    #print("got reciprocal besthits")
    #return(0) 
    # Skip this?
    #remove_streaks = False
    #if remove_streaks == True:
    #    logging.info("Remove streak conflict matches")
    #    rbh_list = remove_streakbreakers(rbh_list, seqs_aas, seqnums, seqlens, streakmin = 3)
    for x in rbh_list:
      print("rbh", x) 
   
    ######################################### Do walktrap clustering
    # Why isn't this directed?
    outnet = "testnet_initial_clustering{}.csv".format(alignment_group)
    with open(outnet, "w") as outfile:
          outfile.write("aa1,aa2,score\n")
          # If do reverse first, don't have to do second resort
          for x in rbh_list:
             outstring = "{},{},{}\n".format(x[0], x[1], x[2])        
             outfile.write(outstring)


    print("Start betweenness calculation to filter cluster-connecting amino acids. Also first round clustering")

    G = graph_from_rbh(rbh_list, directed = False)
  
    # In CBS, how to 10-6-P and 1-6-D end up in same cluster?
    #print(G.vs.find(name = "0-25-H"))
    #connections = G.es.select(_source = 242)
    #connected_aas = [x.target_vertex['name'] for x in connections]
    #print(connected_aas)
 
    #return(0) 
    #print(G)
    # Unused maxclustsize, don't go by numseqs, go by duplication of representative seqs
    #maxclustsize = 2*len(seqs) 
    if len(seqs) > 2:
        minclustsize = int(len(seqs)/2) + 1
        clusters_list = first_clustering(G, betweenness_cutoff = 0.10, minclustsize = minclustsize, ignore_betweenness = False, apply_walktrap = False)
    else:
        minclustsize = 2
        clusters_list = first_clustering(G, betweenness_cutoff = 1, minclustsize = minclustsize, ignore_betweenness = True, apply_walktrap = False)
    #print("High betweenness list, hblist: ", hb_list)
    
    clusters_list = [x for x in clusters_list if len(x) > 1]
    for x in clusters_list:
        print("FIRST clust", x)

    new_clusters_list = dedup_clusters(clusters_list, G, minclustsize)
    print("new_clusters_list", new_clusters_list)
    # Need to uniquify this
    clusters_filt = [x for x in new_clusters_list if len(x) >= minclustsize]
    #return(0)
    # Removing streakbreakers may still be useful
    #clusters_filt = []
    #for cluster in clusters_list:
    #     print("very first", cluster)
    #     cluster_filt = remove_doubles(cluster, minclustsize = minclustsize, keep_higher_degree = False, G = G, keep_higher_score = True)
    #     # Could just do cluster size check here
    #     clusters_filt.append(cluster_filt)
    for x in clusters_filt:
          print("cluster_filt1", x)
    #print("Getting DAG of cluster orders, removing feedback loops")

    logging.info("Get DAG of cluster orders, removing feedback loops")

    dag_reached = False
    count = 0

    while dag_reached == False:
      
       count = count + 1
       #print("Dag finding attempt number {}".format(count))
       # If length gets down to too, the dag will be reached, so this never will happen
       if len(clusters_filt) < 2:
           print("Dag not reached, no edges left to remove".format(count))
           return(1) 
       
       cluster_orders, pos_to_clust, clustid_to_clust, clusters_filt, dag_reached = clusters_to_dag(clusters_filt, seqs_aas , alignment_group = alignment_group, attempt = count, write_ordernet = True, remove_both = True, minclustsize = minclustsize)
       #print("Dag reached?", dag_reached) 

    #print("Dag found, getting cluster order with topological sort, dag_reached = ", dag_reached)
    cluster_order, clustid_to_clust, pos_to_clustid=  dag_to_cluster_order(cluster_orders, seqs_aas, pos_to_clust, clustid_to_clust)
    #print("odd region done")  

    for key, value in clustid_to_clust.items():
        print("pregapfil", key,value)

    #print("Need to get new clusters_filt")
    clusters_filt = list(clustid_to_clust.values())  

    #for x in clusters_filt:
    #      #print("cluster_filt_dag", x)
    logging.info("Make alignment")
    alignment = make_alignment(cluster_order, seqnums, clustid_to_clust)
    #print(alignment_print(alignment, seq_names)[0])

    logging.info("\n{}".format(alignment))

   
    # Observations:
       #Too much first character dependencyi
       # Too much end character dependency
          # Added X on beginning and end seems to fix at least for start
    #print("Get sets of unaccounted for amino acids")
    # Need to 

    ignore_betweenness = False

    if len(seqnums) > 2:
       if len(seqnums) < 5:
            minclustsize = len(seqnums) - 1
       else:
            minclustsize = 5
    else:
       minclustsize = 2

    minscore = 0.1
    betweenness_cutoff = 0.10
    history_unassigned = {'onebefore':[], 'twobefore':[], 'threebefore':[]}
    # MINCLUSTSIZE not being respected, fix
    ############## CONTROL LOOP ###################
    for gapfilling_attempt in range(0, 100):
        gapfilling_attempt = gapfilling_attempt + 1
        print("Align this is gapfilling attempt ", gapfilling_attempt)
        logging.info("gapfilling_attempt {}".format(gapfilling_attempt))
        if gapfilling_attempt > 6 and minclustsize > 2 and gapfilling_attempt % 2 == 1:
                minclustsize = minclustsize - 1
        print("This is the minclustsize", minclustsize)
       #print("Get unassigned aas")
        #print("Dupl troubleshooting")  
        #print(seqs_aas)
        #print(pos_to_clustid)
        unassigned = get_unassigned_aas(seqs_aas, pos_to_clustid)
        for x in unassigned:
           print("unassign", x)
        #print("HERE")

        if len(unassigned) == 0:
            #print("Alignment complete after {} gapfilling attempt".format(gapfilling_attempt - 1))
  
     
            alignment = make_alignment(cluster_order, seqnums, clustid_to_clust)
           


            return(alignment, index, hidden_states, index_to_aa)

        if ( unassigned == history_unassigned['threebefore'] or  unassigned == history_unassigned['twobefore'] ) and gapfilling_attempt > 10:

            print("Align by placing remaining amino acids")
            cluster_order, clustid_to_clust, pos_to_clustid, alignment = fill_in_hopeless2(unassigned, seqs, seqs_aas, seqnums, cluster_order, clustid_to_clust, pos_to_clustid, numseqs, index, hidden_states)
            unassigned = get_unassigned_aas(seqs_aas, pos_to_clustid)
   
 
            alignment = make_alignment(cluster_order, seqnums, clustid_to_clust)

            return(alignment,   index, hidden_states, index_to_aa)
 
        history_unassigned['threebefore'] = history_unassigned['twobefore']
        history_unassigned['twobefore'] = history_unassigned['onebefore']
        history_unassigned['onebefore'] = unassigned
        apply_walktrap = False
        # Do one or two rounds of clustering between guideposts
        if gapfilling_attempt in list(range(1, 100,2)) :#  or gapfilling_attempt in [1, 2, 3, 4]:
            logging.info("Do clustering within guideposts")
            #minclustsize = 2
            print("Align by clustering within guideposts")
            # Don't allow modification of previous guideposts
            #print("Align by rbh between guideposts")
            # Get graph list?
            if gapfilling_attempt > 4:
                 apply_walktrap = True
            cluster_order, clustid_to_clust, pos_to_clustid, alignment = fill_in_unassigned(unassigned, seqs, seqs_aas, seq_names, seqnums, cluster_order, clustid_to_clust, pos_to_clustid, numseqs, I2, minscore = minscore ,minclustsize = minclustsize, ignore_betweenness = ignore_betweenness, betweenness_cutoff = betweenness_cutoff, apply_walktrap = apply_walktrap)
         

            for key,value in clustid_to_clust.items():
                 print(key, value)

        #After than do a sort of aa's into previous clusters
        # First using rbh
        # Then using individual search. 
        else:
            #print(cluster_order)
            #print(clustid_to_clust) 
            #print("Add aa's into previous clusters")

            # Use original rbh to fill in easy cases
            # Or maybe do updated rbh between guideposts
            #if gapfilling_attempt in [1]:
            #    print("Align by best match (original rbh") 
            #    clustid_to_clust = fill_in_unassigned2(unassigned, seqs, seqs_aas, G, clustid_to_clust)
            #    for key,value in clustid_to_clust.items():
            #        print(key, value)


            #else:
            print("Align by best match (looser)")
            logging.info("Add aa's to existing clusters")
            clustid_to_clust = fill_in_unassigned3(unassigned, seqs, seqs_aas, cluster_order, clustid_to_clust, pos_to_clustid, numseqs, index, hidden_states,  index_to_aa)
            for key,value in clustid_to_clust.items():
                print(key, value)


            clusters_filt = list(clustid_to_clust.values())
            cluster_orders, pos_to_clust, clustid_to_clust, clusters, dag_reached = clusters_to_dag(clusters_filt, seqs_aas, remove_both = True, minclustsize = minclustsize)



            dag_attempts = 1
            while dag_reached == False:
                  clusters_filt = list(clustid_to_clust.values())
                  cluster_orders, pos_to_clust, clustid_to_clust, clusters, dag_reached = clusters_to_dag(clusters_filt, seqs_aas, remove_both = True, minclustsize = minclustsize)          
                  #print("check Post gapfilling Dag reached?", dag_reached, " Attempt", dag_attempts)
                  if dag_attempts > 500:
                      print("Dag could not be reached")
                      return(0)
                  dag_attempts = dag_attempts + 1


            #print("Dag found, getting cluster order with topological sort of merged clusters")
            cluster_order, clustid_to_clust, pos_to_clustid =  dag_to_cluster_order(cluster_orders, seqs_aas, pos_to_clust, clustid_to_clust)

            print("prior to stranded")
            alignment = make_alignment(cluster_order, seqnums, clustid_to_clust)

            print("post stranded")
   

            if gapfilling_attempt <= 6: 
                cluster_order, clustid_to_clust = address_stranded(alignment)
                alignment = make_alignment(cluster_order, seqnums, clustid_to_clust) 
            
            #print(alignment_print(alignment, seq_names)[0])
    return( alignment,  index, hidden_states,  index_to_aa)   


    

def fill_in_hopeless2(unassigned, seqs, seqs_aas, seqnums, cluster_order, clustid_to_clust, pos_to_clustid, numseqs, index, hidden_states):
    #print("Working with unassigned")

    clusters_filt = list(clustid_to_clust.values())
    for gap in unassigned:
        #print("GAP", gap)
        starting_clustid = gap[0]
        ending_clustid = gap[2]
       
        gap_seqaas = gap[1]
        gap_seqnum = list(set([x.seqnum for x in gap_seqaas]))[0]
         
        target_seqs_list = get_ranges(seqs_aas, cluster_order, starting_clustid, ending_clustid, pos_to_clustid) 

 
        #print(target_seqs_list)
        target_seqs = list(flatten(target_seqs_list))
        target_seqs = [x for x in target_seqs if not x.seqnum == gap_seqnum]
 
        #print("Working here", gap_seqaas, target_seqs)        
        # Case where there's no candidate amino acids to match toi

        # Actually just deal with these at the squish stage
        #if len(target_seqs) == 0 or len(target_seqs) > 0:
        for aa in gap_seqaas:
               clusters_filt.append([aa])
                  #print("clusters_filt", clusters_filt)
        # If candidate amino acids, search for more remote homology
        # Moving this earlier, add option to change threshold here
        #else:
        #      for aa in gap_seqaas:
        #          #print(aa)
        #          candidates = get_looser_scores(aa, index, hidden_states)    
        #          #print(candidates)
        #          for target_seq in target_seqs:
        #              #print(target_seq.index)
        #              for score in candidates:
        #                  if score[1] == target_seq.index:
        #                        #print("candidate score", target_seq, score)

                  #  
                      #print([x for x in candidates if x[1] == target_seq.index])
     # ???????? where are candidates used?
        # If it's longer, search for remote homology
   
    #for x in clusters_filt:
    #     #print("cluster_filt", x)
    
    cluster_orders, pos_to_clust, clustid_to_clust, clusters, dag_reached = clusters_to_dag(clusters_filt, seqs_aas, remove_both = True)
    #print("Filli in hopeless2 Post gapfilling Dag reached?", dag_reached)
    # HERE NEED DAG CHECK
    dag_attempts = 1
    while dag_reached == False:
                  clusters_filt = list(clustid_to_clust.values())
                  cluster_orders, pos_to_clust, clustid_to_clust, clusters, dag_reached = clusters_to_dag(clusters_filt, seqs_aas, remove_both = True)
                  #print("check Post gapfilling Dag reached?", dag_reached, " Attempt", dag_attempts)
                  if dag_attempts > 15:
                      #print("Dag could not be reached")
                      return(0)
                  dag_attempts = dag_attempts + 1


    #print("Dag found, getting cluster order with topological sort of merged clusters")
    cluster_order, clustid_to_clust, pos_to_clustid =  dag_to_cluster_order(cluster_orders, seqs_aas, pos_to_clust, clustid_to_clust)
 

    #print("First gap filling alignment")
    alignment = make_alignment(cluster_order, seqnums, clustid_to_clust)
    #print(alignment_print(alignment, seq_names)[0])
    return(cluster_order, clustid_to_clust, pos_to_clustid, alignment)


    

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

def fill_in_unassigned3(unassigned, seqs, seqs_aas, cluster_order, clustid_to_clust, pos_to_clustid, numseqs, index, hidden_states,  index_to_aa):

    '''
    Try group based assignment, this time using new search for each unassigned
    Decision between old cluster and new cluster?
    
    '''
    clusters = list(clustid_to_clust.values())
    matches = []
    #print("unassigned")
    #for x in unassigned:
       #print("unassigned", x)

    match_scores = []
    unassigned = format_gaps(unassigned, max(clustid_to_clust.keys())) 
    for gap in unassigned:

        starting_clustid =  gap[0]
        ending_clustid = gap[2]
        

        for gap_aa in gap[1]:
            candidates_aa = get_set_of_scores(gap_aa, index, hidden_states, index_to_aa)

            # For each clustid_to_clust, it should be checked for consistency. 
            output = get_best_matches(starting_clustid, ending_clustid, gap_aa, clustid_to_clust, candidates_aa)
            if output:
               matches.append(output)
    for x in matches:
        print(x)
    clustid_to_clust = get_best_of_matches(clustid_to_clust, matches)

    #print("PREMARKER", clustid_to_clust)
    clustid_to_clust = remove_doubles3(clustid_to_clust,index, hidden_states, index_to_aa)
    # match_score = [gap_aa, current_best_score, current_best_match]
    #print("MARKER", clustid_to_clust)
    return(clustid_to_clust)
       
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
      


def fill_in_unassigned2(unassigned, seqs, seqs_aas, G, clustid_to_clust):
    '''
    Try group based assignment
    Decision between old cluster and new cluster?
    
    '''
    clusters = list(clustid_to_clust.values())

    #print("unassigned")
    #for x in unassigned:
       #print("unassigned", x)
    numclusts = []
    #new_clusters = []
    matches = []
    unassigned = format_gaps(unassigned, max(clustid_to_clust.keys())) 

    for gap in unassigned:

        starting_clustid =  gap[0]
        ending_clustid = gap[2]
        #if gap[1][0].seqnum in to_exclude:
        #    #print(gap[1], gap[1][0].seqnum, "exclude", to_exclude)
        #    continue
        for gap_aa in gap[1]:
            gap_aa_cluster_max = []
            scores = []
             
            unassigned_index = G.vs.find(name = gap_aa).index
            #print("unassigned index", unassigned_index) 
            connections = G.es.select(_source = unassigned_index)
            scores = connections['weight']
      
            connected_index  = [x.target for x in connections]
            #print(connected_index)
            connected_aas = [x.target_vertex['name'] for x in connections]
            # Source vs target seems random, potentially make graph directed and unsimplified
            if gap_aa in connected_aas:
                connected_aas = [x.source_vertex['name'] for x in connections]

            #print("unassigned", gap_aa, starting_clustid, ending_clustid)
            #print(connected_aas)
            #print(scores)  
            both = list(zip(connected_aas, scores))
            #print(both)
            output = get_best_matches(starting_clustid, ending_clustid, gap_aa, clustid_to_clust, both)
            if output:
               matches.append(output)

        
    clustid_to_clust = get_best_of_matches(clustid_to_clust, matches)

    return(clustid_to_clust)
            # Get all edges from gap aa
            # Then sort out to groups 

    # For each cluster, only add in the top scoring from each sequence
def get_best_of_matches(clustid_to_clust, matches):
    for clustid in clustid_to_clust.keys():
         potential_matches = [x for x in matches if x[1] == clustid]
         
         if potential_matches :
             match_seqnums = [x[0].seqnum for x in potential_matches]
             match_seqnums = list(set(match_seqnums))
             for seqnum in match_seqnums:
                 potential_matches_seqnum = [x for x in potential_matches if x[0].seqnum == seqnum]
                 #print("seqnum: {}, matches {}".format(seqnum, potential_matches_seqnum)) 
  
                 current_bestscore = 0
                 current_bestmatch = ""
                 for match in potential_matches_seqnum:
                     if match[2] > current_bestscore:
                           current_bestscore = match[2]
                           current_bestmatch = match[0]
        

                 newclust = clustid_to_clust[clustid] + [current_bestmatch]
                 #print("Updating {} from {} to {}".format(clustid, clustid_to_clust[clustid], newclust))
                 clustid_to_clust[clustid] = newclust
    #print(clustid_to_clust)
    return(clustid_to_clust)

 
def get_best_matches(starting_clustid, ending_clustid, gap_aa, clustid_to_clust, candidates_w_score):
     # Don't update the cluster here, send it back. 
     # candidate_w_score = zipped tuple [aa, score]
    scores = []
    current_best_score = 0
    current_best_match = ""
    match_found = False
    #print(starting_clustid, ending_clustid)
    #print(gap_aa)
    #print(clustid_to_clust)
    for cand in range(starting_clustid + 1, ending_clustid):
         print("candidate", gap_aa, cand,  clustid_to_clust[cand], "bestscore", current_best_score, current_best_match)
     
         candidate_aas =  clustid_to_clust[cand]
         incluster_scores = [x for x in candidates_w_score if x[0] in candidate_aas]
         print("incluster scores", incluster_scores)
         total_incluster_score = sum([x[1] for x in incluster_scores])
         print("totla_inclucster", total_incluster_score)
         if total_incluster_score > current_best_score:
              current_best_score = total_incluster_score
              current_best_match = cand
              match_found = True


    if match_found: 
        #print("Match found!", current_best_score, current_best_match, clustid_to_clust[current_best_match]) 
        #old = clustid_to_clust[current_best_match]
        #new = old + [gap_aa]
        #clustid_to_clust[current_best_match] = new
        output = [gap_aa, current_best_match, current_best_score]
        #print("Updating cluster {} from \n{}\nto\n{}".format(current_best_match, old, new)) 
        #match_score = [gap_aa, current_best_score, current_best_match]

    else:

         #print("no match found in (existing clusters")    
         output = []
    return(output)

def removeSublist(lst):
    #https://www.geeksforgeeks.org/python-remove-sublists-that-are-present-in-another-sublist/
    curr_res = []
    result = []
    for ele in sorted(map(set, lst), key = len, reverse = True):
        if not any(ele <= req for req in curr_res):
            curr_res.append(ele)
            result.append(list(ele))
          
    return result
      

def fill_in_unassigned(unassigned, seqs, seqs_aas, seq_names, seqnums, cluster_order, clustid_to_clust, pos_to_clustid, numseqs, I2,  minscore = 0.1, minclustsize = 2, ignore_betweenness = False, betweenness_cutoff = 0.15, apply_walktrap = True ):        
    '''
    Run the same original clustering, ??? allows overwritting of previous clusters
    
    '''
    clusters_filt = list(clustid_to_clust.values())
    print("TESTING OUT CLUSTERS_FILT")
    for x in clusters_filt:
        print("preassignment clusters_filt", x)
    # extra arguments?
    #unassigned = get_unassigned_aas(seqs, pos_to_clustid)
    new_clusters = []
  
    # Not using hopelessly unassigned
    #hopelessly_unassigned = []
    #print("I2 I2 I2")

    #for key, value in I2.items():
    #         print(key, value)
    newer_rbhs = []
    for gap in unassigned:
        newer_clusters, newer_rbh = address_unassigned(gap, seqs, seqs_aas, seqnums, pos_to_clustid, cluster_order, clustid_to_clust, numseqs, I2,  minscore = minscore, ignore_betweenness = ignore_betweenness, betweenness_cutoff = betweenness_cutoff, minclustsize = minclustsize, apply_walktrap = True)

        new_clusters  = new_clusters + newer_clusters
        newer_rbhs = newer_rbhs + newer_rbh
        #hopelessly_unassigned = hopelessly_unassigned + newer_hopelessly_unassigned


    unique_rbhs = [list(x) for x in set(tuple(x) for x in newer_rbhs)]

    G2 = graph_from_rbh(unique_rbhs)
    clusters_list = dedup_clusters(new_clusters, G2, minclustsize)

    # Going to do an additional step
    # If a cluster has doubles
    # See if there is overlap with another cluster
    # See if removing the overlap amino acids resolves the doubles
    #
 


    # Remove small clusters
    new_clusters = [x for x in new_clusters if len(x) >= minclustsize]

    for x in new_clusters:
        print("All new clusters", x)

    #print("hopelessly unassigned, in fill_in")
    #print("New clusters:", new_clusters)
    # Very important to removeSublist here
    new_clusters = removeSublist(new_clusters)
    # Need to remove overlaps from clusters
    # WORKING HERE NOE
    # Get amino acids in more than one cluster, remove them. 

    #print("New clusters after sublist removal",  new_clusters) 
    aa_counter = {}
    new_clusters_flat  = flatten(new_clusters) 
    #print("flat_clusters", new_clusters_flat)
    aa_counts = Counter(new_clusters_flat)
    dupped_aas = {key for key, val in aa_counts.items() if val != 1}
    print("dupped aas", dupped_aas)

    # From doubled aas from clusters list of lists
    new_clusters = [[aa for aa in clust if aa not in dupped_aas] for clust in new_clusters]
        

    # If this makes clusters too small remove them
    new_clusters = [clust for clust in new_clusters if len(clust) >= minclustsize]

    print("WHAT is minclustsize", minclustsize)
    for x in new_clusters:
        print("All new new clusters", x)

    #print("New clusters after overlap removal",  new_clusters) 

    # HERE ASAP REMOVE DUPLICATES FROM NEW CLUSTERS 

    # Due to additional walktrap, there's always a change that a new cluster won't be entirely consistent with previous clusters. 
    # In this section, remove any members of a new cluster that would bridge between previous clusters and cause over collapse
    #print(pos_to_clustid)
    new_clusters_filt = []
    for clust in new_clusters:
         clustids = []
         posids = []
         new_additions = []
         for pos in clust:      
            #print(pos)
            if pos in pos_to_clustid.keys():
               clustid = pos_to_clustid[pos]
               clustids.append(clustid)
               posids.append(pos)
               print(pos, clustid)
            else:
                # Position wasn't previously clustered
                print("new_additions", clust,pos)
                new_additions.append(pos)
         #print("posids", posids)                  
         if len(list(set(clustids))) > 1:
            #print("new cluster contains component of multiple previous clusters. Keeping largest matched cluster")
            clustcounts = Counter(clustids)
            largest_clust = max(clustcounts, key=clustcounts.get)   
            sel_pos = [posids[x] for x in range(len(posids)) if clustids[x] == largest_clust]
            #print("Split cluster catch", clustcounts, largest_clust, posids, clustids, sel_pos)
            new_clust = sel_pos + new_additions
                
         else:
            new_clusters_filt.append(clust)             
    #print("diagnose step 2")

    for x in new_clusters_filt:
       print("new_clusters_filt", x)

    # T0o much merging happening
    # See s4-0-I, s4-1-L in cluster 19 of 0-60 ribo

    # Add check here: Don't merge if causes more than one pos from one seq
    #print("Start merge")

    #for x in new_clusters_filt:
    #      #print("New_cluster_filt", x)
    
    new_clusters_filt = removeSublist(new_clusters_filt)

    clusters_new = remove_overlap_with_old_clusters(new_clusters_filt, clusters_filt)
    clusters_merged = clusters_new + clusters_filt

    #clusters_merged = removeSublist(clusters_merged)

    #print("Get merged cluster order")
    # To do: more qc?

    # Update with two step
    dag_reached = False
    count = 0
           #return(1) 


    while dag_reached == False:

       if len(clusters_merged) < 2:
           #print("Dag not reached, no edges left to remove".format(count))
           return(1)
       #cluster_orders_dag, pos_to_clust_dag, clustid_to_clust_dag, clusters_filt, dag_reached = clusters_to_dag(clusters_filt, seqs_aas)
       cluster_orders_merge, pos_to_clust_merge, clustid_to_clust_merge, clusters_merged, dag_reached = clusters_to_dag(clusters_merged, seqs_aas, remove_both = True, minclustsize= minclustsize)
       #print("Post gapfilling Dag reached?", dag_reached)
    # HERE NEED DAG CmHECK

    #print("Dag found, getting cluster order with topological sort of merged clusters")
    cluster_order_merge, clustid_to_clust_merge, pos_to_clustid_merge =  dag_to_cluster_order(cluster_orders_merge, seqs_aas, pos_to_clust_merge, clustid_to_clust_merge)
 
    #cluster_order_merge, clustid_to_clust_merge, pos_to_clustid_merge =  clusters_to_cluster_order(clusters_merged, seqs_aas, remove_both = False)

    #print("First gap filling alignment")
    alignment = make_alignment(cluster_order_merge, seqnums, clustid_to_clust_merge)
    #print(alignment_print(alignment, seq_names)[0])
    return(cluster_order_merge, clustid_to_clust_merge, pos_to_clustid_merge, alignment)


# Make parameter actually control this
def format_sequences(fasta, extra_padding = True):
   
    # What are the arguments to this? what is test.fasta? 
    seq_names, seqs, seqs_spaced = parse_fasta_for_embed(fasta, extra_padding = extra_padding)
    
    return(seq_names, seqs, seqs_spaced)


def get_align_args():

    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--in", dest = "fasta_path", type = str, required = True,
                        help="Path to fasta")
    
    parser.add_argument("-e", "--emb", dest = "embedding_path", type = str, required = False,
                        help="Path to embeddings")

    parser.add_argument("-o", "--outfile", dest = "out_path", type = str, required = True,
                        help="Path to outfile")


    parser.add_argument("-ex", "--exclude", dest = "exclude", action = "store_true",
                        help="Exclude outlier sequences from initial alignment process")


    parser.add_argument("-fx", "--fully_exclude", dest = "fully_exclude", action = "store_true",
                        help="Additionally exclude outlier sequences from final alignment")

    parser.add_argument("-l", "--layers", dest = "layers", nargs="+", type=int, default = [-4,-3,-2,-1],
                        help="Additionally exclude outlier sequences from final alignment")
    parser.add_argument("-m", "--model", dest = "model_name",  type=str, required = True,
                        help="Model name or path to local model")

    parser.add_argument("-p", "--pca_plot", dest = "pca_plot",  action = "store_true", required = False, 
                        help="If flagged, output 2D pca plot of amino acid clusters")

    parser.add_argument("-l2", "--headnorm", dest = "headnorm",  action = "store_true", required = False, 
                        help="Take L2 normalization of each head")



     

    args = parser.parse_args()

    return(args)

def  do_pca_plot(hidden_states, index_to_aa, clustid_to_clust, outfile):

        #filter down the hidden states, TODO
        aa_to_clustid = {} 
        print(clustid_to_clust)
        for clustid, aas in clustid_to_clust.items(): 
             for aa in aas: 
                aa_to_clustid[aa] = clustid

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
        print(reduced.shape)

        colorlist = []

        clustid_to_color = {}
        for key in clustid_to_clust.keys():
           print(key)
           clustid_to_color[key] = (random.random(), random.random(),random.random())

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
              clustid = aa_to_clustid[aa] 
              print(clustid)
              color = clustid_to_color[clustid]
              print(color)
              colorlist.append(color)
              labellist.append(clustid)
        label_arr = np.array(labellist)
        color_arr = np.array(colorlist)
       
  
        for dim1 in [1,2,3,4,5]:
          for dim2 in [1,2,3,4,5]:
            if dim1 == dim2:
                continue

            plt.figure()            
            for iclust in clustid_to_clust.keys(): 
               plt.scatter(reduced[:,dim1-1][label_arr == iclust], reduced[:,dim2-1][label_arr == iclust], c = color_arr[label_arr == iclust], alpha = 0.8, label = iclust)
            plt.legend()
            plt.xlabel('component {}'.format(dim1))
            plt.ylabel('component {}'.format(dim2))


            plt.savefig("{}.pca{}{}.png".format(outfile,dim1,dim2))

        pcasave= pd.DataFrame(reduced[:,[0,1,2,3,4,5,6]])
        pcasave['clustid'] = labellist
        pcasave['color'] = colorlist
        pcasave['seq'] = seqlist
        pcasave['pos'] = poslist
        pcasave['aa'] = aalist

        print(pcasave)
        pcasave.to_csv("{}.pca.csv".format(outfile), index = False) 

if __name__ == '__main__':

    args = get_align_args()

    fasta_path = args.fasta_path
    embedding_path = args.embedding_path
    outfile = args.out_path
    exclude = args.exclude
    fully_exclude = args.fully_exclude
    layers = args.layers
    model_name = args.model_name
    pca_plot = args.pca_plot
    headnorm = args.headnorm
 
    # Keep to demonstrate effect of clustering or not
    do_clustering = True
 
    logname = "align.log"
    #print("logging at ", logname)
    log_format = "%(asctime)s::%(levelname)s::"\
             "%(filename)s::%(lineno)d::%(message)s"
    logging.basicConfig(filename=logname, level='DEBUG', format=log_format)

    #logging.info("Check for torch")
    #logging.info(torch.cuda.is_available())

    #model_name = 'prot_bert_bfd'

    logging.info("Check for torch")
    logging.info(torch.cuda.is_available())

    padding = True 
    minscore1 = 0.5

    logging.info("model: {}".format(model_name))
    logging.info("fasta: {}".format(fasta_path))
    logging.info("padding: {}".format(padding))
    logging.info("first score thresholds: {}".format(minscore1))
   

    seq_names, seqs, seqs_spaced= format_sequences(fasta_path, extra_padding = padding)
 
    print("SEQS!", seqs)    
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
                                    padding = padding)

    # Padding irrelevant at this point 
    cluster_seqnums_list, cluster_seqs_list,  cluster_names_list, cluster_hstates_list, to_exclude = get_seq_groups(seqs ,seq_names, embedding_dict, logging, exclude, do_clustering)


    aln_fasta_list = []
    excluded_records = []
    for excluded_seqnum in to_exclude:
         
         excluded_record = SeqRecord(Seq(seqs[excluded_seqnum]), id=seq_names[excluded_seqnum], description = '')
         excluded_records.append(excluded_record)
         # Option to keep poor matches out
         if fully_exclude != True:
            aln_fasta_list.append([">{}\n{}\n".format(seq_names[excluded_seqnum], seqs[excluded_seqnum])])

    with open("excluded.fasta", "w") as output_handle:
        SeqIO.write(excluded_records, output_handle, "fasta")

    alignments = []
    hidden_states_list = []
    index_to_aas_list = []

    for i in range(len(cluster_names_list)):
        group_seqs = cluster_seqs_list[i]
        group_seqnums = cluster_seqnums_list[i]
        group_names = cluster_names_list[i]
        group_embeddings = cluster_hstates_list[i] 
        print("group seqnames", group_names, group_seqnums)

        group_seqs_out = "alignment_group{}.fasta".format(i)
        group_records = []

        for j in range(len(group_seqs)):
             group_records.append(SeqRecord(Seq(group_seqs[j]), id=group_names[j], description = ''))
 
        with open(group_seqs_out, "w") as output_handle:
            SeqIO.write(group_records, output_handle, "fasta")

        alignment, index, hidden_states, index_to_aa = get_similarity_network(group_seqs, group_names, group_seqnums, group_embeddings, logging, padding = padding, minscore1 = minscore1, alignment_group = i, headnorm = headnorm)
        alignments.append(alignment)
        index_to_aas_list.append(index_to_aa)
        hidden_states_list.append(hidden_states)

        cluster_order, clustid_to_clust = clusts_from_alignment(alignment)

        print("attempt squish")       
        for rep in range(0,10):
             prevclust = alignment
             cluster_order, clustid_to_clust = squish_clusters2(alignment, index, hidden_states, index_to_aa)                
             alignment = make_alignment(cluster_order, group_seqnums, clustid_to_clust)
             if alignment == prevclust:
                    break
        alignment = make_alignment(cluster_order, group_seqnums, clustid_to_clust)
         
        # Too much unpredictable behavior, need to include scores if doing this  
        #cluster_order, clustid_to_clust = squish_clusters_longrange2(alignment)
        alignment = make_alignment(cluster_order, group_seqnums, clustid_to_clust)
        print("squish  done")

        # Need all the embeddings from the sequence
        # Need clustid_to_clust
        
        if pca_plot: 
            png_align_out = "alignment_group{}.fasta.png".format(i)
            do_pca_plot(hidden_states, index_to_aa, clustid_to_clust, outfile)

        #address_stranded(alignment)


        str_alignment = obj_aln_to_str(alignment)
      
        aln_fasta_list_group = []
        for k in range(len(str_alignment)):
               aln_fasta_list_group.append(">{}\n{}\n".format(group_names[k], str_alignment[k]))    
 
        aln_fasta_list.append(aln_fasta_list_group)
        alignments_i = alignment_print(alignment, group_names)

        


        fasta_align_out = "alignment_group{}.fasta.aln".format(i)
        fasta_align_i = alignments_i[1]
        with open(fasta_align_out, "w") as o:
              o.write(fasta_align_i)

        clustal_align_out = "alignment_group{}.clustal.aln".format(i)
        clustal_align_i = alignments_i[0]
        with open(clustal_align_out, "w") as o:
              o.write(clustal_align_i)
        # If nothing to merge
        if len(cluster_names_list) == 1 and ( len(excluded_records) == 0 or fully_exclude == True ) :
            with open(outfile, "w") as o:
                  o.write(clustal_align_i)
            sys.exit()
       
         

            
    
   
    consolidator = "mafft"
    if consolidator == "mafft":

        #print("Consolidate alignments with mafft")
    
        seq_count = 1
    
        #for x in aln_fasta_list:
            #print(x)
        

        with open("all_fastas_aln.fasta", "w") as o:
    
            with open("key_table.txt", "w") as tb:
                for k in range(len(aln_fasta_list)):
                  
                   for s in range(len(aln_fasta_list[k])):
                        #print(aln_fasta_list[k][s])
                        o.write(aln_fasta_list[k][s])
                        tb.write("{} ".format(seq_count))
                        seq_count = seq_count + 1
                   tb.write("\n")
        

        try:
            
            #os.system("mafft --clustalout --merge key_table.txt --auto all_fastas_aln.fasta > {}".format(outfile))
            os.system("singularity exec /scratch/gpfs/cmcwhite/mafft_7.475.sif mafft --clustalout --merge key_table.txt --auto all_fastas_aln.fasta > {}".format(outfile))
               
            os.system("cat {}".format(outfile))
        except Exception as E:
            print("Not doing mafft merge") 
    
    if consolidator == "embeddings":
              for i in range(len(alignments)):
                    new_array = np.array()
                    cluster_order, clustid_to_clust = clusts_from_alignment(alignments[i])
                    print(cluster_order)
                    print(clustid_to_clust)
                    for key, value in clustid_to_clust.items():
                          clustid_embeddings = []
                          indexes = [x.index for x in value]
                          print("indexes", indexes)
                          clustid_embeddings = np.take(hidden_states_list[i], indexes, 0)
                          print("clustid_embeddings", clustid_embeddings) 
                          mean_embedding = clustid_embeddings.mean(axis=0) 
                          print(mean_embedding)
                          new_array = np.append(new_array, mean_embedding, axis=1)





# Train independent indices?
# Evaluate first. 
def consolidate_w_clustering(clusters_dict, seqs_aas_dict):
    return(0) 


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
#          #print("c1", c1)
#          #print("c2", c2)
#          combo = c1 + c2
#          scores = [x[2] for x in intra_clust_hits if x is not None]
#          # Ad hoc, get ones where multiple acceptable hits to second column
#          if len(scores) > (0.5 * len(c1) * len(c2)):
#              #print("An acceptable squish")
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
#    #print("Old cluster order", cluster_order)
#    cluster_order = [x for x in cluster_order if x not in removed_clustids]
#    #print("New cluster order", cluster_order)
#
#    return(cluster_order, clustid_to_clust)
#
#def remove_doubles2(cluster, rbh_list, numseqs, minclustsize):
#    """
#    Will need to resolve ties with scores
#    """
#    seqcounts = [0] * numseqs # Will each one replicated like with [[]] * n?
#    for pos in cluster:
#       seqnum = get_seqnum(pos)
#       #print(seq, seqnum)
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
#            #print(aas)       
#            degrees = []
#            for aa in aas: 
#                  degrees.append(G.degree(aa))
#                  # TODO: Get rbh to return scores
#                  # get highest score if degree tie
#                  # gap_scores.append(G
#            
#            #print(degrees)
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
#       #print(seq, seqnum)
#       seqcounts[seqnum] = seqcounts[seqnum] + 1
#    remove_list = [i for i in range(len(seqcounts)) if seqcounts[i] > 1]
#    clust = []
#    for pos in cluster:
#       seqnum =  get_seqnum(pos)
#       if seqnum in remove_list:
#          #print("{} removed from cluster {}".format(seq, i))
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
#def load_model_old(model_name):
#    logging.info("Load tokenizer")
#    #print("load tokenizer")
#    tokenizer = AutoTokenizer.from_pretrained(model_name)
#
#    logging.info("Load model")
#    #print("load model")
#    model = AutoModel.from_pretrained(model_name, output_hidden_states=True)
#
#    return(model, tokenizer)


#def split_distances_to_sequence(D, I, seqnums, index_to_aa, numseqs, padded_seqlen):
#   I_tmp = []
#   D_tmp = []
#   #X_tmp = []
#   #print(D.shape)
#   #print(I.shape)
#   # For each amino acid...
#   for i in range(len(I)):
#      #print(i)
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
#   #print("X_tmp", X_tmp)
#   #print(padded_seqlen)
#   #for x in X_tmp:
#   #   print("X_tmp X", x)
#   # Split amino acid matches into sequences 
#   D =  [D_tmp[i:i + padded_seqlen] for i in range(0, len(D_tmp), padded_seqlen)]
#   I =  [I_tmp[i:i + padded_seqlen] for i in range(0, len(I_tmp), padded_seqlen)]
#   #X =  [X_tmp[i:i + padded_seqlen] for i in range(0, len(X_tmp), padded_seqlen)]
#   #print("X_final", X)
#   #for x in X:
#   #   print("X final x",X)
#   #print(D.shape)
#   #print(I.shape)
#  
#  
#   return(D, I)
#
#
#
#
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
#   #print(seqnums)
#   #print(padded_seqlen)
#   #print(len(D))
#   #print(len(D[0][0]))
#   #print(len(I))
#   #print(len(I[0][0]))
#   #print(index_to_aa)
#   for query_i in range(len(D)):
#      query_seq = seqnums[query_i]
#      for query_aa in range(len(D[query_i])):
#           # Non-sequence padding isn't in dictionary
#           try:
#              query_id = index_to_aa[query_i * padded_seqlen + query_aa] 
#
#           except Exception as E:
#              #print("exception", query_i, padded_seqlen, query_aa)
#              continue
#           for target_i in range(len(D[query_i][query_aa])):
#               target_seq = seqnums[target_i]
#               #print(target_seq, target_i, "seq, i")
#               scores = D[query_i][query_aa][target_i]
#               if len(scores) == 0:
#                  continue
#               ids = I[query_i][query_aa][target_i]
#               #if query_seq in [4]:
#                    #print(query_id)
#                    #print("scores", scores)
#                    #print("ids", ids)
#               bestscore = scores[0]
#               bestmatch_id = ids[0]
#
#               if bestscore >= minscore:
#                  hitlist.append([query_id, bestmatch_id, bestscore])
#   #for x in hitlist:
#   #    #print("errorcheck", x)
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
#                   #print(query_id)
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


