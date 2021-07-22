from transformer_infrastructure.hf_utils import build_index
from transformer_infrastructure.run_tests import run_tests
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

import numpy as np
import torch
from transformers import AutoTokenizer, AutoModel

import os
import igraph
from pandas.core.common import flatten 
import pandas as pd 

from collections import Counter

import logging

class AA:
   def __init__(self):
       self.seqnum = ""
       self.seqpos = ""
       self.seqaa = ""
       self.index = ""
       #self.clustid = ""

   #__str__ and __repr__ are for pretty printing
   def __str__(self):
        return("{}-{}-{}".format(self.seqnum, self.seqpos, self.seqaa))

   def __repr__(self):
    return str(self)
 

# This is in the goal of finding sequences that poorly match before aligning
def graph_from_distindex(index, dist):  

    edges = []
    weights = []
    for i in range(len(index)):
       for j in range(len(index[i])):
          edge = (i, index[i, j])
          #if edge not in order_edges:
          edges.append(edge)
          weights.append(dist[i,j])

    for i in range(len(edges)):
      print(edges[i], weights[i])
       
    G = igraph.Graph.TupleList(edges=edges, directed=False)
    G.es['weight'] = weights
    return(G)

# If removing a protein leads to less of a drop in total edgeweight that other proteins



def candidate_to_remove(G, numseqs):

    weights = []  
    for i in range(numseqs):
        # Potentially put in function
        g_new = G.copy()
        vs = g_new.vs.find(name = i)
        weight = sum(g_new.es.select(_source=vs)['weight'])
        weights.append(weight)
    questionable_z = []
    print("Sequence z scores")
    for i in range(numseqs):
        others = [weights[x] for x in range(len(weights)) if x != i]
        z = (weights[i] - np.mean(others))/np.std(others)

        if z < -3:
            questionable_z.append(i)
       
    #print(questionable_z) 
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
    return(G_order)

def get_topological_sort(cluster_orders_lol):
    print("start topological sort")
    cluster_orders_nonempty = [x for x in cluster_orders_lol if len(x) > 0]
    dag_or_not = graph_from_cluster_orders(cluster_orders_nonempty).simplify().is_dag()
    # 
    

    print ("Dag or Not?, dag check immediately before topogical sort", dag_or_not)
     # Something deletd from here? maybe a print
    #if dag_or_not == False:
  
    G_order = graph_from_cluster_orders(cluster_orders_nonempty)
    G_order = G_order.simplify()

    topo_sort_indices = G_order.topological_sorting()
    cluster_order = []

    # Note: this is in vertex indices. Need to convert to name to get clustid
    for i in topo_sort_indices:
       cluster_order.append(G_order.vs[i]['name'])
    return(cluster_order) #, clustid_to_clust_dag)

def remove_order_conflicts(cluster_order, seqs_aas, pos_to_clustid):
   print("remove_order_conflicts, before: ", cluster_order)
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
              print("Order violation", posid, clustid)
              bad_clustids.append(clustid)
   cluster_order =  [x for x in cluster_order if x not in bad_clustids]
   print("remove_order_conflicts, after: ", cluster_order)
   return(cluster_order)
def remove_order_conflicts2(cluster_order, seqs_aas,numseqs, pos_to_clustid):
    """ 
    After topological sort,
    remove any clusters that conflict with sequence order 
     This doesn't seem to be working?
    """
    print("pos_to_clustid", pos_to_clustid)   
    print("cluster-order remove_order_conflict", cluster_order)  
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
                print("order_violation", order_index, clust)
           prev_cluster  = order_index
    #print(cluster_order)
    #print(clusters_w_order_conflict)
    cluster_order = [x for x in cluster_order if x not in clusters_w_order_conflict]
    return(cluster_order)
 
def make_alignment(cluster_order, seqnums, clustid_to_clust):
    # Set up a bunch of vectors of "-"
    # Replace with matches
    # cluster_order = list in the order that clusters go
    numseqs = len(seqnums)
    alignment =  [["-"] * len(cluster_order) for i in range(numseqs)]
    #print(cluster_order)
    print("test cluster order", cluster_order)
    for order in range(len(cluster_order)):
       cluster = clustid_to_clust[cluster_order[order]]
       c_dict = {}
       for x in cluster:
           #for pos in x:
           c_dict[x.seqnum]  = x.seqaa
       for seqnum_index in range(numseqs):
               try:
                  # convert list index position to actual seqnum
                  seqnum = seqnums[seqnum_index]
                  alignment[seqnum_index][order] = c_dict[seqnum]
               except Exception as E:
                   continue
    alignment_str = ""
    print("Alignment")
    for line in alignment:
       row_str = "".join(line)
       print("Align: ", row_str[0:150])
       alignment_str = alignment_str + row_str + "\n"
        
    alignment_str_list = ["".join(x) for x in alignment]


    return(alignment_str_list)

def alignment_print(alignment, seq_names):
       
        records = []
        
              
        for i in range(len(alignment)):
             #print(seq_names[i], alignment[i])
             print(alignment[i], seq_names[i])
             records.append(SeqRecord(Seq(alignment[i]), id=seq_names[i]))
        align = MultipleSeqAlignment(records)
        clustal_form = format(align, 'clustal')
        fasta_form = format(align, 'fasta')
        return(clustal_form, fasta_form)


def get_ranges(seqs_aas, cluster_order, starting_clustid, ending_clustid, pos_to_clustid):
    #print("start get ranges")
  
    #print(cluster_order, starting_clustid, ending_clustid) 

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






def address_unassigned(gap, seqs, seqs_aas, seqnums, pos_to_clustid, cluster_order, clustid_to_clust, numseqs, I2, D2, to_exclude, minscore = 0.1, ignore_betweenness = False, betweenness_cutoff = 0.45):
        new_clusters = []

        starting_clustid = gap[0]
        ending_clustid = gap[2] 
        gap_seqnum = gap[3]
        gap_seqaas = gap[1]
        print('EXCLUDE', to_exclude)
        if gap_seqnum in to_exclude:
              return([], [])
        target_seqs_list = get_ranges(seqs_aas, cluster_order, starting_clustid, ending_clustid, pos_to_clustid)

        target_seqs_list[gap_seqnum] = gap_seqaas
  
       

        target_seqs = list(flatten(target_seqs_list))
    
        #print("For each of the unassigned seqs, get their top hits from the previously computed distances/indices")
 
        # Get reciprocal best hits in a limited range
        new_hitlist = []
    
        for seq in target_seqs_list:
           for query_id in seq:
               query_seqnum = query_id.seqnum
               query_seqnum_ind = seqnums.index(query_seqnum)
               seqpos = query_id.seqpos
               ind = I2[query_seqnum_ind][seqpos]
               dist = D2[query_seqnum_ind][seqpos]    
               for j in range(len(ind)):
                   ids = ind[j]
                   #print(query_id)
                   #scores = dist[j]
                   good_indices = [x for x in range(len(ids)) if ids[x] in target_seqs]
                   ids_target = [ids[g] for g in good_indices]
                   scores_target = [dist[j][g] for g in good_indices]
                   if len(ids_target) > 0:
                       bestscore = scores_target[0]
                       bestmatch_id = ids_target[0]
                       if query_seqnum == bestmatch_id.seqnum:
                            continue
                       if bestmatch_id in target_seqs:
                           if bestscore >= minscore:
                              new_hitlist.append([query_id, bestmatch_id, bestscore])#, pos_to_clustid[bestmatch_id]])


 
        new_rbh = get_rbhs(new_hitlist)
       
  

        G = graph_from_rbh(new_rbh) 
        new_clusters, hb_list  = first_clustering(G, betweenness_cutoff = betweenness_cutoff, minclustsize = 2,  ignore_betweenness = ignore_betweenness) 

        clustered_aas = list(flatten(new_clusters))

        unmatched = [x for x in gap_seqaas if not x in clustered_aas]     
        hopelessly_unmatched  = []
        #This means that's it's the last cycle
        if ignore_betweenness == True:
          
           # If no reciprocal best hits
           
           for aa in unmatched: 
                 #print("its index, ", aa.index)
                 hopelessly_unmatched.append(aa)
                 #new_clusters.append([aa])

        return(new_clusters, hopelessly_unmatched)

def get_looser_scores(aa, index, hidden_states):
     '''Get all scores with a particular amino acid''' 
     #print(hidden_states)
     #print(hidden_states.shape) 
     #print(index)
     hidden_state_aa = np.take(hidden_states, [aa.index], axis = 0)
     # Search the total number of amino acids
     # Cost of returning higher n is minimal
     #print(hidden_state_aa)
     n_aa = hidden_states.shape[0]
     D_aa, I_aa =  index.search(hidden_state_aa, k = n_aa)
     print("looser scores")
     print(aa)
     print(D_aa.tolist())
     print(I_aa.tolist())
     return(list(zip(D_aa.tolist()[0], I_aa.tolist()[0])))


      
def get_particular_score(D, I, aa1, aa2):
        ''' Not used yet '''

        #print(aa1, aa2)
        
        scores = D[aa1.seqnum][aa1.seqpos][aa2.seqnum]
        #print(scores)
        ids = I[aa1.seqnum][aa1.seqpos][aa2.seqnum]
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
    print("Address isolated aas")
    connections = []
    for cohort_aa in cohort_aas:
        score = get_particular_score(unassigned_aa, cohort_aa, D, I)
        print(unassigned_aa, cohort_aa, score)
 
    return(cluster)

def squish_clusters(cluster_order, clustid_to_clust, D, I, full_cov_numseq):
    
    '''
    This will probably not necessary
    There are cases where adjacent clusters should be one cluster. 
    If any quality scores, squish them together(tetris style)
    XA-X  ->  XAX
    X-AX  ->  XAX
    XA-X  ->  XAX
    Start with doing this at the end
    With checks for unassigned aa's could do earlier
    '''

    removed_clustids = [] 
    for i in range(len(cluster_order)-1):

       c1 = clustid_to_clust[cluster_order[i]]
       # skip cluster that was 2nd half of previous squish
       if len(c1) == 0:
         continue
       c2 = clustid_to_clust[cluster_order[i + 1]]
       c1_seqnums = [x.seqnum for x in c1]
       c2_seqnums = [x.seqnum for x in c2]
       seqnum_overlap = set(c1_seqnums).intersection(set(c2_seqnums))
       #if len(list(seqnum_overlap)) < target_n:
           #continue
       # Can't merge if two clusters already have same sequences represented
       if len(seqnum_overlap) > 0:
          continue            
       else:

          #combo = c1 + c2
          ## Only allow full coverage for now
          #if len(combo) < target_n:
          #     continue
          intra_clust_hits= []
          for aa1 in c1:
             for aa2 in c2:
                score = get_particular_score(D, I, aa1, aa2)
                if score > 0.001:
                   intra_clust_hits.append([aa1,aa2,score] )

                   # intra_clust_hits.append(x)
          print("c1", c1)
          print("c2", c2)
          combo = c1 + c2
          scores = [x[2] for x in intra_clust_hits if x is not None]
          # Ad hoc, get ones where multiple acceptable hits to second column
          if len(scores) > (0.5 * len(c1) * len(c2)):
              print("An acceptable squish")
              removed_clustids.append(cluster_order[i + 1])
              clustid_to_clust[cluster_order[i]] = combo
              clustid_to_clust[cluster_order[i + 1]] = []
          # If full tetris effect. 
          # If complete, doesn't matter
          # Change? don't worry with score?
          elif len(combo) == full_cov_numseq:
              removed_clustids.append(cluster_order[i + 1])
              clustid_to_clust[cluster_order[i]] = combo
              clustid_to_clust[cluster_order[i + 1]] = []
 
                        

    print("Old cluster order", cluster_order)
    cluster_order = [x for x in cluster_order if x not in removed_clustids]
    print("New cluster order", cluster_order)

    return(cluster_order, clustid_to_clust)

 
def remove_overlap_with_old_clusters(new_clusters, prior_clusters):
    '''
    Discard any new clusters that contain elements of old clusters
    Only modify new clusters in best match-to-cluster process
    '''
    
    aas_in_prior_clusters = list(flatten(prior_clusters))
    print("aas in prior", aas_in_prior_clusters)
      
    final_new_clusters = []
    for n in new_clusters:
        #for p in prior_clusters:
        overlap =  list(set(aas_in_prior_clusters).intersection(set(n)))
        if len(overlap) > 0:
             #print("prior", p)
             print("new with overlap of old ", n)
             continue
        elif n in final_new_clusters:
             continue
        else:
             final_new_clusters.append(n)

    return(final_new_clusters)
    
 

def merge_clusters(new_clusters, prior_clusters):

  '''
  Don't modify any old clusters with these old clusters, just replace them.

  Need to add situation where overlap = an unassigned
  '''

  combined_clusters = []


 

  
  accounted_for = []
  for p in prior_clusters:
     if p in accounted_for:
        continue

     overlaps_new = False
     for n in new_clusters:
        if n in accounted_for:
           continue
        overlap =  list(set(p).intersection(set(n)))
        if len(overlap) > 0: 
            print("overlap", overlap)
            # If the new cluster fully contains the old cluster, overwrite it
            if len(overlap) == len(p):
               print("p_old", p)
               print("n    ", n)
               print("contained in previous")
               combined_clusters.append(n)
               accounted_for.append(n)
               overlaps_new = True
               continue
            # If overlap is less than the prior cluster, 
            # It means resorting occured. 
            # Only keep things from old cluster that are in new cluster
            # And add the new cluster
            elif len(overlap) < len(p):
               print("modified clustering found")
               print("p_old", p)
               print("n    ", n)
               p_new = [x for x in p if x in n]
               print("p_new", p_new)
               combined_clusters.append(p_new)
               accounted_for.append(n)
               overlaps_new = True
               #combined_clusters.append(n)
               continue
     if overlaps_new == False:
         print("Not found to overlap with new clusters, appending", p)
         combined_clusters.append(p)

  # If a new cluster has no overlap with previous clusters
  for x in new_clusters:
      if x not in accounted_for:
           combined_clusters.append(x)

  #for x in prior_clusters:
  #    print("prior ", x)
  #for x in combined_clusters:
  #        print("combo ", x)
   
  combined_clusters = merge_clusters_no_overlaps(combined_clusters) 
  return(combined_clusters) 
        


def merge_clusters_no_overlaps(combined_clusters):
    ''' Merge clusters using graph
    Must have no overlaps

    '''
    # If slow, try alternate https://stackoverflow.com/questions/9110837/python-simple-list-merging-based-on-intersections
    
   
    reduced_edgelist = []
    
    # Add edge from first component of each cluster to rest of components in cluster
    # Simplifies clique determination
    #print("reduced edgelist")
    for cluster in combined_clusters:
       for i in range(0,len(cluster)):
            reduced_edgelist.append([cluster[0],cluster[i]])
   
    new_G = igraph.Graph.TupleList(edges=reduced_edgelist, directed=False)
    #new_G = new_G.simplify()
    merged_clustering = new_G.clusters(mode = "weak")
    clusters_merged = clustering_to_clusterlist(new_G, merged_clustering)

    clusters_merged = [remove_doubles(x, new_G) for x  in clusters_merged]
    #print("change: ", len(combined_clusters), len(clusters_merged))
    return(clusters_merged)

#def get_next_clustid(seq_aa, seq_aas, pos_to_clustid):
       


 
def remove_feedback_edges(cluster_orders_dict, clusters_filt, remove_both):
    """
    Remove both improves quality of initial alignment by remove both aas that are found out of order
    For final refinement, only remove the first one that occurs out of order

    """
    G_order = graph_from_cluster_orders(list(cluster_orders_dict.values()))
    weights = [1] * len(G_order.es)

    # Remove multiedges and self loops
    #print(G_order)
    G_order.es['weight'] = weights
    G_order = G_order.simplify(combine_edges=sum)

    dag_or_not = G_order.is_dag()
    print ("Dag or Not before remove_feedback?, ", dag_or_not)



    # The edges to remove to make a directed acyclical graph
    # Corresponds to "look backs"
    # With weight, fas, with try to remove lighter edges
    # Feedback arc sets are edges that point backward in directed graph
    
    fas = G_order.feedback_arc_set(weights = 'weight')

  
    i = 0
    to_remove = []
    for edge in G_order.es():
        source_vertex = G_order.vs[edge.source]["name"]
        target_vertex = G_order.vs[edge.target]["name"]
        if i in fas:
            to_remove.append([source_vertex, target_vertex])
        i = i + 1
   
    # REMOVE all places where seqnum is inferred from order
    remove_dict = {}
   
    print("cluster_orders_dict", cluster_orders_dict)
    
    for seqnum, clustorder in cluster_orders_dict.items():
      remove = []
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
    clusters_filt_dag = []
    #print(clusters_filt)
    for i in range(len(clusters_filt)):
         clust = []
         for aa in clusters_filt[i]:
            print(aa)
            seqnum = aa.seqnum
            remove_from = remove_dict[seqnum] 
            if i in remove_from:
                print("removing ", i, seqnum) 
            else:
               clust.append(aa)
         clusters_filt_dag.append(clust)
    print("remove feedback")
    #dag_or_not = graph_from_cluster_orders(cluster_orders_dag).is_dag()
    #print ("Dag or Not?, ", dag_or_not)

    for x in clusters_filt_dag:
           print(x)

    return(clusters_filt_dag)

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

def remove_doubles(cluster, G, minclustsize = 0, keep_higher_degree = False, check_order_consistency = False):
            ''' If a cluster contains more 1 amino acid from the same sequence, remove that sequence from cluster'''
           
      
            '''
            If 
            '''
            print("cluster", cluster)
            seqnums = [x.seqnum for x in cluster]


            clustcounts = Counter(seqnums)
            #print(clustcounts)
            to_remove = []
            for key, value in clustcounts.items():
                if value > 1:
                   to_remove.append(key)
            #print(cluster)
            #print(keep_higher_degree, to_remove)
            # If there's anything in to_remove, keep the one with highest degree
            if len(to_remove) > 0 and keep_higher_degree == True:

                 G = G.vs.select(name_in=cluster).subgraph()
                 print(G)
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
                   print("what")
                   print(cluster)
                   print(seqnums)
                   print(clustcounts) 

                cluster = [x for x in cluster if x.seqnum not in to_remove]
            if len(cluster) < minclustsize:
               return([])
            else:
                return(cluster)

#def resolve_conflicting_clusters(clusters)

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
            
    highest_degree = target_aas[np.argmax(degrees)]
    to_remove = [x for x in target_aas if x != highest_degree]
    cluster_filt = [x for x in cluster if x not in to_remove]
    return(cluster_filt)
   

def graph_from_rbh(rbh_list, directed = False):

    weights = [x[2] for x in rbh_list]
    G = igraph.Graph.TupleList(edges=rbh_list, directed = directed)
    G.es['weight'] = weights 
    G = G.simplify(combine_edges = "first")
    return(G)

def remove_doubles2(cluster, rbh_list, numseqs, minclustsize):
    """
    Will need to resolve ties with scores
    """
    seqcounts = [0] * numseqs # Will each one replicated like with [[]] * n?
    for pos in cluster:
       seqnum = get_seqnum(pos)
       #print(seq, seqnum)
       seqcounts[seqnum] = seqcounts[seqnum] + 1
    #doubled = [i for i in range(len(seqcounts)) if seqcounts[i] > 1]

    G = igraph.Graph.TupleList(edges=rbh_list, directed = False)
    G = G.simplify() 
 
    # To do: check if doing extra access hashing other places
    for seqnum in range(len(seqcounts)):
        if seqcounts[seqnum] > 1:
            aas = [x for x in cluster if get_seqnum(x) == seqnum]
            #print(aas)       
            degrees = []
            for aa in aas: 
                  degrees.append(G.degree(aa))
                  # TODO: Get rbh to return scores
                  # get highest score if degree tie
                  # gap_scores.append(G
            
            #print(degrees)
            highest_degree = aas[np.argmax(degrees)]
            to_remove = [x for x in aas if x != highest_degree]
            cluster = [x for x in cluster if x not in to_remove]


    if len(cluster) < minclustsize:
         return([])

    else:
         return(cluster)



def remove_doubles_old(cluster, numseqs, minclustsize = 3):
    """
    If a cluster has two aas from the same sequence, remove from cluster
    Also removes clusters smaller than a minimum size (default: 3)
    Parameters:
       clusters (list): [aa1, aa2, aa3, aa4]
                       with aa format of sX-X-N
       numseqs (int): Total number of sequence in alignment
       minclustsize (int):
    Returns:
       filtered cluster_list 
    Could do this with cluster orders instead
    """
    #clusters_filt = []
    #for i in range(len(clusters_list)): 
    seqcounts = [0] * numseqs # Will each one replicated like with [[]] * n?
    for pos in cluster:
       seqnum = get_seqnum(pos)
       #print(seq, seqnum)
       seqcounts[seqnum] = seqcounts[seqnum] + 1
    remove_list = [i for i in range(len(seqcounts)) if seqcounts[i] > 1]
    clust = []
    for pos in cluster:
       seqnum =  get_seqnum(pos)
       if seqnum in remove_list:
          print("{} removed from cluster {}".format(seq, i))
          continue
       else:
          clust.append(pos)
    if len(clust) < minclustsize:
         return([])

    else:
         return(clust)




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
            print("Seq {} is poorly matching, fraction positions matched {}, removing until later".format(i, matched_prop[i]))
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

#ERROR HERE, w/ indexing somehow
def split_distances_to_sequence(D, I, seqnums, index_to_aa, numseqs, padded_seqlen):
   I_tmp = []
   D_tmp = []
   print(D.shape)
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
              # ISSUE LOCTION
              I_query[seqnum_index].append(aa) 
              D_query[seqnum_index].append(D[i][j])
           except Exception as E:
               continue
      I_tmp.append(I_query)
      D_tmp.append(D_query)
   print(padded_seqlen)
   D =  [D_tmp[i:i + padded_seqlen] for i in range(0, len(D_tmp), padded_seqlen)]
   I =  [I_tmp[i:i + padded_seqlen] for i in range(0, len(I_tmp), padded_seqlen)]

   return(D, I)



def get_besthits(D, I, seqnums, index_to_aa, padded_seqlen, minscore = 0.1, to_exclude = [] ):

   #aa_to_index = {value: key for key, value in index_to_aa.items()}

   hitlist = []

   #
      #          query_seqnum = query_id.seqnum
      #         query_seqnum_ind = seqnums.index(query_seqnum)
      #         seqpos = query_id.seqpos
      #         ind = I2[query_seqnum_ind][seqpos]
      #         dist = D2[query_seqnum_ind][seqpos]
   print(seqnums)
   print(len(D))
   print(len(D[0][0]))
   print(len(I))
   print(len(I[0][0]))
   print(index_to_aa)
   for query_i in range(len(D)):
      query_seq = seqnums[query_i]
      if query_seq in [3,19]:

         print(query_seq)
      # Remove this
      if query_seq in to_exclude: 
          print("excluding?")
          continue
      for query_aa in range(len(D[query_i])):
           # Non-sequence padding isn't in dictionary
           try:
              query_id = index_to_aa[query_i * padded_seqlen + query_aa] 

           except Exception as E:
              #print("exception", query_i, padded_seqlen, query_aa)
              continue
           # THE ERROR IS IN SWITCHIN OUT INDEX IDS TO AA's 
           # Looks like error is earlier in getting the index 
           for target_i in range(len(D[query_i][query_aa])):
               target_seq = seqnums[target_i]
               print(target_seq, target_i, "seq, i")
               scores = D[query_i][query_aa][target_i]
               #if query_seq in [3,19]:
               #     print("scores", scores)

               if len(scores) == 0:
                  continue
               ids = I[query_i][query_aa][target_i]
               print("IDS", ids) 
               bestscore = scores[0]
               bestmatch_id = ids[0]

               if bestscore >= minscore:
                  # WHAT
                  if query_seq in [ 19]:
                      print(["preckec", query_id, bestmatch_id, bestscore])
                  hitlist.append([query_id, bestmatch_id, bestscore])
   for x in hitlist:
       print("errorcheck", x)
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

def first_clustering(G, betweenness_cutoff = 0.45, minclustsize = 0, ignore_betweenness = False):
    '''
    Get betweenness centrality
    Each node's betweenness is normalized by dividing by the number of edges that exclude that node. 
    n = number of nodes in disconnected subgraph
    correction = = ((n - 1) * (n - 2)) / 2 
    norm_betweenness = betweenness / correction 
    '''

    #G = igraph.Graph.TupleList(edges=rbh_list, directed=False)


    # Remove multiedges and self loops
    #print("Remove multiedges and self loops")
    #G = G.simplify()

    # Could estimate this with walktrap clustering with more steps if speed is an issue  
    islands = G.clusters(mode = "weak")
    print("islands", islands) 

    #bet_dict = {}
    new_subgraphs = []
    cluster_list = []
    hb_list = []
    for sub_G in islands.subgraphs():

        if ignore_betweenness == False:
            n = len(sub_G.vs())
            bet = sub_G.betweenness(directed=True) # experiment with cutoff based on n for speed
            bet_norm = []
            # Remove small subgraphs
            if n < 3:
                if n < minclustsize:
                   continue
                else:
                   print("here1", sub_G.vs()['name'])
                   cluster_list.append(sub_G.vs()['name'])
                   continue
            correction = ((n - 1) * (n - 2)) / 2
            for x in bet:
                x_norm = x / correction
                #if x_norm > 0.45:
                bet_norm.append(x_norm)
            
                #bet_dict[sub_G.vs["name"]] = norm
            sub_G.vs()['bet_norm'] = bet_norm          
            print("before", sub_G.vs()['name'])
 
            #bet_names = list(zip(sub_G.vs()['name'], bet_norm))
            # A node with bet_norm 0.5 is perfectly split between two clusters
            # Only select nodes with normalized betweenness before 0.45
            pruned_vs = sub_G.vs.select([v for v, b in enumerate(bet_norm) if b < betweenness_cutoff]) 
                
            new_G = sub_G.subgraph(pruned_vs)
    
            # It's not necesarrily a connected set since the hb nodes were removed
            connected_set = new_G.vs()['name']
            hb_list = hb_list + [x for x in sub_G.vs['name'] if x not in connected_set]
            print("High betweenness list", hb_list)

        else:
            print("We are ignoring betweenness now")
            connected_set = sub_G.vs()['name']
            print(connected_set)


        if len(connected_set) < minclustsize:
           continue

        # If a cluster doesn't contain more than one aa from each sequence, return it
        #finished = check_completeness(connected_set)
        #if finished == True: 
        #     cluster_list.append(connected_set) 
        # If it does contain doubles, do further clustering

        # Before an after trimming connected set. 

        # Need to introduce place for unassigned high betweenness edges to be incorporated to clusters. 
        # Only at second pass
        # ex s16-73-G
        # Otherwise will never be sorted/end up in own group at the end
        # Only things with zero affinity should end up in their own cluster

        # Get any new islands after high_betweenness removed
        if ignore_betweenness == False:
            sub_islands = new_G.clusters(mode = "weak")
            for sub_sub_G in sub_islands.subgraphs():
                sub_sub_connected_sets = get_new_clustering(sub_sub_G, minclustsize, G)
                print("sub_sub", sub_sub_connected_sets)
                if sub_sub_connected_sets is not None:
                      cluster_list = cluster_list + sub_sub_connected_sets

        else:
            print("Dealing with connected set, ignore betweenness")
            # Potentially break apply_walktrap False to a second step
            
            sub_sub_connected_sets = get_new_clustering(sub_G, minclustsize, G, apply_walktrap = False)
            cluster_list= cluster_list  + sub_sub_connected_set
        
 
    print("First pass clusters")
    #for x in cluster_list:
    #     print(x)
    print("First pass done")
    return(cluster_list, hb_list)


def get_new_clustering(sub_sub_G, minclustsize, G, apply_walktrap = True):

                sub_connected_set = sub_sub_G.vs()['name']
                if len(sub_connected_set) < minclustsize:
                      return([])
                print("after ", sub_connected_set)

            

 
                sub_finished = check_completeness(sub_connected_set)
                # Complete if no duplicates
                if sub_finished == True:
                    return([sub_connected_set])
                else:

                    # Start with this
                    new_clusters = []
                    if apply_walktrap:
                        clustering = sub_sub_G.community_walktrap(steps = 1).as_clustering()
                        i = 0
                        for cl_sub_G in clustering.subgraphs():
                             print("subgraph # ", i)
                             sub_sub_connected_set =  cl_sub_G.vs()['name']
                             if len(sub_sub_connected_set) < minclustsize:
                                 continue 
    
                             print("before remove doubles", sub_sub_connected_set)
                             sub_sub_connected_set = remove_doubles(sub_sub_connected_set, G, keep_higher_degree = True)
                             print("cluster: ", sub_sub_connected_set)
                             new_clusters.append(sub_sub_connected_set)
                             i = i + 1

                        return(new_clusters)
                             #return(sub_sub_connected_set)
                    else:
                        trimmed_connected_set = remove_doubles(sub_connected_set, G, keep_higher_degree = True)
                        print("after trimming by removing doubles", trimmed_connected_set)
                        if len(trimmed_connected_set) < minclustsize:
                            return([])
                        return([trimmed_connected_set])
  



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
    
    #print("start walktrap")
    clustering = G.community_walktrap(steps = 1).as_clustering()
    #print("walktrap done")
    

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

def clusters_to_dag(clusters_filt, seqs_aas, remove_both = True, dag_reached = False):
    ######################################3
    # Remove feedback loops in paths through clusters
    # For getting consensus cluster order
  
    print("status of remove_both", remove_both)
    numseqs = len(seqs_aas)
    #for x in clusters_filt:
    #     print(x)
    pos_to_clustid, clustid_to_clust = get_cluster_dict(clusters_filt, seqs_aas)
    print("test1")
    print(pos_to_clustid)
    print("test2")
    print(clustid_to_clust)
    cluster_orders_dict = get_cluster_orders(pos_to_clustid, seqs_aas)
    #print(cluster_orders_dict)
    print("test3")

    #for i in cluster_orders:
    #      print(i)
          

    #for i in range(len( cluster_orders)):
    #      print(seqs[i])
    #      print(cluster_orders[i])

    print("Find directed acyclic graph")   
    clusters_filt_dag = remove_feedback_edges(cluster_orders_dict, clusters_filt, remove_both)

    #for x in clusters_filt_dag:
    #   print(x)
    print("Feedback edges removed")

    print("Get cluster order after feedback removeal")
    pos_to_clust_dag, clustid_to_clust_dag = get_cluster_dict(clusters_filt_dag, seqs_aas)

    cluster_orders_dict = get_cluster_orders(pos_to_clust_dag, seqs_aas)

    #print(cluster_orders_dict)
    dag_or_not_func = graph_from_cluster_orders(list(cluster_orders_dict.values())).simplify().is_dag()
    print("Dag or Not? from function, ", dag_or_not_func) 

    if dag_or_not_func == True:
          dag_reached = True
    
    else:
          print("DAG not reached, will try to remove further edges")
          dag_reached = False
    clusters_filt = list(clustid_to_clust_dag.values())
    return(cluster_orders_dict, pos_to_clust_dag, clustid_to_clust_dag, clusters_filt, dag_reached)




def dag_to_cluster_order(cluster_orders_dict, seqs_aas, pos_to_clust_dag, clustid_to_clust_dag):
    print("calling_topo_sort from function")
    cluster_order = get_topological_sort(list(cluster_orders_dict.values())) 
    print("For each sequence check that the cluster order doesn't conflict with aa order")
    cluster_order = remove_order_conflicts(cluster_order, seqs_aas, pos_to_clust_dag)


    clustid_to_clust = {key:val for key, val in clustid_to_clust_dag.items() if key  in cluster_order}

    # Another dag check


    print("renumber")
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
         pos_to_clust_inorder[key] = cluster_order_dict[pos_to_clust_dag[key]]
    print("returning cluster_order_inorder, clustid_to_clust_inorder, pos_to_clust_inorder")
    return(cluster_order_inorder, clustid_to_clust_inorder, pos_to_clust_inorder)


def load_model_old(model_name):
    logging.info("Load tokenizer")
    print("load tokenizer")
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    logging.info("Load model")
    print("load model")
    model = AutoModel.from_pretrained(model_name, output_hidden_states=True)

    return(model, tokenizer)

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


def divide_sequences(layers, model, tokenizer, seqs, seq_names, padding, exclude):

    # list of hidden_states lists
    return(0)






def get_seq_groups(seqs, seq_names, embedding_dict, logging, padding, exclude):
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
 
    #print("sentence_array", sentence_array)
    s_index = build_index(sentence_array)
    s_distance, s_index2 = s_index.search(sentence_array, k = k_select)

    #print(s_distance) 
    #print(s_index2)
    G = graph_from_distindex(s_index2, s_distance)
    #print(G)
    G = G.simplify(combine_edges = "first")  # symmetrical, doesn't matter
    print(G)
    #print("not excluding?", exclude)
    if exclude == True:
        to_exclude = candidate_to_remove(G, numseqs)
        print('name', to_exclude)
        to_delete_ids = [v.index for v in G.vs if v['name'] in to_exclude]
        print('vertix_id', to_delete_ids)
        G.delete_vertices(to_delete_ids) 

        logging.info("Excluding following sequences: {}".format(",".join([str(x) for x in to_exclude])))

    else:
       logging.info("Not removing outlier sequences")
       to_exclude = []
 
    

    print("fastgreedy")
    seq_clusters = G.community_fastgreedy(weights = 'weight').as_clustering() 
       # This has about same output as fastgreedy
        #print("multilevel")
        #seq_clusters = G.community_multilevel(weights = 'weight')
  

    print(seq_clusters)
    group_hstates_list = []
    cluster_seqnums_list = []
    cluster_names_list = []
    cluster_seqs_list = []

    # TODO use two variable names for spaced and unspaced seqs
    logging.info("Removing spaces from sequences")
    if padding:
        seqs = [x.replace(" ", "")[padding:-padding] for x in seqs]
    else:
        seqs = [x.replace(" ", "") for x in seqs]



    for seq_cluster_G in seq_clusters.subgraphs():
        hstates = []
        seq_cluster = seq_cluster_G.vs()['name']
        #print(seq_cluster)
        seq_cluster.sort()
        print(seq_cluster)
        cluster_seqnums_list.append(seq_cluster)

        filter_indices = seq_cluster
        axis = 0
        group_hstates = np.take(embedding_dict['aa_embeddings'], filter_indices, axis)
        group_hstates_list.append(group_hstates)
        #Aprint(group_hstates.shape)

        cluster_names = [seq_names[i] for i in filter_indices]
        cluster_names_list.append(cluster_names)
   
        cluster_seq = [seqs[i] for i in filter_indices]
        cluster_seqs_list.append(cluster_seq)
       #print(seq_clusters)
    

    return(cluster_seqnums_list, cluster_seqs_list,  cluster_names_list, group_hstates_list, to_exclude)



def get_similarity_network(seqs, seq_names, seqnums, hstates_list, logging, padding = 10, minscore1 = 0.5, remove_outlier_sequences = True, to_exclude = []):
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
    
    print(seqs_aas)
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
    logging.info("get best hitlist")
    print("get best hitlist")
    minscore1 = 0.5

    #Don't need to propagate to_exclude, since won't be in sequence list in the first place
 
    # THIS IS LIKELY h
    hitlist_all = get_besthits(D2, I2, seqnums, index_to_aa, padded_seqlen, minscore = 0.01, to_exclude = to_exclude)




    hitlist_top = [ x for x in hitlist_all if x[2] >= minscore1]
 
    #print("hitlist_top")
    #for x in hitlist_top:
    #      print("hitlist_top", x)
    logging.info("Get reciprocal best hits")
    print("Get reciprocal best hits")
    rbh_list = get_rbhs(hitlist_top) 


    print("got reciprocal besthits")
   
    # Skip this?
    remove_streaks = False
    if remove_streaks == True:
        logging.info("Remove streak conflict matches")
        rbh_list = remove_streakbreakers(rbh_list, seqs_aas, seqnums, seqlens, streakmin = 3)
    #for x in rbh_list:
    #  print("rbh", x) 
   
    ######################################### Do walktrap clustering
    # Why isn't this directed?

    with open("testnet.csv", "w") as outfile:
          outfile.write("aa1,aa2,score\n")
          # If do reverse first, don't have to do second resort
          for x in rbh_list:
             outstring = "{},{}\n".format(x[0], x[1])        
             outfile.write(outstring)


    logging.info("Start betweenness calculation to filter cluster-connecting amino acids. Also first round clustering")

    G = graph_from_rbh(rbh_list, directed = False)
  

    #print(G.vs.find(name = "0-25-H"))
    #connections = G.es.select(_source = 242)
    #connected_aas = [x.target_vertex['name'] for x in connections]
    #print(connected_aas)
 
    #return(0) 
    clusters_list, hb_list = first_clustering(G, betweenness_cutoff = 0.45, minclustsize = 3, ignore_betweenness = False)
    print("High betweenness list, hblist: ", hb_list)


    print("start rbh_select, check this")
    logging.info("Start rbh select")
    # Filter rbhs down to just ones that are in clusters?
    # Why does remove doubles take rbhs?
    


    #clusters_filt = remove_doubles3(cluster)

    # Removing streakbreakers may still be useful
    clusters_filt = []
    for cluster in clusters_list:
         cluster_filt = remove_doubles(cluster, minclustsize = 3, keep_higher_degree = True, G= G)
         # Could just do cluster size check here
         clusters_filt.append(cluster_filt)
    for x in clusters_filt:
          print("cluster_filt1", x)
    print("Get DAG of cluster orders, removing feedback loops")

    logging.info("Get DAG of cluster orders, removing feedback loops")

    dag_reached = False
    count = 0

    # I don't think this is going
    while dag_reached == False:
      
       count = count + 1
       print("Dag finding attempt number {}".format(count))
       if count > 20:
           print("Dag not reached after {} iteractions".format(count))
           return(1) 
       cluster_orders, pos_to_clust, clustid_to_clust, clusters_filt, dag_reached = clusters_to_dag(clusters_filt, seqs_aas)
       print("Dag reached?", dag_reached) 

    print("Dag found, getting cluster order with topological sort, dag_reached = ", dag_reached)
    cluster_order, clustid_to_clust, pos_to_clustid=  dag_to_cluster_order(cluster_orders, seqs_aas, pos_to_clust, clustid_to_clust)
    print("odd region done")  
    print("Need to get new clusters_filt")
    clusters_filt = list(clustid_to_clust.values())  

    for x in clusters_filt:
          print("cluster_filt_dag", x)
    logging.info("Make alignment")
    alignment = make_alignment(cluster_order, seqnums, clustid_to_clust)
    print(alignment_print(alignment, seq_names)[0])

    logging.info("\n{}".format(alignment))

   
    # Observations:
       #Too much first character dependencyi
       # Too much end character dependency
          # Added X on beginning and end seems to fix at least for start
    print("Get sets of unaccounted for amino acids")
    # Need to 

    prev_unassigned = []

    #all_prev_unassigned = []
    #hopelessly_unassigned = []
    ignore_betweenness = False

    minclustsize = 2
    minscore = 0.1
    betweenness_cutoff = 0.42

    ############## CONTROL LOOP ###################
    for gapfilling_attempt in range(0, 10):
        gapfilling_attempt = gapfilling_attempt + 1

        unassigned = get_unassigned_aas(seqs_aas, pos_to_clustid)


        if len(unassigned) == 0:
            print("Alignment complete after {} gapfilling attempt".format(gapfilling_attempt - 1))
            return(alignment)
        else:
            print("Gap filling attempt: {}".format(gapfilling_attempt))       
            unassigned_seqs = []
            print("These are still unassigned")
            for x in unassigned:
               print(x)
               unassigned_seqs.append(x[3])
            if list(set(unassigned_seqs)) == list(set(to_exclude)):
               print("Alignment complete, following sequences excluded")
               print(to_exclude)
                


               return(alignment)

        # This is the start of the last tries
        print(unassigned)
        print(prev_unassigned)
        if unassigned == prev_unassigned:
            print("Align by placing remaining amino acids")
            alignment = fill_in_hopeless2(unassigned, seqs, seqs_aas, seqnums, cluster_order, clustid_to_clust, pos_to_clustid, numseqs, index, hidden_states, to_exclude)
            return(alignment)
 
        prev_unassigned = unassigned

        # Do one or two rounds of clustering between guideposts
        if gapfilling_attempt == 1:
            # Do clustering within guideposts
            # Don't allow modification of previous guideposts
            print("Align by rbh between guideposts")
            cluster_order, clustid_to_clust, pos_to_clustid, alignment, hopelessly_unassigned = fill_in_unassigned(unassigned, seqs, seqs_aas, seq_names, seqnums, cluster_order, clustid_to_clust, pos_to_clustid, numseqs, I2, D2, to_exclude, minscore = minscore ,minclustsize = minclustsize, ignore_betweenness = ignore_betweenness, betweenness_cutoff = betweenness_cutoff)

        # After than do a sort of aa's into previous clusters
        # First using rbh
        # Then using individual search. 
        else:
            #print(cluster_order)
            #print(clustid_to_clust) 
            print("Add aa's into previous clusters")

            # Use original rbh to fill in easy cases
            # Or maybe do updated rbh between guideposts
            if gapfilling_attempt == 2:
                print("Align by best match (original rbh") 
                clustid_to_clust = fill_in_unassigned2(unassigned, seqs, seqs_aas, G, clustid_to_clust, to_exclude)


            else:
                print("Align by best match (looser)")
                clustid_to_clust = fill_in_unassigned3(unassigned, seqs, seqs_aas, cluster_order, clustid_to_clust, pos_to_clustid, numseqs, index, hidden_states, to_exclude, index_to_aa)


            clusters_filt = list(clustid_to_clust.values())
            cluster_orders, pos_to_clust, clustid_to_clust, clusters, dag_reached = clusters_to_dag(clusters_filt, seqs_aas, remove_both = False)
            print("Post gapfilling Dag reached?", dag_reached)
            # HERE NEED DAG CHECK

            print("Dag found, getting cluster order with topological sort of merged clusters")
            cluster_order, clustid_to_clust, pos_to_clustid =  dag_to_cluster_order(cluster_orders, seqs_aas, pos_to_clust, clustid_to_clust)

            print(clustid_to_clust)
            alignment = make_alignment(cluster_order, seqnums, clustid_to_clust)
            print(alignment_print(alignment, seq_names)[0])
    return(alignment)   


def get_cluster_score(cluster, query, index, hidden_states):
   '''
   Without messing with order, match to clusters
   During gap filling, don't mess with prior clusters
   Only mess with prior clusters during feedback arc removal

   '''
   



   return(0)



def fill_in_hopeless2(unassigned, seqs, seqs_aas, seqnums, cluster_order, clustid_to_clust, pos_to_clustid, numseqs, index, hidden_states, to_exclude):
    print("Working with unassigned")

    clusters_filt = list(clustid_to_clust.values())
    for gap in unassigned:
        print("GAP", gap)
        starting_clustid = gap[0]
        ending_clustid = gap[2]
        gap_seqnum = gap[3]
        gap_seqaas = gap[1]
        if gap_seqnum in to_exclude:
              continue
        target_seqs_list = get_ranges(seqs_aas, cluster_order, starting_clustid, ending_clustid, pos_to_clustid) 

        print(target_seqs_list)
        target_seqs = list(flatten(target_seqs_list))
        target_seqs = [x for x in target_seqs if not x.seqnum == gap_seqnum]
        print(target_seqs)        
        # Case where there's no candidate amino acids to match to
        if len(target_seqs) == 0:
            for aa in gap_seqaas:
                  clusters_filt.append([aa])

        # If candidate amino acids, search for more remote homology
        # Moving this earlier, add option to change threshold here
        else:
              for aa in gap_seqaas:
                  candidates = get_looser_scores(aa, index, hidden_states)    
                  #print(z[0])
                  #print(z[1])
                  for target_seq in target_seqs:
                      print(target_seq.index)
                      for score in candidates:
                          if score[1] == target_seq.index:
                                print("candidate score", target_seq, score)
                    
                      print([x for x in candidates if x[1] == target_seq.index])

        # If it's longer, search for remote homology
   
    for x in clusters_filt:
         print("cluster_filt", x)
    
    cluster_orders, pos_to_clust, clustid_to_clust, clusters, dag_reached = clusters_to_dag(clusters_filt, seqs_aas, remove_both = False)
    print("Post gapfilling Dag reached?", dag_reached)
    # HERE NEED DAG CHECK

    print("Dag found, getting cluster order with topological sort of merged clusters")
    cluster_order, clustid_to_clust, pos_to_clustid =  dag_to_cluster_order(cluster_orders, seqs_aas, pos_to_clust, clustid_to_clust)
 
    #cluster_order_merge, clustid_to_clust_merge, pos_to_clustid_merge =  clusters_to_cluster_order(clusters_merged, seqs_aas, remove_both = False)

    #print("First gap filling alignment")
    alignment = make_alignment(cluster_order, seqnums, clustid_to_clust)
    print(alignment_print(alignment, seq_names)[0])
    return(alignment)

   
    

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

def fill_in_unassigned3(unassigned, seqs, seqs_aas, cluster_order, clustid_to_clust, pos_to_clustid, numseqs, index, hidden_states, to_exclude, index_to_aa):

    '''
    Try group based assignment, this time using new search for each unassigned
    Decision between old cluster and new cluster?
    
    '''
    clusters = list(clustid_to_clust.values())

    print("unassigned")
    for x in unassigned:
       print("unassigned", x)
    numclusts = []
    new_clusters = []
    unassigned = format_gaps(unassigned, max(clustid_to_clust.keys())) 
    for gap in unassigned:

        starting_clustid =  gap[0]
        ending_clustid = gap[2]
        
        if gap[1][0].seqnum in to_exclude:
            continue

        for gap_aa in gap[1]:
            gap_aa_cluster_max = []
            scores = []
            candidates = get_looser_scores(gap_aa, index, hidden_states)    
            print(candidates)
            candidates_aa = []
            for score in candidates:
                try:
                   target_aa = index_to_aa[score[1]]
                except Exception as E:
                   # Not all indices correspond to an aa.
                   continue
                candidates_aa.append([target_aa, score[0]])


            print(candidates_aa)
            clustid_to_clust = get_best_matches(starting_clustid, ending_clustid, gap_aa, clustid_to_clust, candidates_aa)


    return(clustid_to_clust)
       
                  #print(z[0])
                  #print(z[1])
#                  for target_seq in target_seqs:
#                      print(target_seq.index)
#                      for score in candidates:
#                          if score[1] == target_seq.index:
#                                print("candidate score", target_seq, score)
#                    
#                      print([x for x in candidates if x[1] == target_seq.index])


def fill_in_unassigned2(unassigned, seqs, seqs_aas, G, clustid_to_clust, to_exclude):
    '''
    Try group based assignment
    Decision between old cluster and new cluster?
    
    '''
    clusters = list(clustid_to_clust.values())

    print("unassigned")
    for x in unassigned:
       print("unassigned", x)
    numclusts = []
    new_clusters = []
    unassigned = format_gaps(unassigned, max(clustid_to_clust.keys())) 

    for gap in unassigned:

        starting_clustid =  gap[0]
        ending_clustid = gap[2]
        if gap[1][0].seqnum in to_exclude:
            print(gap[1], gap[1][0].seqnum, "exclude", to_exclude)
            continue
        for gap_aa in gap[1]:
            gap_aa_cluster_max = []
            scores = []
             
            unassigned_index = G.vs.find(name = gap_aa).index
            print("unassigned index", unassigned_index) 
            connections = G.es.select(_source = unassigned_index)
            scores = connections['weight']
      
            connected_index  = [x.target for x in connections]
            print(connected_index)
            connected_aas = [x.target_vertex['name'] for x in connections]
            # Source vs target seems random, potentially make graph directed and unsimplified
            if gap_aa in connected_aas:
                connected_aas = [x.source_vertex['name'] for x in connections]

            print("unassigned", gap_aa, starting_clustid, ending_clustid)
            print(connected_aas)
            print(scores)  
            both = list(zip(connected_aas, scores))
            print(both)
            clustid_to_clust = get_best_matches(starting_clustid, ending_clustid, gap_aa, clustid_to_clust, both)


    return(clustid_to_clust)
            # Get all edges from gap aa
            # Then sort out to groups 


def get_best_matches(starting_clustid, ending_clustid, gap_aa, clustid_to_clust, both):
     # both = zipped tuple [aa, score]
    scores = []
    current_best_score = 0
    current_best_match = ""
    for cand in range(starting_clustid + 1, ending_clustid):
         print("candidate", gap_aa, cand,  clustid_to_clust[cand])
     
         candidate_aas =  clustid_to_clust[cand]
         incluster_scores = [x for x in both if x[0] in candidate_aas]
         print("incluster scores", incluster_scores)
         total_incluster_score = sum([x[1] for x in incluster_scores])
         print("totla_inclucster", total_incluster_score)
         if total_incluster_score > current_best_score:
              current_best_score = total_incluster_score
              current_best_match = cand
           
    if current_best_match: 
        print("Match found!", current_best_score, current_best_match, clustid_to_clust[current_best_match]) 
        old = clustid_to_clust[current_best_match]
        new = old + [gap_aa]
        clustid_to_clust[current_best_match] = new
         
        print("Updating cluster {} from \n{}\nto\n{}".format(current_best_match, old, new)) 

    else:

       print("no match found in existing clusters")    
             
    return(clustid_to_clust)



def fill_in_unassigned(unassigned, seqs, seqs_aas, seq_names, seqnums, cluster_order, clustid_to_clust, pos_to_clustid, numseqs, I2, D2, to_exclude, minscore = 0.1, minclustsize = 2, ignore_betweenness = False, betweenness_cutoff = 0.45 ):        
    '''
    Run the same original clustering, allows overwritting of previous clusters
    
    '''
    clusters_filt = list(clustid_to_clust.values())
    #print("TESTING OUT CLUSTERS_FILT")
    #for x in clusters_filt:
    #    print(x)
    # extra arguments?
    #unassigned = get_unassigned_aas(seqs, pos_to_clustid)
    new_clusters = []
  
    hopelessly_unassigned = []
    for gap in unassigned:
        print("BETWEEENNESS BTW", betweenness_cutoff)
        newer_clusters, newer_hopelessly_unassigned = address_unassigned(gap, seqs, seqs_aas, seqnums, pos_to_clustid, cluster_order, clustid_to_clust, numseqs, I2, D2, to_exclude, minscore = minscore, ignore_betweenness = ignore_betweenness, betweenness_cutoff = betweenness_cutoff)

        new_clusters  = new_clusters + newer_clusters
        hopelessly_unassigned = hopelessly_unassigned + newer_hopelessly_unassigned

    print("hopelessly unassigned, in fill_in")
    print("New clusters:", new_clusters)
    print("diagnose s0-4-G")
    for x in new_clusters:
      if "s6-28-V" in [str(y) for y in x] :
          print(x)   

      if "s6-28-V" in [str(y) for y in x] :
          print(x)   
      

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
               #print(pos, clustid)
            else:
                # Position wasn't previously clustered
                new_additions.append(pos)
         #print("posids", posids)                  
         if len(list(set(clustids))) > 1:
            print("new cluster contains component of multiple previous clusters. Keeping largest matched cluster")
            clustcounts = Counter(clustids)
            largest_clust = max(clustcounts, key=clustcounts.get)   
            sel_pos = [posids[x] for x in range(len(posids)) if clustids[x] == largest_clust]
            print("Split cluster catch", clustcounts, largest_clust, posids, clustids, sel_pos)
            new_clust = sel_pos + new_additions
                
         else:
            new_clusters_filt.append(clust)             
    print("diagnose step 2")

    #for x in new_clusters_filt:
    #   print("new cluster", x)

    # T0o much merging happening
    # See s4-0-I, s4-1-L in cluster 19 of 0-60 ribo

    # Add check here: Don't merge if causes more than one pos from one seq
    print("Start merge")
    #clusters_merged = merge_clusters(new_clusters_filt, clusters_filt)

    for x in new_clusters_filt:
          print("New_cluster_filt", x)
    clusters_new = remove_overlap_with_old_clusters(new_clusters_filt, clusters_filt)
    clusters_merged = clusters_new + clusters_filt

    print("Get merged cluster order")
    # To do: more qc?

    # Update with two step
    dag_reached = False
    count = 0
           #return(1) 


    while dag_reached == False:

       count = count + 1
       print("Post gapfilling Dag finding attempt number {}".format(count))
       if count > 20:
           print("Post gapfilling Dag not reached after {} iteractions".format(count))
           return(1)
       #cluster_orders_dag, pos_to_clust_dag, clustid_to_clust_dag, clusters_filt, dag_reached = clusters_to_dag(clusters_filt, seqs_aas)
       cluster_orders_merge, pos_to_clust_merge, clustid_to_clust_merge, clusters_merged, dag_reached = clusters_to_dag(clusters_merged, seqs_aas, remove_both = True)
       print("Post gapfilling Dag reached?", dag_reached)
    # HERE NEED DAG CHECK

    print("Dag found, getting cluster order with topological sort of merged clusters")
    cluster_order_merge, clustid_to_clust_merge, pos_to_clustid_merge =  dag_to_cluster_order(cluster_orders_merge, seqs_aas, pos_to_clust_merge, clustid_to_clust_merge)
 
    #cluster_order_merge, clustid_to_clust_merge, pos_to_clustid_merge =  clusters_to_cluster_order(clusters_merged, seqs_aas, remove_both = False)

    #print("First gap filling alignment")
    alignment = make_alignment(cluster_order_merge, seqnums, clustid_to_clust_merge)
    print(alignment_print(alignment, seq_names)[0])
    return(cluster_order_merge, clustid_to_clust_merge, pos_to_clustid_merge, alignment, hopelessly_unassigned)


# Make parameter actually control this
def format_sequences(fasta, padding =  5):
   
    # What are the arguments to this? what is test.fasta? 
    seq_names, seqs, seqs_spaced = parse_fasta_for_embed(fasta, extra_padding = True)

    #df = pd.DataFrame.from_records(sequence_lols,  columns=['id', 'sequence_spaced'])
    #seq_names = df['id'].tolist()
    #seqs = df['sequence_spaced'].tolist() 

    #padding_aa = " X" * padding
    #padding_left = padding_aa.strip(" ")
    #
    #newseqs = []
    #for seq in seqs:
    #     newseq = padding_left + seq  # WHY does this help embedding to not have a space?
    #     newseq = newseq + padding_aa
    #     newseqs.append(newseq)
    #newseqs = newseqs
    
    return(seq_names, seqs, seqs_spaced)

 
if __name__ == '__main__':
    # Embedding not good on short sequences without context Ex. HEIAI vs. HELAI, will select terminal I for middle I, instead of context match L
    # Potentially maximize local score? 
    # Maximize # of matches
    # Compute closes embedding of each amino acid in all target sequences
    # Compute cosine to next amino acid in seq. 
    logname = "align.log"
    print("logging at ", logname)
    log_format = "%(asctime)s::%(levelname)s::"\
             "%(filename)s::%(lineno)d::%(message)s"
    logging.basicConfig(filename=logname, level='DEBUG', format=log_format)

    #logging.info("Check for torch")
    #logging.info(torch.cuda.is_available())

    #model_name = 'prot_bert_bfd'

    #logging.info("Check for torch")
    #logging.info(torch.cuda.is_available())
    model_name = '/scratch/gpfs/cmcwhite/prot_bert_bfd'
    #model_name = "/scratch/gpfs/cmcwhite/afproject_model/0_Transformer"
     #model_name = 'prot_bert_bfd'
    #seqs = ['A A H K C Q T C G K A F N R S S T L N T H A R I H Y A G N P', 'Y K C K Q C G K A F A R S G G L Q K H K R T H', 'Y E C K E C G K S F S A H S S L V T H K R T H', 'Y K C E E C G K A F N R S S N L T K H K I I H', 'A A H K C Q T C G K A F N R S S T L N T H A R I H H A G N P', 'Y K C K Q C G K A F A R S G G L Q K H K R T H', 'Y E C K E C G K S F S A H S S L V T H Y R T H', 'Y K C E E C G K A F N R S S N L T K H K I I Y']
    #seqs = ['H E A L A I', 'H E A I A L']

    #seq_names = ['seq1','seq2', 'seq3', 'seq4']

    #fasta = '/scratch/gpfs/cmcwhite/aln_datasets/quantest2/QuanTest2/Test/zf-CCHH.vie'
    fasta = '/scratch/gpfs/cmcwhite/aln_datasets/quantest2/QuanTest2/Test/Ribosomal_L1.vie'
    #fasta = '/scratch/gpfs/cmcwhite/aln_datasets/bb3_release/RV11/BB11001.tfa'

    # Very gappy
    # fasta  = '/scratch/gpfs/cmcwhite/aln_datasets/bb3_release/RV11/BBS11018.tfa'
  
    
    #fasta = "/scratch/gpfs/cmcwhite/aln_datasets/bb3_release/RV11/BB11030.tfa"

     #fasta = "ribotest.fasta"
    padding = 5 
    minscore1 = 0.5

    logging.info("model: {}".format(model_name))
    logging.info("fasta: {}".format(fasta))
    logging.info("padding: {}".format(padding))
    logging.info("first score thresholds: {}".format(minscore1))
    
   #fasta = '/scratch/gpfs/cmcwhite/quantest2/QuanTest2/Test/ung.vie'


    # Dag problem needs more work
    #fasta = "tests/znfdoubled.fasta"
    seq_names, seqs, seqs_spaced= format_sequences(fasta, padding = padding)#, truncate = [0,20])
 
    
    #layers = [ -4, -3,-2, -1]
    layers = [-4, -3, -2, -1]
    #layers = [-4, -3, -2, -1]
    #layers = [ -5, -10, -9, -8, -3] 
    #layers = [-28]
    # CHECK exclude  list correspond with actual seqnum (NOT index)
    # The exclude system is broken.
    # But no excluding anymore, just clustering...
    exclude = True
    do_clustering = False
    #model, tokenizer = load_model(model_name)

    logging.info("Get hidden states")
    print("get hidden states for each seq")
    seqs_spaced = seqs_spaced[0:20]
    #embedding_pkl = None
    #embedding_pkl = "tester_10znfs.pkl"
    #embedding_pkl = "tester_BBS11018.pkl"
    embedding_pkl = "tester_rbs2_20.pkl"
    if embedding_pkl:
       with open(embedding_pkl, "rb") as f:
             embedding_dict = pickle.load(f)
             print(embedding_dict['aa_embeddings'].shape)

    else:
        seqlens = [len(x) for x in seqs]
        embedding_dict = get_embeddings(seqs_spaced,
                                    model_name,
                                    seqlens = seqlens,
                                    get_sequence_embeddings = True,
                                    get_aa_embeddings = True,
                                    padding = 5)

   
     

    cluster_seqnums_list, cluster_seqs_list,  cluster_names_list, cluster_hstates_list, to_exclude = get_seq_groups(seqs_spaced ,seq_names, embedding_dict, logging, padding, exclude)#, do_clustering)


    aln_fasta_list = []
    print("HERE")
    print(cluster_seqnums_list)
    print(cluster_seqs_list) 
    print("THERE")
    excluded_records = []
    for excluded_seqnum in to_exclude:
         
         excluded_record = SeqRecord(Seq(seqs[excluded_seqnum]), id=seq_names[excluded_seqnum], description = '')
         excluded_records.append(excluded_record)
         aln_fasta_list.append([">{}\n{}\n".format(seq_names[excluded_seqnum], seqs[excluded_seqnum])])

    with open("excluded.fasta", "w") as output_handle:
        SeqIO.write(excluded_records, output_handle, "fasta")


    for i in range(len(cluster_names_list)):
        group_seqs = cluster_seqs_list[i]
        group_seqnums = cluster_seqnums_list[i]
        group_names = cluster_names_list[i]
        group_embeddings = cluster_hstates_list[i] 
        print(group_embeddings.shape) 

        group_seqs_out = "alignment_group{}.fasta".format(i)
        group_records = []

        for j in range(len(group_seqs)):
             group_records.append(SeqRecord(Seq(group_seqs[j]), id=group_names[j], description = ''))
 
        with open(group_seqs_out, "w") as output_handle:
            SeqIO.write(group_records, output_handle, "fasta")

        alignment = get_similarity_network(group_seqs, group_names, group_seqnums, group_embeddings, logging, padding = padding, minscore1 = minscore1, to_exclude = to_exclude )
        aln_fasta_list_group = []
        for i in range(len(alignment)):
               aln_fasta_list_group.append(">{}\n{}\n".format(group_names[i], alignment[i]))     
 
        aln_fasta_list.append(aln_fasta_list_group)
        alignments_i = alignment_print(alignment, group_names)

        

        clustal_align_out = "alignment_group{}.clustal.aln".format(i)
        clustal_align_i = alignments_i[0]
        with open(clustal_align_out, "w") as o:
              o.write(clustal_align_i)

        fasta_align_out = "alignment_group{}.fasta.aln".format(i)
        fasta_align_i = alignments_i[1]
        with open(fasta_align_out, "w") as o:
              o.write(fasta_align_i)

    print("Consolidate alignments with mafft")

    seq_count = 1

    for x in aln_fasta_list:
        print(x)
    with open("all_fastas_aln.fasta", "w") as o:

        with open("key_table.txt", "w") as tb:
            for k in range(len(aln_fasta_list)):
              
               for s in range(len(aln_fasta_list[k])):
                    print(aln_fasta_list[k][s])
                    o.write(aln_fasta_list[k][s])
                    tb.write("{} ".format(seq_count))
                    seq_count = seq_count + 1
               tb.write("\n")

    os.system("singularity exec /scratch/gpfs/cmcwhite/mafft_7.475.sif mafft --merge key_table.txt --auto all_fastas_aln.fasta > out.mafft")

    os.system("cat out.mafft")

         

               # Not doing squish for now
               #print("try squish")
               #full_cov_seqnum = numseqs - len(to_exclude)
               #cluster_order, clustid_to_clust = squish_clusters(cluster_order, clustid_to_clust, D2, I2, full_cov_seqnum)
               #logging.info("Make squished alignment")
               #alignment = make_alignment(cluster_order, numseqs, clustid_to_clust)
               #logging.info("\n{}".format(alignment))


#def fill_in_hopeless(unassigned, seqs, seqs_aas, seqnums, cluster_order, clustid_to_clust, pos_to_clustid, numseqs, index, hidden_states, to_exclude):
#
#
#
#    clusters_filt = list(clustid_to_clust.values())
#    for gap in unassigned:
#        print("GAP", gap)
#        starting_clustid = gap[0]
#        ending_clustid = gap[2]
#        gap_seqnum = gap[3]
#        gap_seqaas = gap[1]
#        if gap_seqnum in to_exclude:
#              continue
#        target_seqs_list = get_ranges(seqs_aas, cluster_order, starting_clustid, ending_clustid, pos_to_clustid) 
#        target_seqs_list[gap_seqnum]= []
#        for x in target_seqs_list:
#           print("target", x)
#        target_seqs = list(flatten(target_seqs_list))
#        if len(target_seqs) == 0:
#            for aa in gap_seqaas:
#                  clusters_filt.append([aa])
#        else:
#
#           if len(target_seqs) > 0: 
#              print("Need to fill in here")
#              #testing with s3-256-F
#              for aa in gap_seqaas:
#                  z = get_looser_scores(aa, index, hidden_states)    
#                  #print(z[0])
#                  #print(z[1])
#                  for target_seq in target_seqs:
#                      print(target_seq.index)
#                      for score in z:
#                          if score[1] == target_seq.index:
#                                print(score)
#                    
#                      print([x for x in z if x[1] == target_seq.index])
#           # Need to take care of in end == [] or start = []
#           if starting_clustid == []:
#                  starting_clustid = -1
#           if ending_clustid == []:
#                  ending_clustid == cluster_order[-1] + 1
#           if ending_clustid - starting_clustid ==2:
#               print("Just one space")
#               # Trim the hidden states to just the targets?
#               #get_looser_scores(D, I, aa, hidden_states):
#
#               #for x in target_seqs_list:
#                   
#           print("Here need to fill in with distant reaches")
#           
#
#    # Until do the else statement
#    clusters_merged = clusters_filt
#
#    print("This needs to check for whether dag is found")
#    cluster_orders_merge, pos_to_clust_merge, clustid_to_clust_merge, clusters_merged, dag_reached = clusters_to_dag(clusters_merged, seqs_aas, remove_both = False)
#
#    print("Is this where it's running into trouble") 
#   
#    print("Dag found, getting cluster order with topological sort of merged clusters")
#    cluster_order_merge, clustid_to_clust_merge, pos_to_clustid_merge =  dag_to_cluster_order(cluster_orders_merge, seqs_aas, pos_to_clust_merge, clustid_to_clust_merge)
#
#    #cluster_order_merge, clustid_to_clust_merge, pos_to_clustid_merge =  clusters_to_cluster_order(clusters_merged, seqs_aas, remove_both = False)
#
#    #print("First gap filling alignment")
#    alignment = make_alignment(cluster_order_merge, seqnums, clustid_to_clust_merge)
#
#    return(alignment)
#
#
