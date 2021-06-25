from transformer_infrastructure.hf_utils import parse_fasta, get_hidden_states, build_index
from transformer_infrastructure.run_tests import run_tests

import faiss
#import unittest
fasta = '/scratch/gpfs/cmcwhite/quantest2/QuanTest2/Test/zf-CCHH.vie'
from sentence_transformers import util
#from iteration_utilities import  duplicates

from Bio import SeqIO
import pickle
import argparse

import numpy as np
import torch
from transformers import AutoTokenizer, AutoModel

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
       #self.clustid = ""

   #__str__ and __repr__ are for pretty printing
   def __str__(self):
        return("s{}-{}-{}".format(self.seqnum, self.seqpos, self.seqaa))

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

    G = igraph.Graph.TupleList(edges=edges, directed=False)
    G.es['weight'] = weights
    return(G)

# If removing a protein leads to less of a drop in total edgeweight that other proteins

def candidate_to_remove(G, numseqs):

    weights = []  
    for i in range(numseqs):
        g_new = G.copy()
        vs = g_new.vs.find(name = i)
        weight = sum(g_new.es.select(_source=vs)['weight'])
        weights.append(weight)
    questionable_z = []
    print("Sequence z scores")
    for i in range(numseqs):
        others = [weights[x] for x in range(len(weights)) if x != i]
        z = (weights[i] - np.mean(others))/np.std(others)
        print(i,z)

        if z < -3:
            questionable_z.append(i)
       
    #print(questionable_z) 
    return(questionable_z)

def graph_from_cluster_orders(cluster_orders):
    order_edges = []
    for order in cluster_orders:
       for i in range(len(order) - 1):
          edge = (order[i], order[i + 1])
          #if edge not in order_edges:
          order_edges.append(edge)
          
          #print(edge)

    G_order = igraph.Graph.TupleList(edges=order_edges, directed=True)
    return(G_order)

def get_topological_sort(cluster_orders):
    print("start topological sort")

    cluster_orders_nonempty = [x for x in cluster_orders if len(x) > 0]
    dag_or_not = graph_from_cluster_orders(cluster_orders_nonempty).simplify().is_dag()
    # 
    

    print ("Dag or Not?, dag check immediately before topogical sort", dag_or_not)
    if dag_or_not == False:
         for x in cluster_orders_nonempty:
              print(x)
  
    G_order = graph_from_cluster_orders(cluster_orders_nonempty)
    G_order = G_order.simplify()

    topo_sort_indices = G_order.topological_sorting()
    cluster_order = []

    # Note: this is in vertex indices. Need to convert to name to get clustid
    for i in topo_sort_indices:
       cluster_order.append(G_order.vs[i]['name'])
    return(cluster_order) #, clustid_to_clust_dag)

def remove_order_conflicts(cluster_order, seqs_aas, pos_to_clustid_dag):
   print("remove_order_conflicts, before: ", cluster_order)
   bad_clustids = []
   for i in range(len(seqs_aas)):
      prevpos = -1  
      for posid in seqs_aas[i]:

          try:
              clustid = pos_to_clustid_dag[posid]
          except Exception as E:
              continue

          pos = posid.seqpos   
          if pos < prevpos:
              print("Order violation", posid, clustid)
              bad_clustids.append(clustid)
   cluster_order =  [x for x in cluster_order if x not in bad_clustids]
   print("remove_order_conflicts, after: ", cluster_order)
   return(cluster_order)
def remove_order_conflicts2(cluster_order, seqs_aas,numseqs, pos_to_clustid_dag):
    """ 
    After topological sort,
    remove any clusters that conflict with sequence order 
     This doesn't seem to be working?
    """
    print("pos_to_clustid", pos_to_clustid_dag)   
    print("cluster-order remove_order_conflict", cluster_order)  
    clusters_w_order_conflict= []
    for i in range(numseqs): 
        prev_cluster = 0
        for j in range(len(seqs_aas[i])):
           key = seqs_aas[i][j]
           try:
               clust = pos_to_clustid_dag[key]
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
 
def make_alignment(cluster_order, numseqs, clustid_to_clust):
    # Set up a bunch of vectors of "-"
    # Replace with matches
    # cluster_order = list in the order that clusters go
    alignment =  [["-"] * len(cluster_order) for i in range(numseqs)]
    print(cluster_order)
    for order in range(len(cluster_order)):
       cluster = clustid_to_clust[cluster_order[order]]
       c_dict = {}
       for x in cluster:
           #for pos in x:
           c_dict[x.seqnum]  = x.seqaa
       for seqnum in range(numseqs):
               try:
                   
                  alignment[seqnum][order] = c_dict[seqnum]
               except Exception as E:
                   continue
    alignment_str = ""
    print("Alignment")
    for line in alignment:
       row_str = "".join(line)
       print(row_str[0:150])
       alignment_str = alignment_str + row_str + "\n"
        

    return(alignment_str)

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
    for i in range(len(seqs_aas)):
            pos_list = []
            startfound = False

            # If no starting clustid, add sequence until hit ending_clustid
            if starting_clustid == -np.inf:
                 startfound = True
                
            prevclust = "" 
            for pos in seqs_aas[i]:
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





def get_unassigned_aas(seqs_aas, pos_to_clustid_dag):
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
              clust = pos_to_clustid_dag[key]
              prevclust = clust
           # If it's not in a clust, it's unsorted
           except Exception as E:
              unsorted = []
              unsorted.append(key)
              for k in range(j + 1, len(seqs_aas[i])):
                  key = seqs_aas[i][k]
                  try:
                     nextclust = pos_to_clustid_dag[key]
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

def address_unassigned(gap, seqs, seqs_aas, pos_to_clustid, cluster_order, clustid_to_clust, numseqs, I2, D2, to_exclude, minscore = 0.1, ignore_betweenness = False):
        new_clusters = []
        starting_clustid = gap[0]
        ending_clustid = gap[2] 
        gap_seqnum = gap[3]
        gap_seqaas = gap[1]
        if gap_seqnum in to_exclude:
              return([])
        target_seqs_list = get_ranges(seqs_aas, cluster_order, starting_clustid, ending_clustid, pos_to_clustid)

        target_seqs_list[gap_seqnum] = gap_seqaas
        for x in to_exclude:
            target_seqs_list[x] = []
            

        #print("these are the target seqs")
        #for x in target_seqs_list:
        #     print(x)

       

        target_seqs = list(flatten(target_seqs_list))
    
        #print("For each of the unassigned seqs, get their top hits from the previously computed distances/indices")
 
        new_hitlist = []
    
        for seq in target_seqs_list:
           for query_id in seq:
               query_seqnum = query_id.seqnum
               seqpos = query_id.seqpos
               ind = I2[query_seqnum][seqpos]
               dist = D2[query_seqnum][seqpos]    
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
                              new_hitlist.append([query_id, bestmatch_id, bestscore])#, pos_to_clustid_dag[bestmatch_id]])


 
        new_rbh = get_rbhs(new_hitlist)

        print("gapfilling rbh")
        for x in new_rbh:
      
              print(x)
  

        
        new_clusters, hb_list  = first_clustering(new_rbh, betweenness_cutoff = 0.45, minclustsize = 2,  ignore_betweenness = ignore_betweenness) 
        # INTRO HERE
 
        #if new_rbh:        
        #   new_walktrap = get_walktrap(new_rbh)
        #   for cluster in new_walktrap:
        #        if cluster not in new_clusters:
        #             # Instead of removing double, remove one with lower degree
        #             #cluster_filt = remove_doubles(cluster, numseqs, 0)
        #             cluster_filt = remove_doubles(cluster, keep_higher_degree = True, rbh_list = new_rbh)
        #
        #             #print("before remove doubles2", cluster)
        #             #print("after remove doubles2", cluster_filt)
        #                  
        #             new_clusters.append(cluster_filt)
        #             # For final unresolved, use sort order info. 

        clustered_aas = list(flatten(new_clusters))

        unmatched = [x for x in gap_seqaas if not x in clustered_aas]     
        if ignore_betweenness == True:
          
           # If no reciprocal best hits
           for aa in unmatched: 
                 new_clusters.append([aa])

        return(new_clusters)

def squish_clusters(cluster_order, clustid_to_clust, D, I, full_cov_numseq):
    
    '''
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
          #for x in intra_clust_hits:
          #   print(x)      
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

 
        
 

def merge_clusters(new_clusters, prior_clusters):

  '''
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
   
    #for x in reduced_edgelist:
    #     print(x)
    new_G = igraph.Graph.TupleList(edges=reduced_edgelist, directed=False)
    #new_G = new_G.simplify()
    merged_clustering = new_G.clusters(mode = "weak")
    clusters_merged = clustering_to_clusterlist(new_G, merged_clustering)

    clusters_merged = [remove_doubles(x) for x  in clusters_merged]
    #print("change: ", len(combined_clusters), len(clusters_merged))
    return(clusters_merged)

#def get_next_clustid(seq_aa, seq_aas, pos_to_clustid):
       


 
def remove_feedback_edges(cluster_orders, clusters_filt, remove_both):
    """
    Remove both improves quality of initial alignment by remove both aas that are found out of order
    For final refinement, only remove the first one that occurs out of order

    """
    for x in cluster_orders:
         print(x)
    G_order = graph_from_cluster_orders(cluster_orders)
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
   
    #cluster_orders_dag = []
    remove_dict = {}
    for i in range(len(cluster_orders)):
      remove = []
      for j in range(len(cluster_orders[i]) - 1):
 
           if [cluster_orders[i][j], cluster_orders[i][j +1]] in to_remove:
               #print(cluster_orders[i])
               #print(remove_both) 
               #print(cluster_orders[i][j], cluster_orders[i][j + 1])
               if remove_both == True:
                   remove.append(cluster_orders[i][j])
               remove.append(cluster_orders[i][j + 1])
           remove_dict[i] = list(set(remove))
           
    print("remove_dict", remove_dict)
    clusters_filt_dag = []
    print(clusters_filt)
    for i in range(len(clusters_filt)):
         clust = []
         for aa in clusters_filt[i]:
            seqnum = aa.seqnum
            #seqsplit = seq.split("-")
            #seqnum = int(seqsplit[0].replace("s", ""))
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

def remove_streakbreakers(hitlist, seqs_aas, seqlens, streakmin = 3):
    # Remove initial RBHs that cross a streak of matches
    # Simplify network for feedback search
    filtered_hitlist = []
    for i in range(len(seqs_aas)):
       query_prot = [x for x in hitlist if x[0].seqnum == i]
       for j in range(len(seqs_aas)):
          target_prot = [x for x in query_prot if x[1].seqnum == j]
         
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

def remove_doubles(cluster, minclustsize = 0, keep_higher_degree = False, rbh_list = [], check_order_consistency = False):
            ''' If a cluster contains more 1 amino acid from the same sequence, remove that sequence from cluster'''
           
      
            '''
            If 
            '''
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
                 rbh_sel = [x for x in rbh_list if x[0] in cluster and x[1] in cluster]
                 G = igraph.Graph.TupleList(edges=rbh_sel, directed = False)
                 G = G.simplify() 
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

def split_distances_to_sequence(D, I, index_to_aa, numseqs, padded_seqlen):
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
              I_query[seqnum].append(aa) 
              D_query[seqnum].append(D[i][j])
           except Exception as E:
               continue
      I_tmp.append(I_query)
      D_tmp.append(D_query)
   print(padded_seqlen)
   D =  [D_tmp[i:i + padded_seqlen] for i in range(0, len(D_tmp), padded_seqlen)]
   I =  [I_tmp[i:i + padded_seqlen] for i in range(0, len(I_tmp), padded_seqlen)]

   return(D, I)



def get_besthits(D, I, index_to_aa, padded_seqlen, minscore = 0.1, to_exclude = [] ):

   #aa_to_index = {value: key for key, value in index_to_aa.items()}

   hitlist = []
   for query_seq in range(len(D)):
     
      if query_seq in to_exclude: 
          continue
      for query_aa in range(len(D[query_seq])):
           # Non-sequence padding isn't in dictionary
           try:
              query_id = index_to_aa[query_seq * padded_seqlen + query_aa] 

           except Exception as E:
              continue
           
           for target_seq in range(len(D[query_seq][query_aa])):
               scores = D[query_seq][query_aa][target_seq]
               if len(scores) == 0:
                  continue
               ids = I[query_seq][query_aa][target_seq]
              
               bestscore = scores[0]
               bestmatch_id = ids[0]
               if bestscore >= minscore:
                  hitlist.append([query_id, bestmatch_id, bestscore])

   return(hitlist) 

def get_particular_score(D, I, aa1, aa2):
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

def first_clustering(rbh_list, betweenness_cutoff = 0.45, minclustsize = 0, ignore_betweenness = False):
    '''
    Get betweenness centrality
    Each node's betweenness is normalized by dividing by the number of edges that exclude that node. 
    n = number of nodes in disconnected subgraph
    correction = = ((n - 1) * (n - 2)) / 2 
    norm_betweenness = betweenness / correction 
    '''

    G = igraph.Graph.TupleList(edges=rbh_list, directed=False)


    # Remove multiedges and self loops
    #print("Remove multiedges and self loops")
    G = G.simplify()

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
            print("n", n)   
            bet_norm = []
            # Remove small subgraphs
            if n < 3:
                if n < minclustsize:
                   continue
                else:
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
                sub_sub_connected_set = get_new_clustering(sub_sub_G, minclustsize, rbh_list)
                cluster_list.append(sub_sub_connected_set)
 
    print("First pass clusters")
    for x in cluster_list:
         print(x)
    print("First pass done")
    return(cluster_list, hb_list)


def get_new_clustering(sub_sub_G, minclustsize, rbh_list):

                sub_connected_set = sub_sub_G.vs()['name']
                if len(sub_connected_set) < minclustsize:
                      return([])
                print("after ", sub_connected_set)

            

 
                sub_finished = check_completeness(sub_connected_set)
                if sub_finished == True:
                    return(sub_connected_set)
                else:
                    clustering = sub_sub_G.community_walktrap(steps = 1).as_clustering()
                    for cl_sub_G in clustering.subgraphs():
                         sub_sub_connected_set =  cl_sub_G.vs()['name']
                         if len(sub_sub_connected_set) < minclustsize:
                             continue 

                         print("before remove doubles", sub_sub_connected_set)
                         sub_sub_connected_set = remove_doubles(sub_sub_connected_set, keep_higher_degree = True, rbh_list = rbh_list)
                         print("cluster: ", sub_sub_connected_set)
                         return(sub_sub_connected_set)
           

  



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
    cluster_orders = []

    for i in range(len(seqs_aas)):
        cluster_order = []
        for j in range(len(seqs_aas[i])):

           key = seqs_aas[i][j]
           try:
              clust = cluster_dict[key]
              cluster_order.append(clust)
           except Exception as E:
              # Not every aa is sorted into a cluster
              continue
        cluster_orders.append(cluster_order)
    return(cluster_orders)

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
    cluster_orders = get_cluster_orders(pos_to_clustid, seqs_aas)
    print(cluster_orders)
    print("test3")

    #for i in cluster_orders:
    #      print(i)
          

    #for i in range(len( cluster_orders)):
    #      print(seqs[i])
    #      print(cluster_orders[i])

    print("Find directed acyclic graph")   
    clusters_filt_dag = remove_feedback_edges(cluster_orders, clusters_filt, remove_both)

    #for x in clusters_filt_dag:
    #   print(x)
    print("Feedback edges removed")

    print("Get cluster order after feedback removeal")
    pos_to_clust_dag, clustid_to_clust_dag = get_cluster_dict(clusters_filt_dag, seqs_aas)

    cluster_orders_dag = get_cluster_orders(pos_to_clust_dag, seqs_aas)

    print("Get cluster order after feedback removeal")
    pos_to_clust_dag, clustid_to_clust_dag = get_cluster_dict(clusters_filt_dag, seqs_aas)

    cluster_orders_dag = get_cluster_orders(pos_to_clust_dag, seqs_aas)

    dag_or_not_func = graph_from_cluster_orders(cluster_orders_dag).simplify().is_dag()
    print ("Dag or Not? from function, ", dag_or_not_func) 
    for x in cluster_orders_dag:
           print(x)


    if dag_or_not_func == True:
          dag_reached = True
    
    else:
          print("DAG not reached, will try to remove further edges")
          dag_reached = False
    clusters_filt = list(clustid_to_clust_dag.values())
    return(cluster_orders_dag, pos_to_clust_dag, clustid_to_clust_dag, clusters_filt, dag_reached)

#    # Recursive potential
#        count = count + 1
#        print("not yet acyclic, removing more feedback loops, iteration: ", count)
#        clusters_filt_dag = list(clustid_to_clust_dag.values())
#        cluster_orders_dag, pos_to_clust_dag = clusters_to_cluster_order(clusters_filt_dag, seqs_aas, remove_both = True, count = count, dag_reached)
#        return
#        print("WILL THIS PRINT?")
#        if count > 5:
#            
#           print("max recursion (5) in trimming to directed acyclic graph reached")
#           return(cluster_orders_dag, pos_to_clust_dag)



def dag_to_cluster_order(cluster_orders_dag, seqs_aas, pos_to_clust_dag, clustid_to_clust_dag):
    print("calling_topo_sort from function")
    cluster_order = get_topological_sort(cluster_orders_dag) 
    print("For each sequence check that the cluster order doesn't conflict with aa order")
    cluster_order = remove_order_conflicts(cluster_order, seqs_aas, pos_to_clust_dag)


    clustid_to_clust_topo = {key:val for key, val in clustid_to_clust_dag.items() if key  in cluster_order}

    # Another dag check


    print("renumber")
    cluster_order_dict = {}
    for i in range(len(cluster_order)):
        cluster_order_dict[cluster_order[i]] = i

    clustid_to_clust_inorder = {}
    pos_to_clust_inorder = {}
    cluster_order_inorder = []
    for i in range(len(cluster_order)):
         clustid_to_clust_inorder[i] = clustid_to_clust_topo[cluster_order[i]]    
         cluster_order_inorder.append(i)

    for key in pos_to_clust_dag.keys():
         pos_to_clust_inorder[key] = cluster_order_dict[pos_to_clust_dag[key]]
    print("returning cluster_order_inorder, clustid_to_clust_inorder, pos_to_clust_inorder")
    return(cluster_order_inorder, clustid_to_clust_inorder, pos_to_clust_inorder)




def get_similarity_network(layers, model_name, seqs, seq_names, logging, padding = 10, minscore1 = 0.5, remove_outlier_sequences = True, exclude = True):
    """
    Control for running whol alignment process
    Last four layers [-4, -3, -2, -1] is a good choice for layers
    seqs should be spaced
    padding tells amount of padding to remove from seqs
    model = prot_bert_bfd
    """

    
    numseqs = len(seqs)

    logging.info("Load tokenizer")
    print("load tokenizer")
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    logging.info("Load model")
    print("load model")
    model = AutoModel.from_pretrained(model_name, output_hidden_states=True)
    
    logging.info("Get hidden states")
    print("get hidden states for each seq")

    hstates_list, sentence_embeddings = get_hidden_states(seqs, model, tokenizer, layers, return_sentence = True)
    logging.info("Hidden states complete")
    print("end hidden states")

    
    if exclude == True:
        logging.info("Removing outlier sequences")
        sentence_array = np.array(sentence_embeddings) 
        s_index = build_index(sentence_array)
        s_distance, s_index2 = s_index.search(sentence_array, k = numseqs)

        prot_scores = []

        for i in range(len(s_index2)):
           #prot = s_index2[i]
           prot_score = []
           for j in range(numseqs):
                ind = s_index2[i,j]
                if ind == i:
                  continue
                prot_score.append(s_distance[i,j])
           prot_scores.append(prot_score)
    
        G = graph_from_distindex(s_index2, s_distance)
        to_exclude = candidate_to_remove(G, numseqs)
        


        logging.info("Excluding following sequences: {}".format(",".join([str(x) for x in to_exclude])))

    else:
       logging.info("Not removing outlier sequences")
       to_exclude = []

    # Drop X's from here
    #print(hstates_list.shape)
    # Remove first and last X padding
    if padding:
        logging.info("Adding {} characters of neutral padding X".format(padding))
        hstates_list = hstates_list[:,padding:-padding,:]

    padded_seqlen = hstates_list.shape[1]
    logging.info("Padded sequence length: {}".format(padded_seqlen))


    # After encoding, remove spaces from sequences
    logging.info("Removing spaces from sequences")
    if padding:
        seqs = [x.replace(" ", "")[padding:-padding] for x in seqs]
    else:
        seqs = [x.replace(" ", "") for x in seqs]
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

    #for i in range(len(seqs)):
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
        seq_aas = []
        for j in range(len(seqs[i])):
           aa = AA()
           aa.seqnum = i
           aa.seqpos = j
           aa.seqaa =  seqs[i][j]

           seq_aas.append(aa)
        seqs_aas.append(seq_aas)


    print(seqs_aas)
    index_to_aa = {}
    for i in range(len(seqs_aas)):
        for j in range(padded_seqlen):
           if j >= seqlens[i]:
             continue 
           aa = seqs_aas[i][j]
           
           index_to_aa[i * padded_seqlen + j] = aa
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
    D2, I2 = split_distances_to_sequence(D1, I1, index_to_aa, numseqs, padded_seqlen) 
    logging.info("get best hitlist")
    print("get best hitlist")
    minscore1 = 0.5

    #hitlist_all = 
    hitlist_all = get_besthits(D2, I2, index_to_aa, padded_seqlen, minscore = 0.01, to_exclude = to_exclude)

    hitlist_top = [ x for x in hitlist_all if x[2] >= minscore1]
 
    for x in hitlist_top:
          print(x)
    logging.info("Get reciprocal best hits")
    print("Get reciprocal best hits")
    rbh_list = get_rbhs(hitlist_top) 

    #for x in rbh_list:
    #     print(x)

    print("got reciprocal besthits")
   
    # This isn't the place to do this 
    remove_streaks = True 
    if remove_streaks == True:
        logging.info("Remove streak conflict matches")
        rbh_list = remove_streakbreakers(rbh_list, seqs_aas, seqlens, streakmin = 3)
    for x in rbh_list:
      print(x) 
   
    ######################################### Do walktrap clustering
    # Why isn't this directed?

    with open("testnet.csv", "w") as outfile:
          outfile.write("aa1,aa2,score\n")
          # If do reverse first, don't have to do second resort
          for x in rbh_list:
             outstring = "{},{}\n".format(x[0], x[1])        
             outfile.write(outstring)


    logging.info("Start betweenness calculation to filter cluster-connecting amino acids. Also first round clustering")
    
    clusters_list, hb_list = first_clustering(rbh_list, betweenness_cutoff = 0.45, minclustsize = 3, ignore_betweenness = False)
    print("High betweenness list, hblist: ", hb_list)

    #logging.info("Start Walktrap clustering")
    #print("Walktrap clustering")
    #clusters_list = get_walktrap(rbh_list)
 
    #for x in rbh_list:
    #   print(x) 
    #with open("clustertest.csv", "w") as outfile:
    #   for c in clusters:
    #        outstring = "{},{}\n".format(c[0], c[1])
    #        outfile.write(outstring) 
    print("start rbh_select, check this")
    logging.info("Start rbh select")
    # Filter rbhs down to just ones that are in clusters?
    # Why does remove doubles take rbhs?
    


    #clusters_filt = remove_doubles3(cluster)

    # Removing streakbreakers may still be useful
    clusters_filt = []
    for cluster in clusters_list:
         cluster_filt = remove_doubles(cluster, minclustsize = 3, keep_higher_degree = True, rbh_list = rbh_list)
         # Could just do cluster size check here
         clusters_filt.append(cluster_filt)
    for x in clusters_filt:
          print("cluster_filt1", x)
    print("Get DAG of cluster orders, removing feedback loops")

    logging.info("Get DAG of cluster orders, removing feedback loops")

    dag_reached = False
    count = 0

    while dag_reached == False:
       count = count + 1
       if count > 5:
           print("Dag not reached after {} iteractions".format(count))
           return(1) 
       cluster_orders_dag, pos_to_clust_dag, clustid_to_clust_dag, clusters_filt, dag_reached = clusters_to_dag(clusters_filt, seqs_aas)
       
    print("Dag found, getting cluster order with topological sort")
    cluster_order, clustid_to_clust_topo, pos_to_clustid_dag =  dag_to_cluster_order(cluster_orders_dag, seqs_aas, pos_to_clust_dag, clustid_to_clust_dag)
    print("odd region done")  
    print("Need to get new clusters_filt")
    clusters_filt = list(clustid_to_clust_topo.values())  

    for x in clusters_filt:
          print("cluster_filt_dag", x)
    logging.info("Make alignment")
    alignment = make_alignment(cluster_order, numseqs, clustid_to_clust_topo)
    logging.info("\n{}".format(alignment))

   
    # Observations:
       #Too much first character dependencyi
       # Too much end character dependency
          # Added X on beginning and end seems to fix at least for start
    print("Get sets of unaccounted for amino acids")

    prev_unaligned = []
    ignore_betweenness = False
    minclustsize = 2
    for gapfilling_attempt in range(0, 5):
        gapfilling_attempt = gapfilling_attempt + 1
               

        unassigned = get_unassigned_aas(seqs_aas, pos_to_clustid_dag)
        if unassigned == prev_unaligned:
            ignore_betweenness = True 
            minclustsize = 1 # Change to one once only single aa's are left??

        prev_unassigned = unassigned
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
                
               print("try squish")
               full_cov_seqnum = numseqs - len(to_exclude)
               cluster_order, clustid_to_clust_topo = squish_clusters(cluster_order, clustid_to_clust_topo, D2, I2, full_cov_seqnum)
               logging.info("Make squished alignment")
               alignment = make_alignment(cluster_order, numseqs, clustid_to_clust_topo)
               logging.info("\n{}".format(alignment))



               return(alignment)

        cluster_order, clustid_to_clust_topo, pos_to_clustid_dag, alignment = fill_in_unassigned(unassigned, seqs, seqs_aas, cluster_order, clustid_to_clust_topo, pos_to_clustid_dag, numseqs, I2, D2, to_exclude, minscore = 0.1,minclustsize = minclustsize, ignore_betweenness = ignore_betweenness)
     

    return(alignment)   



def fill_in_unassigned(unassigned, seqs, seqs_aas, cluster_order, clustid_to_clust_topo, pos_to_clustid_dag, numseqs, I2, D2, to_exclude, minscore = 0.1, minclustsize = 2, ignore_betweenness = False ):        

    clusters_filt = list(clustid_to_clust_topo.values())
    print("TESTING OUT CLUSTERS_FILT")
    for x in clusters_filt:
        print(x)
    # extra arguments?
    #unassigned = get_unassigned_aas(seqs, pos_to_clustid_dag)
    new_clusters = []
  
    for gap in unassigned:
        new_clusters  = new_clusters + address_unassigned(gap, seqs, seqs_aas, pos_to_clustid_dag, cluster_order, clustid_to_clust_topo, numseqs, I2, D2, to_exclude, minscore = minscore, ignore_betweenness = ignore_betweenness)
 
    print("New clusters:", new_clusters)
    print("diagnose s0-4-G")
    for x in new_clusters:
      if "s6-28-V" in [str(y) for y in x] :
          print(x)   

      if "s6-28-V" in [str(y) for y in x] :
          print(x)   
      

    # Due to additional walktrap, there's always a change that a new cluster won't be entirely consistent with previous clusters. 
    # In this section, remove any members of a new cluster that would bridge between previous clusters and cause over collapse
    #print(pos_to_clustid_dag)
    new_clusters_filt = []
    for clust in new_clusters:
         clustids = []
         posids = []
         new_additions = []
         for pos in clust:      
            #print(pos)
            if pos in pos_to_clustid_dag.keys():
               clustid = pos_to_clustid_dag[pos]
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
    for x in new_clusters:
      if "s6-28-V" in [str(y) for y in x] :
          print(x)   
      if "s6-28-V" in [str(y) for y in x] :
          print(x)   

    #for x in new_clusters_filt:
    #   print("new cluster", x)

    # T0o much merging happening
    # See s4-0-I, s4-1-L in cluster 19 of 0-60 ribo

    # Add check here: Don't merge if causes more than one pos from one seq
    print("Start merge")
    clusters_merged = merge_clusters(new_clusters_filt, clusters_filt)

    print("diagnose step 3")
    for x in clusters_merged:
      if "s2-4-G" in [str(y) for y in x] :
          print(x)   
      if "s2-2-G" in [str(y) for y in x] :
          print(x)   


    print("Get merged cluster order")
    # To do: more qc?

    # Update with two step
    dag_reached = False
    count = 0
           #return(1) 
    cluster_orders_merge, pos_to_clust_merge, clustid_to_clust_merge, clusters_merged, dag_reached = clusters_to_dag(clusters_merged, seqs_aas, remove_both = False)

    print("Dag found, getting cluster order with topological sort of merged clusters")
    cluster_order_merge, clustid_to_clust_merge, pos_to_clustid_merge =  dag_to_cluster_order(cluster_orders_merge, seqs_aas, pos_to_clust_merge, clustid_to_clust_merge)
 
    #cluster_order_merge, clustid_to_clust_merge, pos_to_clustid_merge =  clusters_to_cluster_order(clusters_merged, seqs_aas, remove_both = False)

    #print("First gap filling alignment")
    alignment = make_alignment(cluster_order_merge, numseqs, clustid_to_clust_merge)
    return(cluster_order_merge, clustid_to_clust_merge, pos_to_clustid_merge, alignment)


# Make parameter actually control this
def format_sequences(fasta, padding =  10, truncate = ""):
   
    # What are the arguments to this? what is test.fasta? 
    sequence_lols = parse_fasta(fasta, "test.fasta", False, truncate)

    df = pd.DataFrame.from_records(sequence_lols,  columns=['id', 'sequence', 'sequence_spaced'])
    seq_names = df['id'].tolist()
    seqs = df['sequence_spaced'].tolist() 

    padding_aa = " X" * padding
    padding_left = padding_aa.strip(" ")
  
    newseqs = []
    for seq in seqs:
         newseq = padding_left + seq  # WHY does this help embedding to not have a space?
         newseq = newseq + padding_aa
         newseqs.append(newseq)
    newseqs = newseqs
    
    return(newseqs, seq_names)

 
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
    #model_name = '/scratch/gpfs/cmcwhite/hfmodels/prot_t5_xxl_bfd'
    model_name = "/scratch/gpfs/cmcwhite/afproject_model/0_Transformer"
     #model_name = 'prot_bert_bfd'
    #seqs = ['A A H K C Q T C G K A F N R S S T L N T H A R I H Y A G N P', 'Y K C K Q C G K A F A R S G G L Q K H K R T H', 'Y E C K E C G K S F S A H S S L V T H K R T H', 'Y K C E E C G K A F N R S S N L T K H K I I H', 'A A H K C Q T C G K A F N R S S T L N T H A R I H H A G N P', 'Y K C K Q C G K A F A R S G G L Q K H K R T H', 'Y E C K E C G K S F S A H S S L V T H Y R T H', 'Y K C E E C G K A F N R S S N L T K H K I I Y']
    #seqs = ['H E A L A I', 'H E A I A L']

    #seq_names = ['seq1','seq2', 'seq3', 'seq4']

    #fasta = '/scratch/gpfs/cmcwhite/quantest2/QuanTest2/Test/zf-CCHH.vie'
    fasta = '/scratch/gpfs/cmcwhite/quantest2/QuanTest2/Test/Ribosomal_L1.vie'
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
    seqs, seq_names = format_sequences(fasta, padding = padding)#, truncate = [0,20])




    # Maybe choose different set of embeddings
    # Avoid final layer?
    # Scan for "mutations" that significantly change "meaning"
    #seqs[3] = "F Q C G L C S R S F S R R D L L L R H A R N L H"[10:]
    #seqs[3] = "F Q C G L C N R A F T R R D L L L R H A R N L H"
    #seqs[3] = "T R R D L L I R H"
    #seqs[3] = seqs[2]
    #seqs[3] = "F Q C G L C N R A F T R R D L L I R"
    #seqs[3] =  "F Q C G L C N R A F S R R D L L L R"
    #seqs[3] =  "F Q C G L C N R A F S R R D L L L R" # Adverseiral S, A, M
    #seqs[3] =  "F Q C G L C N R A F T R R D L L L R" # This one workd

    # STRAT remove difficult ones
    # Use good ones to build a profile
    # Doesn't force an alignment,
    # Flags outliers, which can then be match to the hmm if needed

    # Choice of layers affects quality
    # Some layers likely better matchers
    
    #layers = [ -4, -3,-2, -1]
    layers = [-4, -3, -2, -1]
    #layers = [-4, -3, -2, -1]
    #layers = [ -5, -10, -9, -8, -3] 
    #layers = [-28]
    exclude = False
    get_similarity_network(layers, model_name, seqs[40:60], seq_names[40:60], logging, padding = padding, minscore1 = minscore1, exclude = exclude )

    #run_tests()
    #unittest.main(buffer = True)

