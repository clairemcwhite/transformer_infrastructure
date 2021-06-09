from transformer_infrastructure.hf_utils import parse_fasta, get_hidden_states, build_index
from transformer_infrastructure.run_tests import run_tests
import faiss
#import unittest
fasta = '/scratch/gpfs/cmcwhite/quantest2/QuanTest2/Test/zf-CCHH.vie'
from sentence_transformers import util
from iteration_utilities import  duplicates

from Bio import SeqIO
import pickle
import argparse

import numpy as np
import torch
from transformers import AutoTokenizer, AutoModel

import igraph
from pandas.core.common import flatten 
import pandas as pd 

import collections


# This needs to be replaced with class xture
def get_seqnum(pos):
    seqnum = int(pos.split("-")[0].replace("s", ""))
    return(seqnum)
# This needs to be replaced with class xture

def get_seqpos(pos):
    seqpos = int(pos.split("-")[1])
    return(seqpos)
# This needs to be replaced with class xture

def get_seqaa(pos):
    seqaa = pos.split("-")[2]
    return(seqaa)

def graph_from_cluster_orders(cluster_orders):
    order_edges = []
    for order in cluster_orders:
       for i in range(len(order) - 1):
          edge = (order[i], order[i + 1])
          #if edge not in order_edges:
          order_edges.append((order[i], order[i + 1]))

    G_order = igraph.Graph.TupleList(edges=order_edges, directed=True)
    return(G_order)

def get_topological_sort(cluster_orders):

    G_order = graph_from_cluster_orders(cluster_orders)
    G_order = G_order.simplify()

    topo_sort_indices = G_order.topological_sorting()
    cluster_order = []

    # Note: this is in vertex indices. Need to convert to name to get clustid
    for i in topo_sort_indices:
       cluster_order.append(G_order.vs[i]['name'])
    return(cluster_order) #, clustid_to_clust_dag)
    
def remove_order_conflicts(cluster_order, seqs,numseqs, pos_to_cluster_dag):
    """ 
    After topological sort,
    remove any clusters that conflict with sequence order 
    """
    #print(pos_to_cluster_dag)    
    clusters_w_order_conflict= []
    for i in range(numseqs): 
        prev_cluster = 0
        for j in range(len(seqs[i])):
           key = "s{}-{}-{}".format(i, j, seqs[i][j])
           try:
               clust = pos_to_cluster_dag[key]
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
    for order in range(len(cluster_order)):
       cluster = clustid_to_clust[cluster_order[order]]
       for x in cluster:
           c_dict = {}
           for pos in x:
               c_dict[get_seqnum(x)]  = x
           for seqnum in range(numseqs):
               #Potential some extra iteration happening here, but fine
               try:
                  alignment[seqnum][order] = get_seqaa(c_dict[seqnum])
               except Exception as E:
                   continue
    alignment_str = ""
    for line in alignment:
       row_str = "".join(line)
       print(row_str)
       alignment_str = alignment_str + row_str + "\n"
    

    

    return(alignment_str)


def get_unassigned_aas(seqs, pos_to_cluster_dag):
    unassigned = []
    for i in range(len(seqs)):
        #if i == 3:
        #   continue
        prevclust = []
        nextclust = []
        unsorted = []
        last_unsorted = -1
        for j in range(len(seqs[i])):
           if j <= last_unsorted:
               continue

           key = "s{}-{}-{}".format(i, j, seqs[i][j])
           try:
              # Read to first cluster hit
              clust = pos_to_cluster_dag[key]
              prevclust = clust
           # If it's not in a clust, it's unsorted
           except Exception as E:
              unsorted = []
              unsorted.append(key)
              for k in range(j + 1, len(seqs[i])):
                  key = "s{}-{}-{}".format(i, k, seqs[i][k])
                  try:
                     nextclust = pos_to_cluster_dag[key]
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

def address_unassigned(gap, seqs, seqs_aas, pos_to_cluster, cluster_order, clustid_to_clust, numseqs, I2, D2):
        new_clusters = []
        starting_clustid = gap[0]
        ending_clustid = gap[2] 
        gap_seqnum = gap[3]
        gap_seqaas = gap[1]

        range_list = get_ranges(seqs, cluster_order, starting_clustid, ending_clustid, clustid_to_clust)

        print("better print a range list")
        for x in range_list:
            print(x)
        print("end of range list")

        target_seqs_list = range_to_seq(range_list, numseqs, seqs_aas, gap_seqnum, gap_seqaas)

        #for x in target_seqs_list:
        #     print(x)

        target_seqs = list(flatten(target_seqs_list))
        return(0)
        #for x in range_list:
        #    print(x)

        #for x in target_seqs_list:
        #    if x:
        #      print(x)

        #print(target_seqs)
    
        print("For each of the unassigned seqs, get their top hits from the previously computed distances/indices")
 
        new_hitlist = []
    
        for seq in target_seqs_list:
           for query_id in seq:
               query_seqnum = get_seqnum(query_id)
               seqpos = get_seqpos(query_id)
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
                       if query_seqnum == get_seqnum(bestmatch_id):
                            continue
                       if bestmatch_id in target_seqs:
                           if bestscore >= 0.1:
                              new_hitlist.append([query_id, bestmatch_id, bestscore])#, pos_to_cluster_dag[bestmatch_id]])
  
        new_rbh = get_rbhs(new_hitlist)
        #print("testing here")
        for x in new_rbh:
             print(x)
        if new_rbh:        
           new_walktrap = get_walktrap(new_rbh)
           for cluster in new_walktrap:
                if cluster not in new_clusters:
                     # Instead of removing double, remove one with lower degree
                     #cluster_filt = remove_doubles(cluster, numseqs, 0)
                     cluster_filt = remove_doubles2(cluster, new_rbh, numseqs, 0)
 
                     #print("before remove doubles2", cluster)
                     #print("after remove doubles2", cluster_filt)
                          
                     new_clusters.append(cluster_filt)
                     # For final unresolved, use sort order info. 

        clustered_aas = list(flatten(new_clusters))

        unmatched = [x for x in gap_seqaas if not x in clustered_aas]     
        print("unmatched", unmatched)
          
           # If no reciprocal best hits, 
        for aa in unmatched: 
              new_clusters.append([aa])
        return(new_clusters)

def merge_clusters(new_clusters, prior_clusters):
    # This messed up a merge, try one o fthe non-igraph solutions
    # Merge clusters using graph
    # If slow, try alternate https://stackoverflow.com/questions/9110837/python-simple-list-merging-based-on-intersections
    all_clusters = new_clusters + prior_clusters

    reduced_edgelist = []
    
    # Add edge from first component of each cluster to rest of components in cluster
    # Simplifies clique determination
    for cluster in all_clusters:
       for i in range(0,len(cluster)):
            reduced_edgelist.append([cluster[0],cluster[i]])

    #for x in reduced_edgelist:
    #     print(x)
    new_G = igraph.Graph.TupleList(edges=reduced_edgelist, directed=False)
    #new_G = new_G.simplify()
    merged_clustering = new_G.clusters(mode = "weak")
    clusters_merged = clustering_to_clusterlist(new_G, merged_clustering)
    return(clusters_merged)

 
def get_ranges(seqs, cluster_order, starting_clustid, ending_clustid, clustid_to_clust):

    #Correction factors for filling out ends of sequences 
    # There are more cases to test here
    #print(cluster_order) 
    #print(starting_clustid, ending_clustid)
    #print(clustid_to_clust)
    # Probably not full list of correction factors
    print(cluster_order)
    print(starting_clustid)
    print(ending_clustid)

    open_start = 1
    open_end = 0
    lastmatch = False
    # if not x evaluates to true if x is zero 
    # If unassigned sequence goes to the end of the sequence
    if not ending_clustid and ending_clustid != 0:
       ending_clustid = cluster_order[-1]      
       open_end = 1
    # If unassigned sequence extends before the sequence
    if not starting_clustid and starting_clustid != 0:
       starting_clustid = cluster_order[0]     
       open_start = 0 
       open_end = 1

    # If the unassigned stretch starts at end of matches
    if starting_clustid == cluster_order[-1]:
       open_start = 0
       lastmatch = True

    #print(starting_clustid, ending_clustid)
 
    pos_list = [[] for i in range(len(seqs))] 
    for i in range(len(seqs)):
        for clustid_index in cluster_order[cluster_order.index(starting_clustid)  + open_start : cluster_order.index(ending_clustid) + open_end]:
           
            clustid = cluster_order[clustid_index]
            prev_clust = clustid_to_clust[clustid]
            try:
                pos = [get_seqpos(x) for x in prev_clust if get_seqnum(x) == i][0]
            except Exception as E:
                #print("not found")
                continue
            #print("pos", pos)
            # Prepend any sequence before start
            if clustid == cluster_order[0]:
                #print(pos)
                #If first clustered position is zero, don't add before
                if pos == 0:
                    continue
                else:
                    for p in range(0, pos):
                         pos_list[i].append(p)
                         #seq_list[i].append(seqs_aas[i][p])
            if lastmatch == False:
                pos_list[i].append(pos)
            # Append any remaining sequence
            if clustid == cluster_order[-1]:
                  for p in range(pos + 1, len(seqs[i])): 
                      pos_list[i].append(p)
            
    range_list = [] 
    for x in pos_list:
       #print(x)
       if x: 
          range_list.append([min(x), max(x)])
       else:
          range_list.append([])
 
    #for x in range_list:
    #     print(x)
    return(range_list)


def range_to_seq(range_list, numseqs, seqs_aas, gap_seqnum, gap_seqaas):
    #unassigned = [start, [seq], end, seqnum]
    target_seqs_list = [[] for i in range(numseqs)] 

    #incomplete_seq = unassigned[3]
    for i in range(len(seqs_aas)):
        if i == gap_seqnum:
            target_seqs_list[i] = gap_seqaas
        else:
            #target_seqs_list.append([x for x in seqs_aas[i])
            if range_list[i]:
                seqsel = seqs_aas[i][range_list[i][0]:range_list[i][1] + 1]
                target_seqs_list[i] = seqsel
    return(target_seqs_list)


 
def remove_feedback_edges(cluster_orders, clusters_filt, remove_both, count = 0):
    """
    Remove both improves quality of initial alignment by remove both aas that are found out of order
    For final refinement, only remove the first one that occurs out of order

    """

    G_order = graph_from_cluster_orders(cluster_orders)
    weights = [1] * len(G_order.es)

    # Remove multiedges and self loops
    #print(G_order)
    G_order.es['weight'] = weights
    G_order = G_order.simplify(combine_edges=sum)

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
   
    cluster_orders_dag = []
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
    #print(remove_dict)
    clusters_filt_dag = []
    for i in range(len(clusters_filt)):
         clust = []
         for seq in clusters_filt[i]:
            seqsplit = seq.split("-")
            seqnum = int(seqsplit[0].replace("s", ""))
            remove_from = remove_dict[seqnum] 
            if i in remove_from:
                print("removing ", i, seqnum) 
            else:
               clust.append(seq)
         clusters_filt_dag.append(clust)

    # Recursive potential
    count = count + 1
    if not graph_from_cluster_orders(cluster_orders_dag).is_dag():
        print("not yet acyclic, removing more feedback loops, iteration: ", count)
        remove_feedback_edges(cluster_orders_dag, clusters_filt_dag, remove_both, count) 
        if count > 5:
         
           print("max recursion in trimming to directed acyclic graph reached")
           return(clusters_filt_dag)

    return(clusters_filt_dag)

def remove_streakbreakers(hitlist, seqs, seqlens, streakmin = 3):
    # Remove initial RBHs that cross a streak of matches
    # Simplify network for feedback search
    filtered_hitlist = []
    for i in range(len(seqs)):
       query_prot = [x for x in hitlist if get_seqnum(x[0]) == i]
       for j in range(len(seqs)):
          target_prot = [x for x in query_prot if get_seqnum(x[1]) == j]
         
          # check shy this is happening extra at ends of sequence
          #print("remove lookbehinds")
          prevmatch = 0
          seq_start = -1
          streak = 0

          no_lookbehinds = []
          for match_state in target_prot:
               #print(match_state)
               if get_seqpos(match_state[1]) <= seq_start:
                     #print("lookbehind prevented")
                     streak = 0 
                     continue
               no_lookbehinds.append(match_state)

               if get_seqpos(match_state[1]) - prevmatch == 1:
                  streak = streak + 1
                  if streak >= streakmin:  
                     seq_start = get_seqpos(match_state[1])
               else:
                  streak = 0
               prevmatch = get_seqpos(match_state[1])

          #print("remove lookaheads")
          prevmatch = seqlens[j]
          seq_end = seqlens[j]
          streak = 0

          filtered_target_prot = []
          for match_state in no_lookbehinds[::-1]:
               #print(match_state, streak, prevmatch)
               if get_seqpos(match_state[1]) >= seq_end:
                    #print("lookahead prevented")
                    streak = 0
                    continue
               filtered_target_prot.append(match_state)
               if prevmatch - get_seqpos(match_state[1]) == 1:
                  streak = streak + 1
                  if streak >= streakmin:  
                     seq_end = get_seqpos(match_state[1])
               else:
                  streak = 0
               prevmatch = get_seqpos(match_state[1])
 
          filtered_hitlist = filtered_hitlist + filtered_target_prot
    return(filtered_hitlist) 



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



def remove_doubles(cluster, numseqs, minclustsize = 3):
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
          #print("{} removed from cluster {}".format(seq, i))
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
              seqnum = get_seqnum(aa)
              I_query[seqnum].append(aa) 
              D_query[seqnum].append(D[i][j])
           except Exception as E:
               continue
      I_tmp.append(I_query)
      D_tmp.append(D_query)
   D =  [D_tmp[i:i + padded_seqlen] for i in range(0, len(D_tmp), padded_seqlen)]
   I =  [I_tmp[i:i + padded_seqlen] for i in range(0, len(I_tmp), padded_seqlen)]

   return(D, I)



def get_besthits(D, I, index_to_aa, padded_seqlen):

   #aa_to_index = {value: key for key, value in index_to_aa.items()}

   hitlist = []
   for query_seq in range(len(D)):
     
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
               hitlist.append([query_id, bestmatch_id, bestscore])

   return(hitlist) 


def get_rbhs(hitlist_top):
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
    cluster_ids = clustering.membership
    vertices = G.vs()["name"]
    clusters = list(zip(cluster_ids, vertices))

    clusters_list = []
    for i in range(len(clustering)):
         clusters_list.append([vertices[x] for x in clustering[i]])


    return(clusters_list)

def get_walktrap(hitlist):
    G = igraph.Graph.TupleList(edges=hitlist, directed=True)
    # Remove multiedges and self loops
    print("Remove multiedges and self loops")
    G = G.simplify()
    
    print("start walktrap")
    clustering = G.community_walktrap(steps = 1).as_clustering()
    print("walktrap done")


    clusters_list = clustering_to_clusterlist(G, clustering)
    return(clusters_list)


def get_cluster_dict(clusters, seqs):

    pos_to_cluster = {}
    clustid_to_clust = {}
    for i in range(len(clusters)):
       clust = clusters[i]
       clustid_to_clust[i] = clust 
       for seq in clust:
              pos_to_cluster[seq] = i

    return(pos_to_cluster, clustid_to_clust)
 
def get_cluster_orders(cluster_dict, seqs):
    # This is getting path of each sequence through clusters 
    cluster_orders = []

    for i in range(len(seqs)):
        cluster_order = []
        for j in range(len(seqs[i])):
           key = "s{}-{}-{}".format(i, j, seqs[i][j])
           try:
              clust = cluster_dict[key]
              cluster_order.append(clust)
           except Exception as E:
              # Not every aa is sorted into a cluster
              continue
        cluster_orders.append(cluster_order)
    return(cluster_orders)


def clusters_to_cluster_order(clusters_filt, seqs, remove_both = True):
    ######################################3
    # Remove feedback loops in paths through clusters
    # For getting consensus cluster order
  
    print("status of remove_both", remove_both)
    numseqs = len(seqs)
    pos_to_clustid, clustid_to_clust= get_cluster_dict(clusters_filt, seqs)

    #print(clustid_to_clust)
    cluster_orders = get_cluster_orders(pos_to_clustid, seqs)

    #for i in range(len( cluster_orders)):
    #      print(seqs[i])
    #      print(cluster_orders[i])

    print("Find directed acyclic graph")   
    clusters_filt_dag = remove_feedback_edges(cluster_orders, clusters_filt, remove_both)

    print("Directed acyclic graph found")
    # unnecessary? Make optional 
    # Could cause errors for a short sequence
    # 
    #print("Removed poorly matching seqs after DAG-ification")
    #clusters_filt_dag = remove_low_match_prots(numseqs, seqlens, clusters_filt_dag, threshold_min = 0.5) 

    print("Get cluster order after dag")
    pos_to_clust_dag, clustid_to_clust_dag = get_cluster_dict(clusters_filt_dag, seqs)


    cluster_orders_dag = get_cluster_orders(pos_to_clust_dag, seqs)

    print("Get a single cluster order with topological sort")

    cluster_order = get_topological_sort(cluster_orders_dag) 
 
    print("For each sequence check that the cluster order doesn't conflict with aa order")
    cluster_order = remove_order_conflicts(cluster_order, seqs,numseqs, pos_to_clust_dag)

    clustid_to_clust_topo = {key:val for key, val in clustid_to_clust_dag.items() if key  in cluster_order}

    print("renumber")
    # It's just easier if the cluster order in in order
    # Not strictly necessary with modification to find_ranges function
    # Find ranges is currently dependent on in-order cluster numbers (like 25 -> 0 or 0 -> 25
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

    return(cluster_order_inorder, clustid_to_clust_inorder, pos_to_clust_inorder)




def get_similarity_network(layers, model_name, seqs, seq_names, padding = 10):
    """
    Control for running whol alignment process
    Last four layers [-4, -3, -2, -1] is a good choice for layers
    seqs should be spaced
    padding tells amount of padding to remove from seqs
    model = prot_bert_bfd
    """

    print("load tokenizer")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    print("load model")
    model = AutoModel.from_pretrained(model_name, output_hidden_states=True)
    

    print("get hidden states for each seq")

    #hidden_states_list =[]
    #seqlen = max([len(x.replace(" ", "")) for x in seqs])
    #print(seqlen)

    #seqlens = [len(x.replace(" ", "")) for x in seqs]
    print("start hidden_states")
    #print(seqs)
    hstates_list = get_hidden_states(seqs, model, tokenizer, layers)
    print("end hidden states")
    #print(hstates_list)
    #print(hstates_list.shape)

    # Drop X's from here
    #print(hstates_list.shape)
    # Remove first and last X padding
    hstates_list = hstates_list[:,padding:-padding,:]
    padded_seqlen = hstates_list.shape[1]

    # After encoding, remove spaces from sequences
    seqs = [x.replace(" ", "")[padding:-padding] for x in seqs]
    seqlens = [len(x) for x in seqs]
    numseqs = len(seqs)
    #for seq in seqs:
    #   hidden_states = get_hidden_states([seq], model, tokenizer, layers)
    #   hidden_states_list.append(hidden_states)


    # Build index from all amino acids 
    #d = hidden_states[0].shape[1]

    # Go from (numseqs, seqlen, emb) to (numseqs * seqlen, emb)
    hidden_states = np.array(reshape_flat(hstates_list))  
    print("embedding_shape", hidden_states.shape)


 
    # Convert index position to amino acid position
    index_to_aa = {}
    for i in range(len(seqs)):
        for j in range(padded_seqlen):
           if j >= seqlens[i]:
             continue 
           aa = "s{}-{}-{}".format(i, j, seqs[i][j])    
           
           index_to_aa[i * padded_seqlen + j] = aa
    print("Build index")
   
    index = build_index(hidden_states)
     
    print("search index")
    D1, I1 =  index.search(hidden_states, k = numseqs*3) 

    print("Split results into proteins") 
    # Still annoyingly slow
    D2, I2 = split_distances_to_sequence(D1, I1, index_to_aa, numseqs, padded_seqlen) 

    print("get best hitlist")
    hitlist_top = get_besthits(D2, I2, index_to_aa, padded_seqlen)

    print("Get reciprocal best hits")
    rbh_list = get_rbhs(hitlist_top) 

    print("got reciprocal besthits")
  
    remove_streaks = False  
    if remove_streaks == True:
        print("Remove streak conflict matches")
        rbh_list = remove_streakbreakers(rbh_list, seqs, seqlens, streakmin = 3)
    
   
    ######################################### Do walktrap clustering
    # Why isn't this directed?

    with open("testnet.csv", "w") as outfile:
          outfile.write("aa1,aa2,score\n")
          # If do reverse first, don't have to do second resort
          for x in rbh_list:
             outstring = "{},{}\n".format(x[0], x[1])        
             #outstring = "s{}-{}-{},s{}-{}-{},{}\n".format(x[0], x[1], x[5], x[2], x[3], x[6], x[4])
             outfile.write(outstring)

    print("Walktrap clustering")
    clusters_list = get_walktrap(rbh_list)
 
 
    #with open("clustertest.csv", "w") as outfile:
    #   for c in clusters:
    #        outstring = "{},{}\n".format(c[0], c[1])
    #        outfile.write(outstring) 
    clusters_filt = []
    for cluster in clusters_list:
         rbh_select = []
         for rbh in rbh_list:
             if rbh[0] in cluster:
                 if rbh[1] in cluster:
                      rbh_select.append(rbh) 
         singleton_clusters = remove_doubles2(cluster, rbh_select, numseqs, 3)
         if singleton_clusters:
                clusters_filt.append(singleton_clusters)

    #clusters_filt = [remove_doubles(x, numseqs, 3) for x in clusters_list]
    clusters_filt = remove_low_match_prots(numseqs, seqlens, clusters_filt, threshold_min = 0.1) 
    print("Remove poorly matching seqs after initial RBH seach")

    cluster_order, clustid_to_clust_topo, pos_to_cluster_dag =  clusters_to_cluster_order(clusters_filt, seqs)

    print("Need to get new clusters_filt")
    clusters_filt = list(clustid_to_clust_topo.values())  

 
    make_alignment(cluster_order, numseqs, clustid_to_clust_topo)
   

    # Write sequences with aa ids
    seqs_aas = []
    for i in range(len(seqs)):
        seq_aas = []
        for j in range(len(seqs[i])):
           seq_aas.append("s{}-{}-{}".format(i, j, seqs[i][j]))
        seqs_aas.append(seq_aas)

 
    
    # Observations:
       #Too much first character dependencyi
       # Too much end character dependency
          # Added X on beginning and end seems to fix at least for start
    print("Get sets of unaccounted for amino acids")



    unassigned = get_unassigned_aas(seqs, pos_to_cluster_dag)


    for x in unassigned:
          print(x)
    # Get sequence between two clusters
    
    

    new_clusters = []
  
    for gap in unassigned:
        new_clusters  = new_clusters + address_unassigned(gap, seqs, seqs_aas, pos_to_cluster_dag, cluster_order, clustid_to_clust_topo, numseqs, I2, D2)

    clusters_merged = merge_clusters(new_clusters, clusters_filt)


    print("Get merged cluster order")
    # To do: more qc?
    cluster_order_merge, clustid_to_clust_merge, pos_to_cluster_merge =  clusters_to_cluster_order(clusters_merged, seqs, remove_both = False)


    make_alignment(cluster_order_merge, numseqs, clustid_to_clust_merge)

    still_unassigned = get_unassigned_aas(seqs, pos_to_cluster_merge)
 
    print(still_unassigned)  
    new_clusters_still = []
    for gap in still_unassigned:
        new_clusters_still  = new_clusters_still + address_unassigned(gap, seqs, seqs_aas, pos_to_cluster_merge, cluster_order_merge, clustid_to_clust_merge, numseqs, I2, D2)
    print("this is a new cluster")
    for x in new_clusters_still:
        print(x)


    print("Need to get new clusters_filt")
    clusters_merged = list(clustid_to_clust_merge.values())   



    clusters_merged2 = merge_clusters(new_clusters_still, clusters_merged)
    print("Get order")

    cluster_order_merge2, clustid_to_clust_merge2, pos_to_cluster_merge2 =  clusters_to_cluster_order(clusters_merged2, seqs, remove_both = False)
    alignment = make_alignment(cluster_order_merge2, numseqs, clustid_to_clust_merge2)

    #newer_clusters = []
    #for x in new_clusters:
    #   for y in clusters_filt:
    #       # If there's an overlap
    #       if set(x).intersection(y):
    ##           # Get union of old and new cluster
    #           newer_clusters.append(list(set(x) | set(y)))
    #           continue
    #   newer_clusters.append(x) 
    #newest_clusters = []

    #for x in clusters_filt:
       #for y in newer_clusters:
          # # If there's an overlap
           #if set(x).intersection(y):
           #    # Get union of old and new cluster
           #    newest_clusters.append(list(set(x) | set(y)))
          #i    continue
      # newest_clusters.append(x) 




    #print("Remove poorly matching seqs after initial RBH seach")


    #cluster_order, starting_clustid, ending_clustid, clustid_to_clust_topo)


    return(alignment)


  



def format_sequences(fasta, padding =  10):
   
    # What are the arguments to this? what is test.fasta? 
    sequence_lols = parse_fasta(fasta, "test.fasta", False)

    df = pd.DataFrame.from_records(sequence_lols,  columns=['id', 'sequence', 'sequence_spaced'])
    seq_names = df['id'].tolist()
    seqs = df['sequence_spaced'].tolist() 
    
    newseqs = []
    for seq in seqs:
         newseq = "X X X X X X X X X X" + seq  # WHY does this help embedding to not have a space?
         newseq = newseq + " X X X X X X X X X X"
         newseqs.append(newseq)
    newseqs = newseqs

    return(newseqs, seq_names)

 
if __name__ == '__main__':
    # Embedding not good on short sequences without context Ex. HEIAI vs. HELAI, will select terminal I for middle I, instead of context match L
    # Potentially maximize local score? 
    # Maximize # of matches
    # Compute closes embedding of each amino acid in all target sequences
    # Compute cosine to next amino acid in seq. 

    model_name = 'prot_bert_bfd'
    #seqs = ['A A H K C Q T C G K A F N R S S T L N T H A R I H Y A G N P', 'Y K C K Q C G K A F A R S G G L Q K H K R T H', 'Y E C K E C G K S F S A H S S L V T H K R T H', 'Y K C E E C G K A F N R S S N L T K H K I I H', 'A A H K C Q T C G K A F N R S S T L N T H A R I H H A G N P', 'Y K C K Q C G K A F A R S G G L Q K H K R T H', 'Y E C K E C G K S F S A H S S L V T H Y R T H', 'Y K C E E C G K A F N R S S N L T K H K I I Y']
    #seqs = ['H E A L A I', 'H E A I A L']

    #seq_names = ['seq1','seq2', 'seq3', 'seq4']

    fasta = '/scratch/gpfs/cmcwhite/quantest2/QuanTest2/Test/zf-CCHH.vie'
    #fasta = '/scratch/gpfs/cmcwhite/quantest2/QuanTest2/Test/Ribosomal_L1.vie'
    #fasta = '/scratch/gpfs/cmcwhite/quantest2/QuanTest2/Test/ung.vie'
    seqs, seq_names = format_sequences(fasta, padding = 10)


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
    
    layers = [ -4, -3,-2, -1]
    #layers = [-5, -4, -3, -2, -1]
    #layers = [-4, -3, -2, -1]
 
    get_similarity_network(layers, model_name, seqs[20:40], seq_names[20:40], padding = 10)

    run_tests()
    #unittest.main(buffer = True)
