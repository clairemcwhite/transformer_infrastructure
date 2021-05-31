from transformer_infrastructure.hf_utils import parse_fasta, get_hidden_states, build_index
import faiss
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

    for line in alignment:
       print("".join(line))
 
def remove_feedback_edges(cluster_orders, clusters_filt):

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
 
               #print(cluster_orders[i][j], cluster_orders[i][j + 1])
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



def remove_doubles(clusters_list, numseqs):

    clusters_filt = []
    for i in range(len(clusters_list)): 
         seqcounts = [0] * numseqs # Will each one replicated like with [[]] * n?
         if len(clusters_list[i]) < 3:
           continue
         for pos in clusters_list[i]:
            seqnum = get_seqnum(pos)
            #print(seq, seqnum)
            seqcounts[seqnum] = seqcounts[seqnum] + 1
         remove_list = [i for i in range(len(seqcounts)) if seqcounts[i] > 1]
         clust = []
         for pos in clusters_list[i]:
            seqnum =  get_seqnum(pos)
            if seqnum in remove_list:
               #print("{} removed from cluster {}".format(seq, i))
               continue
            else:
               clust.append(pos)
         clusters_filt.append(clust)
    return(clusters_filt)


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
      print(i)
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
   aa_to_index = {value: key for key, value in index_to_aa.items()}

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


def get_similarity_network(layers, model_name, seqs, seq_names):
    # Use last four layers by default
    #layers = [-4, -3, -2, -1] if layers is None else layers
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
    print(hstates_list.shape)
    # Remove first and last X padding
    hstates_list = hstates_list[:,5:-5,:]
    padded_seqlen = hstates_list.shape[1]

    # After encoding, remove spaces from sequences
    seqs = [x.replace(" ", "")[5:-5] for x in seqs]
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
    D1, I1 =  index.search(hidden_states, k = numseqs*2) 

    print("Split results into proteins") 
    # Still annoyingly slow
    D2, I2 = split_distances_to_sequence(D1, I1, index_to_aa, numseqs, padded_seqlen) 

    print("get best hitlist")
    hitlist_top = get_besthits(D2, I2, index_to_aa, padded_seqlen)
 
    G_hitlist = igraph.Graph.TupleList(edges=hitlist_top, directed=True) 


    print("Get reciprocal best hits")
    rbh_bool = G_hitlist.is_mutual()
    
    hitlist = []
    for i in range(len(G_hitlist.es())):
        if rbh_bool[i] == True:
           source_vertex = G_hitlist.vs[G_hitlist.es()[i].source]["name"]
           target_vertex = G_hitlist.vs[G_hitlist.es()[i].target]["name"]
           hitlist.append([source_vertex, target_vertex])

    print("got reciprocal besthits")
  
    remove_streaks = False  
    if remove_streaks == True:
        print("Remove streak conflict matches")
        hitlist = remove_streakbreakers(hitlist, seqs, seqlens, streakmin = 3)
    
   
    ######################################### Do walktrap clustering
    # Why isn't this directed?

    with open("testnet.csv", "w") as outfile:
          outfile.write("aa1,aa2,score\n")
          # If do reverse first, don't have to do second resort
          for x in hitlist:
             outstring = "{},{}\n".format(x[0], x[1])        
             #outstring = "s{}-{}-{},s{}-{}-{},{}\n".format(x[0], x[1], x[5], x[2], x[3], x[6], x[4])
             outfile.write(outstring)
 
    G = igraph.Graph.TupleList(edges=hitlist, directed=True)
    # Remove multiedges and self loops
    print("Remove multiedges and self loops")
    G = G.simplify()
    

    #fastgreedy = G.community_fastgreedy().as_clustering()
    #print("fastgreedy")
    #print(fastgreedy)

    print("start walktrap")
    walktrap = G.community_walktrap(steps = 1).as_clustering()
    print("walktrap1")

    cluster_ids = walktrap.membership
    vertices = G.vs()["name"]
    clusters = list(zip(cluster_ids, vertices))
 
    with open("clustertest.csv", "w") as outfile:
       for c in clusters:
            outstring = "{},{}\n".format(c[0], c[1])
            outfile.write(outstring)

    
    clusters_list = []
    for i in range(len(walktrap)):
         clusters_list.append([vertices[x] for x in walktrap[i]])


    ##############################
    # No doubles check
    # If a cluster has two entries from same seq, remove thos
    # If a cluster has fewer then 3 members, remove them

    clusters_filt = remove_doubles(clusters_list, numseqs)

    clusters_filt = remove_low_match_prots(numseqs, seqlens, clusters_filt, threshold_min = 0.5) 
    print("Remove poorly matching seqs after initial RBH seach")

    ######################################3
    # Remove feedback loops in paths through clusters
    # Dictionary posname:clusternum 
    # For getting consensus cluster order
    pos_to_clustid, clustid_to_clust= get_cluster_dict(clusters_filt, seqs)
    cluster_orders = get_cluster_orders(pos_to_clustid, seqs)

    print("Find directed acyclic graph")   
    clusters_filt_dag = remove_feedback_edges(cluster_orders, clusters_filt)
 
    print("Removed poorly matching seqs after DAG-ification")
    clusters_filt_dag = remove_low_match_prots(numseqs, seqlens, clusters_filt_dag, threshold_min = 0.5) 

    print("Get cluster order after dag")

    pos_to_cluster_dag, clustid_to_clust_dag = get_cluster_dict(clusters_filt_dag, seqs)
    cluster_orders_dag = get_cluster_orders(pos_to_cluster_dag, seqs)
    for x in cluster_orders_dag:
           print(x)

    print("Get a single cluster order with minim")
    G_order = graph_from_cluster_orders(cluster_orders_dag)
    G_order = G_order.simplify()

    print(G_order)
    #for x in G_order.vs:
    #   print("index", x.index, "name", x['name']) 

    # Note: this is in vertex indices. Need to convert to name to get clustid
    topo_sort_indices = G_order.topological_sorting()
    cluster_order = []
    for i in topo_sort_indices:
       cluster_order.append(G_order.vs[i]['name'])
    print(cluster_order)

 
    print("For each sequence check that the cluster order doesn't conflict with aa order")

    print(pos_to_cluster_dag)    
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
           print(key, clust, order_index)
           if order_index < prev_cluster:
                clusters_w_order_conflict.append(clust)
                print("order_violation", order_index, clust)
           prev_cluster  = order_index
    print(cluster_order)
    print(clusters_w_order_conflict)
    cluster_order = [x for x in cluster_order if x not in clusters_w_order_conflict]

    print(cluster_order)
    clustid_to_clust_dag = {key:val for key, val in clustid_to_clust_dag.items() if key  in cluster_order}

    
    make_alignment(cluster_order, numseqs, clustid_to_clust_dag)
   


 
    
    # Observations:
       #Too much first character dependencyi
       # Too much end character dependency
          # Add X on beginning and end seems to fix at least for start
      # Order is messed up at the end
     # Uniprot doesn't get the last H either so nbd
    #for trouble in unassigned:

    #print(trouble)

    unassigned = []
    for i in range(len(seqs)):
        if i == 3:
           continue
        prevclust = []
        nextclust = []
        unsorted = []
        last_unsorted = 0
        for j in range(len(seqs[i])):
           if j <= last_unsorted:
               continue

           key = "s{}-{}-{}".format(i, j, seqs[i][j])
           try:
              # Read to first cluster hit
              clust = pos_to_cluster_dag[key]
              prevclust = clust
              #prevclusts.append(prevclust)
           # If it's not in a clust, it's unsorted
           except Exception as E:
              unsorted = []
              unsorted.append(key)
              for k in range(j + 1, len(seqs[i])):
                  key = "s{}-{}-{}".format(i, k, seqs[i][k])
                  try:
                     nextclust = pos_to_cluster_dag[key]
                     break
                  except Exception as E:
                     unsorted.append(key)
                     last_unsorted = k
              unassigned.append([prevclust, unsorted, nextclust])
  










#
#              if j == 0:
#                  nextclusts = cluster_orders_dag[i:]
#              
#              else:
#                  prevclust_index =  cluster_orders_dag[i].index(prevclust) 
#                  if prevclust_index == len(cluster_orders_dag[i]) - 1:
#                  #    nextclusts = "[]"
#                  else:
#                      nextclusts = cluster_orders_dag[i][prevclust_index + 1 : ]
#              print(prevclusts, key, nextclusts)
#
#

#


 
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

    #fasta = '/scratch/gpfs/cmcwhite/quantest2/QuanTest2/Test/zf-CCHH.vie'
    fasta = '/scratch/gpfs/cmcwhite/quantest2/QuanTest2/Test/Ribosomal_L1.vie'
    #fasta = '/scratch/gpfs/cmcwhite/quantest2/QuanTest2/Test/ung.vie'

    sequence_lols = parse_fasta(fasta, "test.fasta", False)

    df = pd.DataFrame.from_records(sequence_lols,  columns=['id', 'sequence', 'sequence_spaced'])
    seq_names = df['id'].tolist()
    seqs = df['sequence_spaced'].tolist() 
    
    testseqs = []
    
    for x in range(len(seqs)):
        if x in [3, 39, 69, 94]:
           print(seqs[x])
        seqs[x] = seqs[x]
    print("x")
   
    local = True
   
    if local == True:
        newseqs = []
        for seq in seqs:
             newseq = "X X X X X" + seq  # WHY does this help embedding to not have a space?
             newseq = newseq + " X X X X X"
             newseqs.append(newseq)
        seqs = newseqs

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
 
    get_similarity_network(layers, model_name, seqs[0:50], seq_names[0:50])

    # 
    #http://pfam.xfam.org/protein/A0A1I7UMU0

    for x in range(len(seqs[0:10])):
       
           print(seqs[x])

    #print(seqs[3])
    
    #for i in range(0, numseqs):
    #   # All sequences are padded to the length of the longest sequence
    #   for pos in range(0, padded_seqlen):
    #        # Only retrieve info from positions that aren't padding
    #        if pos > seqlens[i] -1 :
    #           continue
    #        print(pos) 
    #        index_num = padded_seqlen * i + pos
    #        aa = seqs[i][pos]
    #        identified = [i, aa, pos]
    #        index_to_protpos[index_num] = identified
    #print(index_to_protpos)

#def get_matches_2(hidden_states, seqs,seq_names):
#    """Get a word vector by first tokenizing the input sentence, getting all token idxs
#       that make up the word of interest, and then `get_hidden_states
#     BramVanroy`."""
#
#    
#
#    match_edges = []
#
#    
#    seqs = [x.replace(" ", "") for x in seqs]
#    seq_lens = [len(x) for x in seqs]
#   
#    
#    """ Do for all and cluster"""
#    """ Get clusters. Guideposts?"""
#    """ Fill in? """
#    # compare all sequences 'a' to 'b'
#    complete = []
#    for a in range(len(hidden_states)): 
#      complete.append(a)
#      # Remove embeddings for [PAD] past the length of the original sentence
#      hidden_states_a = hidden_states[a][1:seq_lens[a] + 1]
#
#      for b in range(len(hidden_states)):
#          if b in complete:
#              continue
#          hidden_states_b = hidden_states[b][1:seq_lens[b] + 1]
#          # Compare first token for sequence similarity, not the place for this
#          #overall_sim =  util.pytorch_cos_sim(hidden_states[a][0], hidden_states[b][0])
#         
#          # Don't compare first token [CLS] and last token [SEP]
#          print(len(hidden_states_a))
#          print(len(hidden_states_b))
#          cosine_scores_a_b =  util.pytorch_cos_sim(hidden_states_a, hidden_states_b)
#          print(cosine_scores_a_b)
#
#          # 1 because already compared first token [CLS] to [CLS] for overall similarity
#          # - 1 becuase we don't need to compare the final token [SEP]
#          for i in range(0, len(cosine_scores_a_b) ):
#             bestscore = max(cosine_scores_a_b[i])
#             bestmatch = np.argmax(cosine_scores_a_b[i])
#             #top3idx = np.argpartition(cosine_scores_a_b[i], -3)[-3:]
#             #top3scores= cosine_scores_a_b[i][top3idx]
#             print(i)
#             print(seqs)          
#             print(seqs[a])
#             print(seq_names[a])
#             node_a = '{}-{}-{}'.format(seq_names[a], i, seqs[a][i])
#             node_b = '{}-{}-{}'.format(seq_names[b], bestmatch, seqs[b][bestmatch])
#             match_edge = '{},{},{},match'.format(node_a,node_b,bestscore)
#             print(match_edge)
#
#             match_edges.append(match_edge)
#    
#    return match_edges
# 
#def get_seq_edges(seqs, seq_names):
#    seq_edges = []
#    seqs = [x.replace(" ", "") for x in seqs]
#
#    for i in range(len(seqs)):
#        for j in range(len(seqs[i]) - 1):
#           pos1 = '{}-{}-{}'.format(seq_names[i], seqs[i][j], j)
#           pos2 = '{}-{}-{}'.format(seq_names[i], seqs[i][j + 1], j + 1)
#           seq_edges.append('{},{},1,seq'.format(pos1,pos2))
#    return(seq_edges)
#
    # Spliti into sequences


    # numsites * [ scores to everything ] 
    # [[[seq1-aa1 to seq1], [seq1-aa1 to seq2]], [[seq1-aa2 to seq1], [seq1-aa2 to seq2]], [[seq2-aa1 to seq1], [seq2-aa1 to seq2]], [[seq2-aa2 to seq1], [seq2-aa2 to seq2]]]
    #numseqs * padded_seqlen * numseqs * padded seqlen


    #I_list_split =  [I[i:i + padded_seqlen] for i in range(0, len(I), padded_seqlen)]
    #print(len(I))
    #print(len(I_list_split))
    #print(a) 
 #import matplotlib.pyplot as plt
#import seaborn as sns; sns.set() 

    #walktrap = G.community_walktrap(steps = 2).as_clustering()
    #print("walktrap2")
    #print(walktrap)


    #walktrap = G.community_walktrap(steps = 3).as_clustering()
    #print("walktrap3")
    #print(walktrap)

    #walktrap = G.community_walktrap(steps = 4).as_clustering()
    #print("walktrap3")
    #print(walktrap)

    
    #print("infomap")
    #infomap =  G.community_infomap()
    #print(infomap)

    #print("edge betweenness")
    #edge_betweenness =  G.community_edge_betweenness(directed = False).as_clustering()
    #print(edge_betweenness)
 
#    print("Find all fully connected cliques of size > 2") 
#    # Watch for exponential...
#    # This is not sustainable
#    # Great for like 3 sequences
#    all_cliques = list(nx.enumerate_all_cliques(G))
#    print("cliques found")
#    fc_cliques = []
#    fc_cliques_nodes = []
#    # Start from largest potential cluster, count down to size 3
#    for i in range(numseqs, 2, -1):
#        fc = [s for s in all_cliques if len(s) == i]
#        for clique in fc:
#           if not any(item in clique for item in fc_cliques_nodes):
#               fc_cliques.append(clique) 
#        fc_cliques_nodes = list(flatten(fc_cliques))
#
#    fc_cliques_nooverlaps = []
#    node_duplicates = list(duplicates(fc_cliques_nodes))
#    print(node_duplicates)
#    for clique in fc_cliques:
#       if not any(item in clique for item in node_duplicates):
#             fc_cliques_nooverlaps.append(clique)
#
#    for x in fc_cliques:
#      print(x)
#
#    clique_dict = {}
#    for i in range(len(fc_cliques_nooverlaps)):
#         for pos in fc_cliques_nooverlaps[i]:
#             clique_dict[pos] = i
#
#    print(clique_dict)
    

    


        #a = [a for a in fc if no any(item in List1 for item in List2)

    #for x in fc_cliques:
    #  for y in

    #fc = [s for s in nx.enumerate_all_cliques(G) if len(s) > 2] 
    #print(fc)

     #>>> [tuple(c) for c in nx.connected_components(graph)]
     #[(1, 2, 3, 4, 5, 6, 7), (8, 9)]

     #Check that subgraphs are sequential. Otherwise what? Remove non-sequential components?
     # If an aa is in a fully connected clique, it gets masked
     # Then can only look for matches between strings of matches
 

        #for seq in sorted_D_split:
        #        bestscore = max(seq)
        #        bestmatch = np.argmax(seq)
        #        print("".format(i, bestmatch, bestscore))
 


    #for i in range(0, len(cosine_scores_a_b) ):
    #     bestscore = max(cosine_scores_a_b[i])
    #     bestmatch = np.argmax(cosine_scores_a_b[i])
    
 
    #extracted_seqs = [D[index_list] for index_list in indices]

    # Split indexes 
    #I_list = list(flatten(I))
    #I_list_split =  [I_list[i:i + padded_seqlen] for i in range(0, len(I_list), padded_seqlen)]

    #D_list = list(flatten(D))
    #D_list_split =  [D_list[i:i + padded_seqlen] for i in range(0, len(D_list), padded_seqlen)]

    #for i in range(len(I_list_split)):
    #    prot_trunc = I_list_split[i][0:seqlens[i]]

 
 
    #index = build_index(hidden_states)
 
    # Just do full loop for troubleshooting, the work on indexes
    #for i in range(len(seqs)):
    #   for i_pos in range(len(hstates_list[i])):
    #       for j in range(1, len(seqs) - 1):
    #           print(hstates_list[i][i_pos].shape)
    #           print(hstates_list[j].shape) 
    #           pos_vs_prot = torch.cat([hstates_list[i][i_pos]] + hstates_list[j])
    #           print(pos_vs_prot.shape)
    #           index = build_index(pos_vs_prot)
    #           D, I =  index.search(hstates_list[i][i_pos], k = 2)
    #           print(D, I)
    #           print(seqs[i][i_pos]) 

    #           break
    #k = 2 * padded_seqlen

    #https://stackoverflow.com/questions/9835762/how-do-i-find-the-duplicates-in-a-list-and-create-another-list-with-them
        #I_list = list(flatten(I))
        #I_list_split =  [I_list[i:i + padded_seqlen] for i in range(0, len(I_list), padded_seqlen)]
        #repeat_found = True
        #dups = []
        #for i in range(len(I_list_split)):
        #       prot_trunc = I_list_split[i][0:seqlens[i]]
        #       dup_set = list(unique_everseen(duplicates(prot_trunc)))
    #D_list = list(flatten(D))
    #D_list_split =  [D_list[i:i + padded_seqlen] for i in range(0, len(D_list), padded_seqlen)]

    #with open("/home/cmcwhite/testnet.csv", "w") as outfile:
    #    for prot in I_list_split:
    #       for i in range(len(prot) - 1):
    #          outstring ="{},{}\n".format(prot[i], prot[i + 1])
    #          outfile.write(outstring)


    #for i in range(len(I_list_split)):
    #      I_trunc = I_list_split[i][0:seqlens[i]]
    #      print(" ".join([str(item).zfill(2) for item in I_trunc]))

    #fl = np.rot90(np.array(hidden_states))
    #print(np.array(hidden_states).shape)
    #print(fl.shape)

   # pca = faiss.PCAMatrix(4096, 100)
    #pca.train(np.array(hidden_states))
    #tr = pca.apply_py(np.array(hidden_states))
    #print(tr)
    #print(tr.shape)
#



    #plt.scatter(tr[:,0], tr[:, 1],
    #        c=I_list, edgecolor='none', alpha=0.5,
    #        cmap=plt.cm.get_cmap('jet', k))
    #plt.xlabel('component 1')
    #plt.ylabel('component 2')
    #plt.colorbar()

    #plt.plot(tr[:, 0][0:25], tr[:, 1][0:25])
   
    #plt.savefig("/home/cmcwhite/pca12.png")



    #for i in range(len(D_list)):
    #      D_trunc = D_list[i][0:seqlens[i]]
    #      print(" ".join([str(round(item, 3)) for item in D_trunc]))


    

    # Faiss suggested approach is to return more than needed an filter for criteria. 
    # Need to build easy transfer between index and sequence index
    
    # Can we use pure kmeans?
    # Get cluster with maximum representatives in each sequence, with minimal repitions
    # N represented seqs/len(cluster)
    # Starting k = length of longest sequence
    # What we're going to do is use k of longest seqlen. 
    # The for each prot, get path of clusters they pass through
    # Profit?

 
    #index = faiss.index_factory(d, "Flat", faiss.METRIC_INNER_PRODUCT)
    

    #for prot_hidden_states in hidden_states_list:
    #    print(prot_hidden_states[0])
    #    faiss.normalize_L2(np.array(prot_hidden_states[0]))
    #    index.add(np.array(prot_hidden_states[0]))



    #Will be parallelizable.
    #for i in range(len(hidden_states_list)):
    #   query_hs = np.array(hidden_states_list[i])
    #   index = faiss.index_factory(d, "Flat", faiss.METRIC_INNER_PRODUCT)
    #   index.add(query_hs)
    #   for j in range(len(hidden_states_list)):
    #      target_hs = np.array(hidden_states_list[j])
    #      index = faiss.index_factory(d, "Flat", faiss.METRIC_INNER_PRODUCT) 
    #      both = faiss.normalize_L2(np.concatenate(query_hs, target_hs))
    #      index.add(both)

     #     for m in range(len(hidden_states_list[i])):
     #         query = hidden_states_list[i][m]
     #         target = hidden_states_list[j] 
     #         D, I = compare_hidden_states(query, target, 2)
     #         print(I)
                  

    # Build strings of matching substrings. 
 

    #match_edges = get_matches(hidden_states, seqs, seq_names)
    
    #seq_edges = get_seq_edges(seqs, seq_names)
    #all_edges = seq_edges + match_edges 
    #for x in all_edges:
    #  print(x)

#
#    #alignment = np.empty((numseqs, padded_seqlen))
#    alignment = [""] * numseqs
#    header =  range(len(clusters_filt_dag))
#    #alignment.append(header) 
#    for i in  [ 0] : #range(len(seqs)):
#       for j in range(len(seqs[i])):
#          key = "s{}-{}-{}".format(i, j, seqs[i][j])
#          alignment[i] = alignment[i] + seqs[i][j]
#          try:
#             clust = pos_to_cluster_dag[key]
#             cluster = clusters_filt_dag[clust]
#          except Exception as E:
#             #for k in range(len(seqs)):
#                  cluster = ""
#                  #continue
#                #alignment[k] = alignment[k] + "-"            
#          for k in range(len(seqs)):
#            if i ==k:
#               continue
#            try:
#                cluster_member = [x for x in cluster if x.split("-")[0].replace("s", "") == str(k)]
#                #print(cluster_member)
#                alignment[k]= alignment[k] + cluster_member[0].split("-")[2]
#            except Exception as E:
#               alignment[k] = alignment[k] + "-"
#            
#
#    for i in range(len(alignment)):
#      print(i, alignment[i])


    # Go through each position in the sequence. 
    # See if it's already in a guide post
    # If not, find previous and next guidepost within seqeucne
    # Get those guideposts from query sequences. 
    #     


    # Pull region between two clusters 
    #1, 5
    # Get index of first one,
    #[x for x in seq if 
    # Get index of second one  
  
    #return 0       
 
