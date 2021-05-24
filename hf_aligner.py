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


def reshape_flat(hstates_list):

    # Go from (numseqs, seqlen, emb) to (numseqs * seqlen, emb)
    hidden_states = np.reshape(hstates_list, (hstates_list.shape[0]*hstates_list.shape[1], hstates_list.shape[2]))
    return(hidden_states)



def get_similarity_network(layers, model_name, seqs, seq_names):
    # Use last four layers by default
    #layers = [-4, -3, -2, -1] if layers is None else layers
    print("load tokenizer")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    print("load model")
    model = AutoModel.from_pretrained(model_name, output_hidden_states=True)


    

    print("get hidden states for each seq")

    d = 4096
    #hidden_states_list =[]
    #seqlen = max([len(x.replace(" ", "")) for x in seqs])
    #print(seqlen)

    #seqlens = [len(x.replace(" ", "")) for x in seqs]
    print("start hidden_states")
    print(seqs)
    hstates_list = get_hidden_states(seqs, model, tokenizer, layers)
    print("end hidden states")
    #print(hstates_list)
    #print(hstates_list.shape)
    padded_seqlen = hstates_list.shape[1]

    # After encoding, remove spaces from sequences
    seqs = [x.replace(" ", "") for x in seqs]
    seqlens = [len(x) for x in seqs]
    numseqs = len(seqs)
    #for seq in seqs:
    #   hidden_states = get_hidden_states([seq], model, tokenizer, layers)
    #   hidden_states_list.append(hidden_states)


    # Build index from all amino acids 
    #d = hidden_states[0].shape[1]

    # Go from (numseqs, seqlen, emb) to (numseqs * seqlen, emb)
    hidden_states = np.array(reshape_flat(hstates_list))  

    index = build_index(hidden_states)
     
    D, I =  index.search(hidden_states, k = numseqs*padded_seqlen) 
    #np.set_printoptions(threshold=np.inf)
    print(I)
    print(len(I))
    print(D)
    print(numseqs)
    print(padded_seqlen) 
   
    D_tmp = []
    for i in range(len(I)):
        # Sort distances by index (default returns in descending order)
        sorted_D = [x for _,x in sorted(zip(I[i],D[i]))]
        sorted_D_split = [sorted_D[i:i + padded_seqlen] for i in range(0, len(sorted_D), padded_seqlen)]
        D_tmp.append(sorted_D_split)

    D =  [D_tmp[i:i + padded_seqlen] for i in range(0, len(D_tmp), padded_seqlen)]
    hitlist = []

    # Get reciprocal best hits 
    for query_seq in range(len(D)):
     
        for query_aa in range(len(D[query_seq])):
           if query_aa >= seqlens[query_seq]:
               continue
           for target_seq in range(len(D[query_seq][query_aa])):

               # Only take best score of real sequence
               # Can't mess with index start position, otherwise argmax...
               bestscore = max(D[query_seq][query_aa][target_seq][0:seqlens[target_seq]])
               bestmatch = np.argmax(D[query_seq][query_aa][target_seq][0:seqlens[target_seq]])
               recip_bestscore = max(D[target_seq][bestmatch][query_seq][0:seqlens[query_seq]])

               recip_bestmatch = np.argmax(D[target_seq][bestmatch][query_seq][0:seqlens[query_seq]])
               
               if recip_bestmatch == query_aa:
                  print("{}-{},{}-{},{},{},{}".format(query_seq, query_aa, target_seq, bestmatch, bestscore, seqs[query_seq][query_aa], seqs[target_seq][bestmatch]))
                  hitlist.append([query_seq, query_aa, target_seq, bestmatch, bestscore, seqs[query_seq][query_aa], seqs[target_seq][bestmatch]])


    #To do: Make matchstate object
    # Remove streak conflict matches
    filtered_hitlist = []
    for i in range(len(seqs)):
       query_prot = [x for x in hitlist if x[0] == i]
       for j in range(len(seqs)):
          target_prot = [x for x in query_prot if x[2] == j]

          print("remove lookbehinds")
          prevmatch = 0
          seq_start = -1
          streak = 0

          no_lookbehinds = []
          for match_state in target_prot:
               print(match_state)
               if match_state[3] <= seq_start:
                     print("lookbehind prevented")
                     streak = 0 
                     continue
               no_lookbehinds.append(match_state)

               if match_state[3] - prevmatch == 1:
                  streak = streak + 1
                  if streak > 2:  
                     seq_start = match_state[3]
               else:
                  streak = 0
               prevmatch = match_state[3]

          print("remove lookaheads")
          prevmatch = seqlens[j]
          seq_end = seqlens[j]
          streak = 0

          filtered_target_prot = []
          for match_state in no_lookbehinds[::-1]:
               print(match_state, streak, prevmatch)
               if match_state[3] >= seq_end:
                    print("lookahead prevented")
                    streak = 0
                    continue
               filtered_target_prot.append(match_state)
               if prevmatch - match_state[3] == 1:
                  streak = streak + 1
                  if streak > 2:  
                     seq_end = match_state[3]
               else:
                  streak = 0
               prevmatch = match_state[3]
 
          filtered_hitlist = filtered_hitlist + filtered_target_prot

    print(pd.DataFrame(filtered_hitlist))
    with open("testnet.csv", "w") as outfile:
      outfile.write("aa1,aa2,score\n")
      # If do reverse first, don't have to do second resort
      for x in filtered_hitlist[::1]:
 
         outstring = "s{}-{}-{},s{}-{}-{},{}\n".format(x[0], x[1], x[5], x[2], x[3], x[6], x[4])
         outfile.write(outstring)


    edges = []
    for x in filtered_hitlist[::1]:
        node1_id = "s{}-{}-{}".format(x[0], x[1], x[5]) 
        node2_id = "s{}-{}-{}".format(x[2], x[3], x[6])
        edges.append((node1_id, node2_id))
     # >>> edges = [(1, 5), (4, 2), (4, 3), (5, 4), (6, 3), (7, 6), (8, 9)]
    G = igraph.Graph.TupleList(edges=edges, directed=False)
    # Remove multiedges and loops
    G = G.simplify()
    

    #fastgreedy = G.community_fastgreedy().as_clustering()
    #print("fastgreedy")
    #print(fastgreedy)


    walktrap = G.community_walktrap(steps = 1).as_clustering()
    print("walktrap1")
    print(walktrap)

    clusters = walktrap.membership
    vertices = G.vs()["name"]
    cluster_memberships = list(zip(clusters, vertices))
 
    with open("clustertest.csv", "w") as outfile:
       for c in cluster_memberships:
            outstring = "{},{}\n".format(c[0], c[1])
            print(outstring)
            outfile.write(outstring)
            
        

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


    # Plan...hierarchical k means
    # Start low, then break apart

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

    return 1
 
 
if __name__ == '__main__':
    # Embedding not good on short sequences without context Ex. HEIAI vs. HELAI, will select terminal I for middle I, instead of context match L
    # Potentially maximize local score? 
    # Maximize # of matches
    # Compute closes embedding of each amino acid in all target sequences
    # Compute cosine to next amino acid in seq. 

    layers = [-4, -3, -2, -1]
    model_name = 'prot_bert_bfd'
    #seqs = ['A A H K C Q T C G K A F N R S S T L N T H A R I H Y A G N P', 'Y K C K Q C G K A F A R S G G L Q K H K R T H', 'Y E C K E C G K S F S A H S S L V T H K R T H', 'Y K C E E C G K A F N R S S N L T K H K I I H', 'A A H K C Q T C G K A F N R S S T L N T H A R I H H A G N P', 'Y K C K Q C G K A F A R S G G L Q K H K R T H', 'Y E C K E C G K S F S A H S S L V T H Y R T H', 'Y K C E E C G K A F N R S S N L T K H K I I Y']
    #seqs = ['H E A L A I', 'H E A I A L']

    #seq_names = ['seq1','seq2', 'seq3', 'seq4']

    fasta = '/scratch/gpfs/cmcwhite/quantest2/QuanTest2/Test/zf-CCHH.vie'
    #fasta = '/scratch/gpfs/cmcwhite/quantest2/QuanTest2/Test/Ribosomal_L1.vie'
    sequence_lols = parse_fasta(fasta, "test.fasta", False)

    df = pd.DataFrame.from_records(sequence_lols,  columns=['id', 'sequence', 'sequence_spaced'])
    seq_names = df['id'].tolist()
    seqs = df['sequence_spaced'].tolist() 


    get_similarity_network(layers, model_name, seqs[0:20], seq_names[0:20])


#input_ids = torch.tensor(tokenizer.encode("Hello, my dog is cute")).unsqueeze(0)  # Batch size 1
#outputs = model(input_ids)
    #index_to_protpos = {}
    # convert index to sequence position/aa
    
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
 
