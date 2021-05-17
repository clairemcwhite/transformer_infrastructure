from transformer_infrastructure.hf_utils import parse_fasta, get_hidden_states, compare_hidden_states, kmeans_hidden_states_aas
import faiss
fasta = '/scratch/gpfs/cmcwhite/quantest2/QuanTest2/Test/zf-CCHH.vie'
from sentence_transformers import util
from iteration_utilities import  duplicates, unique_everseen

from Bio import SeqIO
import pickle
import argparse

import numpy as np
import torch
from transformers import AutoTokenizer, AutoModel

from pandas.core.common import flatten 
import pandas as pd 
import matplotlib.pyplot as plt
import seaborn as sns; sns.set() 
 

def get_matches_2(hidden_states, seqs,seq_names):
    """Get a word vector by first tokenizing the input sentence, getting all token idxs
       that make up the word of interest, and then `get_hidden_states
     BramVanroy`."""

    

    match_edges = []

    
    seqs = [x.replace(" ", "") for x in seqs]
    seq_lens = [len(x) for x in seqs]
   
    
    """ Do for all and cluster"""
    """ Get clusters. Guideposts?"""
    """ Fill in? """
    # compare all sequences 'a' to 'b'
    complete = []
    for a in range(len(hidden_states)): 
      complete.append(a)
      # Remove embeddings for [PAD] past the length of the original sentence
      hidden_states_a = hidden_states[a][1:seq_lens[a] + 1]

      for b in range(len(hidden_states)):
          if b in complete:
              continue
          hidden_states_b = hidden_states[b][1:seq_lens[b] + 1]
          # Compare first token for sequence similarity, not the place for this
          #overall_sim =  util.pytorch_cos_sim(hidden_states[a][0], hidden_states[b][0])
         
          # Don't compare first token [CLS] and last token [SEP]
          print(len(hidden_states_a))
          print(len(hidden_states_b))
          cosine_scores_a_b =  util.pytorch_cos_sim(hidden_states_a, hidden_states_b)
          print(cosine_scores_a_b)

          # 1 because already compared first token [CLS] to [CLS] for overall similarity
          # - 1 becuase we don't need to compare the final token [SEP]
          for i in range(0, len(cosine_scores_a_b) ):
             bestscore = max(cosine_scores_a_b[i])
             bestmatch = np.argmax(cosine_scores_a_b[i])
             #top3idx = np.argpartition(cosine_scores_a_b[i], -3)[-3:]
             #top3scores= cosine_scores_a_b[i][top3idx]
             print(i)
             print(seqs)          
             print(seqs[a])
             print(seq_names[a])
             node_a = '{}-{}-{}'.format(seq_names[a], i, seqs[a][i])
             node_b = '{}-{}-{}'.format(seq_names[b], bestmatch, seqs[b][bestmatch])
             match_edge = '{},{},{},match'.format(node_a,node_b,bestscore)
             print(match_edge)

             match_edges.append(match_edge)
    
    return match_edges
 
def get_seq_edges(seqs, seq_names):
    seq_edges = []
    seqs = [x.replace(" ", "") for x in seqs]

    for i in range(len(seqs)):
        for j in range(len(seqs[i]) - 1):
           pos1 = '{}-{}-{}'.format(seq_names[i], seqs[i][j], j)
           pos2 = '{}-{}-{}'.format(seq_names[i], seqs[i][j + 1], j + 1)
           seq_edges.append('{},{},1,seq'.format(pos1,pos2))
    return(seq_edges)

 
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

    seqlens = [len(x.replace(" ", "")) for x in seqs]
    print("start hidden_states")
    hidden_states = get_hidden_states(seqs, model, tokenizer, layers)
    print("end hidden states")
    padded_seqlen = hidden_states.shape[1]

    num_seqs = len(seqs)
    #for seq in seqs:
    #   hidden_states = get_hidden_states([seq], model, tokenizer, layers)
    #   hidden_states_list.append(hidden_states)


    # Build index from all amino acids 
    #d = hidden_states[0].shape[1]

    # Go from (numseqs, seqlen, emb) to (numseqs * seqlen, emb)
    hidden_states = np.reshape(hidden_states, (hidden_states.shape[0]*hidden_states.shape[1], hidden_states.shape[2]))
   
    k = padded_seqlen

    #https://stackoverflow.com/questions/9835762/how-do-i-find-the-duplicates-in-a-list-and-create-another-list-with-them
    completed = False
    while completed == False:
        print("start kmeans")
        D, I = kmeans_hidden_states_aas(hidden_states, k)
        print("end kmeans")
        I_list = list(flatten(I))
        I_list_split =  [I_list[i:i + padded_seqlen] for i in range(0, len(I_list), padded_seqlen)]
        repeat_found = True
        dups = []
        e = 0
        for i in range(len(I_list_split)):
               prot_trunc = I_list_split[i][0:seqlens[i]]
               dup_set = list(unique_everseen(duplicates(prot_trunc)))
               dups = dups + dup_set
               e = e + 1
               print(dups)
               if e == 5:
                  break
        completed = True

    D_list = list(flatten(D))
    D_list_split =  [D_list[i:i + padded_seqlen] for i in range(0, len(D_list), padded_seqlen)]

    with open("/home/cmcwhite/testnet.csv", "w") as outfile:
        for prot in I_list_split:
           for i in range(len(prot) - 1):
              outstring ="{},{}\n".format(prot[i], prot[i + 1])
              outfile.write(outstring)


    #for i in range(len(I_list_split)):
    #      I_trunc = I_list_split[i][0:seqlens[i]]
    #      print(" ".join([str(item).zfill(2) for item in I_trunc]))

    #fl = np.rot90(np.array(hidden_states))
    print(np.array(hidden_states).shape)
    #print(fl.shape)

    pca = faiss.PCAMatrix(4096, 100)
    pca.train(np.array(hidden_states))
    tr = pca.apply_py(np.array(hidden_states))
    print(tr)
    print(tr.shape)




    plt.scatter(tr[:,0], tr[:, 1],
            c=I_list, edgecolor='none', alpha=0.5,
            cmap=plt.cm.get_cmap('jet', k))
    plt.xlabel('component 1')
    plt.ylabel('component 2')
    plt.colorbar()

    plt.plot(tr[:, 0][0:25], tr[:, 1][0:25])
   
    plt.savefig("/home/cmcwhite/pca12.png")


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
    sequence_lols = parse_fasta(fasta, "test.fasta", False)

    df = pd.DataFrame.from_records(sequence_lols,  columns=['id', 'sequence', 'sequence_spaced'])
    seq_names = df['id'].tolist()
    seqs = df['sequence_spaced'].tolist() 


    get_similarity_network(layers, model_name, seqs[0:800], seq_names[0:800])


#input_ids = torch.tensor(tokenizer.encode("Hello, my dog is cute")).unsqueeze(0)  # Batch size 1
#outputs = model(input_ids)
#
