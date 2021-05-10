from transformer_notebooks.hf_utils import parse_fasta
fasta = '/scratch/gpfs/cmcwhite/quantest2/QuanTest2/Test/zf-CCHH.vie'
from sentence_transformers import util

from Bio import SeqIO
import pickle
import argparse

import numpy as np
import torch
from transformers import AutoTokenizer, AutoModel
 
 
 
 
def get_hidden_states(seqs, model, layers):

    encoded = tokenizer.batch_encode_plus(seqs, return_tensors="pt", padding=True)
    with torch.no_grad():
        output = model(**encoded)
 
    # Get all hidden states
    hidden_states = output.hidden_states
    #BramVanroy  https://github.com/huggingface/transformers/issues/1328#issuecomment-534956703
    #Concatenate final for hidden states into long vector
    pooled_output = torch.cat(tuple([hidden_states[i] for i in [-4, -3, -2, -1]]), dim=-1)

 
    return pooled_output
 
 

def get_matches(hidden_states, seqs,seq_names):
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
      hidden_states_a = hidden_states[a][0:seq_lens[a] + 1]

      for b in range(len(hidden_states)):
          if b in complete:
              continue
          hidden_states_b = hidden_states[b][0:seq_lens[b]]
          # Compare first token for sequence similarity
          overall_sim =  util.pytorch_cos_sim(hidden_states[a][0], hidden_states[b][0])
         
          # Don't compare first token [CLS] and last token [SEP]
          cosine_scores_a_b =  util.pytorch_cos_sim(hidden_states[a][1:-1], hidden_states[b][1:-1])

          # 1 because already compared first token [CLS] to [CLS] for overall similarity
          # - 1 becuase we don't need to compare the final token [SEP]
          for i in range(0, len(cosine_scores_a_b) ):
             bestscore = max(cosine_scores_a_b[i])
             bestmatch = np.argmax(cosine_scores_a_b[i])
             #top3idx = np.argpartition(cosine_scores_a_b[i], -3)[-3:]
             #top3scores= cosine_scores_a_b[i][top3idx]

             node_a = '{}-{}-{}'.format(seq_names[a], i, seqs[a][i])
             node_b = '{}-{}-{}'.format(seq_names[b], bestmatch, seqs[b][bestmatch])
             match_edges.append('{},{},{},match'.format(node_a,node_b,bestscore) 

    

    return match_edges
 
def get_seq_edges(seqs, seq_names):
    seq_edges = []
    for i in range(len(seqs)):
        for j in range(len(seqs[i]) - 1):
           pos1 = '{}-{}-{}'.format(seq_names[i], seqs[i][j], j)
           pos2 = '{}-{}-{}'.format(seq_names[i], seqs[i][j + 1], j + 1)
           seq_edges.append('{},{},1,seq'.format(pos1,pos2)
    return(seq_edges)

 
def get_similarity_network(layers, model_name, seqs, seq_names):
    # Use last four layers by default
    #layers = [-4, -3, -2, -1] if layers is None else layers
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name, output_hidden_states=True)

    hidden_states = get_hidden_states(encoded, model, layers)
    print(hidden_states.shape)

    match_edges = get_matches(seqs, tokenizer, model, layers)
    
    seq_edges = get_seq_edges(seqs, seq_names)

    return 1
 
 
if __name__ == '__main__':

    layers = [-4, -3, -2, -1]
    model_name = 'prot_bert_bfd'
    seqs = ['H K C Q T C G K A F N R S S T L N T H A R I H A G N P', 'Y K C K Q C G K A F A R S G G L Q K H K R T H']
    seqs = ['H E A T', 'H A T H E A T']

    seq_names = ['seq1','seq2']
    get_similarity_network(layers, model_name, seqs, seq_names)


#input_ids = torch.tensor(tokenizer.encode("Hello, my dog is cute")).unsqueeze(0)  # Batch size 1
#outputs = model(input_ids)
#
