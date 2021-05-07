from transformer_notebooks.hf_utils import parse_fasta
fasta = '/scratch/gpfs/cmcwhite/quantest2/QuanTest2/Test/zf-CCHH.vie'
from sentence_transformers import util

from Bio import SeqIO
import pickle
import argparse

import numpy as np
import torch
from transformers import AutoTokenizer, AutoModel
 
 
def get_word_idx(sent: str, word: str):
    #BramVanroy
    return sent.split(" ").index(word)
 
 
def get_hidden_states(encoded, model, layers):
    with torch.no_grad():
        output = model(**encoded)
 
    # Get all hidden states
    hidden_states = output.hidden_states

    #BramVanroy  https://github.com/huggingface/transformers/issues/1328#issuecomment-534956703
    #Concatenate final for hidden states into long vector
    pooled_output = torch.cat(tuple([hidden_states[i] for i in [-4, -3, -2, -1]]), dim=-1)
    print("postpool")
    print(pooled_output) 
    print(pooled_output.shape)

 
    return pooled_output
 
 

def get_word_vectors(seqs, tokenizer, model, layers):
    """Get a word vector by first tokenizing the input sentence, getting all token idxs
       that make up the word of interest, and then `get_hidden_states
     BramVanroy`."""

    
    encoded = tokenizer.batch_encode_plus(seqs, return_tensors="pt", padding=True)
    # get all token idxs that belong to the word of interest
    #token_ids_word = np.where(np.array(encoded.word_ids()) == idx)
    
    #print(encoded.word_ids())
    #print(encoded.word_ids()[0])

    #for  x in encoded.word_ids():
    hidden_states = get_hidden_states(encoded, model, layers)
    print(hidden_states.shape)
    #cos_scores = util.pytorch_cos_sim(hidden_states, hidden_states)        

    pairs = []

    seqs = [x.replace(" ", "") for x in seqs]
    """ Do for all and cluster"""
    """ Get clusters. Guideposts?"""
    """ Fill in? """
    # compare all sequences 'a' to 'b'
    for a in range(len(hidden_states)):
      for b in range(len(hidden_states)):
          if a == b:
              continue
          cosine_scores_a_b =  util.pytorch_cos_sim(hidden_states[a], hidden_states[b])
          print(cosine_scores_a_b)
          for i in range(len(cosine_scores_a_b)):
             bestscore = max(cosine_scores_a_b[i])
             bestmatch = np.argmax(cosine_scores_a_b[i])
             top3idx = np.argpartition(cosine_scores_a_b[i], -3)[-3:]
             top3scores= cosine_scores_a_b[i][top3idx]
             print(top3idx)
             print(top3scores)
             print("shouldnve printed")
             for n in [0,1,2]:
                 match_idx = top3idx[n]
                 print(n, a,b, i, match_idx, seqs[a][i], seqs[b][match_idx], cosine_scores_a_b[match_idx])

             try: 
                print(a, b, i, bestmatch, seqs[a][i], seqs[b][bestmatch], bestscore)

             except Exception as E:
                print(a, b, i, bestmatch, bestscore)
    
          #for complete mapping
          #   for j in range(len(cosine_scores_a_b)):
          #       pairs.append({seqs : [a,b], 'index': [i, j], 'score': cosine_scores_a_b[i][j]})

    

    print(pairs) 

 
    #Sort scores in decreasing order
    pairs = sorted(pairs, key=lambda x: x['score'], reverse=True)
    
    for pair in pairs[0:10]:
      i, j = pair['index']
      print("{} \t\t {} \t\t Score: {:.4f}".format(sentences[i], sentences[j], pair['score']))

    print(cos_scores)



    return True
 
 
def main(layers=None):
    # Use last four layers by default
    #layers = [-4, -3, -2, -1] if layers is None else layers
    layers = [-4, -3, -2, -1]     #tokenizer = AutoTokenizer.from_pretrained("bert-base-cased")
    #model = AutoModel.from_pretrained("bert-base-cased", output_hidden_states=True)
 
    model_name = 'prot_bert_bfd'
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name, output_hidden_states=True)

    #sent = "I like cookies ." 
    #idx = get_word_idx(sent, "cookies")
    seqs = ['H K C Q T C G K A F N R S S T L N T H A R I H A G N P', 'Y K C K Q C G K A F A R S G G L Q K H K R T H']
    word_embedding = get_word_vectors(seqs, tokenizer, model, layers)
    
    return word_embedding 
 
 
if __name__ == '__main__':
    main()


#input_ids = torch.tensor(tokenizer.encode("Hello, my dog is cute")).unsqueeze(0)  # Batch size 1
#outputs = model(input_ids)
#
