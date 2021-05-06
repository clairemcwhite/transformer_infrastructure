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
 
 

def get_word_vectors(sent, tokenizer, model, layers):
    """Get a word vector by first tokenizing the input sentence, getting all token idxs
       that make up the word of interest, and then `get_hidden_states
     BramVanroy`."""
    sent2   = "M A F G N I J K"
    encoded = tokenizer.batch_encode_plus([sent, sent2], return_tensors="pt", padding=True)
    # get all token idxs that belong to the word of interest
    #token_ids_word = np.where(np.array(encoded.word_ids()) == idx)
    
    #print(encoded.word_ids())
    #print(encoded.word_ids()[0])

    #for  x in encoded.word_ids():
    hidden_states = get_hidden_states(encoded, model, layers)

    cos_scores = util.pytorch_cos_sim(hidden_states, hidden_states)        
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
    sent = "M A P R A G F E T Q A"
    word_embedding = get_word_vectors(sent, tokenizer, model, layers)
    
    return word_embedding 
 
 
if __name__ == '__main__':
    main()


#input_ids = torch.tensor(tokenizer.encode("Hello, my dog is cute")).unsqueeze(0)  # Batch size 1
#outputs = model(input_ids)
#
