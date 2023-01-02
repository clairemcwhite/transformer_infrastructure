import io
import sys 
import argparse
import time
import torch
from Bio import SeqIO
from transformers import AutoModel, AutoTokenizer


def get_attention_args():
    parser = argparse.ArgumentParser()
    #                    help="Model directory Ex. /path/to/model_dir")
    parser.add_argument("-f", "--fasta", dest = "fasta_path", type = str, required = True,
                        help="Fasta file")
    parser.add_argument("-mo", "--model", dest = "model_path", type = str, required = True,
                        help="Model directory Ex. /path/to/model_dir")

    parser.add_argument("-ma", "--min_attn", dest = "min_attn", type = float, required = False, default= 0.1,
                        help="Minimum attention to plot, default: 0.1")
 
    parser.add_argument("-mu", "--mut", dest = "mut", type = str, required = False,
                        help="mutate position x to y. ex. '102_w' or 'p_102_w' ")
    parser.add_argument("-mf", "--mutfile", dest = "mutfile", type = str, required = False,
                        help="File with mutate positions x to y. One per line ex. '102_w' or 'p_102_w' ")
 
    parser.add_argument("-o", "--outfile", dest = "outfile", type = str, required = False,
                        help="Optional outfile name")


    args = parser.parse_args()
 
    return(args)


def parse_mut(mut):
    old = None
    if mut[0].isalpha():
          #meaning ex. I40V
          old = mut[0]
          pos = int(mut[1:-1]) 
          new = mut[-1]
    elif not mut[0].isalpha():
           #meaning ex. 40V
           pos = int(mut[0:-1])  
           new = mut[-1]
 
    return(old,  new, pos)

def format_tokens(tokens, mut = None):

    # mutation pos is indexed from 1
    print("tokens at format tokens", tokens)
    if mut is None:
        return(tokens)

    if mut is not None:

       old, new, pos = parse_mut(mut)
       print("Adding in mutation {}".format(mut))
       if old is not None:
          if not tokens[pos -1 ] == old:
               print("Expected amino acid {} not found at position {}. Instead found {}".format(old, pos, tokens[pos - 1]))

       tokens[pos - 1] = new # Replace with new position
    return(tokens)

def get_attn_data(model, tokenizer, tokens, min_attn = 0.1, start_index=0, end_index=None, max_seq_len=1024):


    #if max_seq_len:
    #    tokens = tokens[:max_seq_len - 2]  # Account for SEP, CLS tokens (added in next step)

    print("tokens", tokens)
    token_idxs = tokenizer.encode(tokens, is_split_into_words=True)#.tolist()
    #if max_seq_len:
    #    assert len(token_idxs) == min(len(tokens) + 2, max_seq_len)
    #else:
    #    assert len(token_idxs) == len(tokens) + 2

    print("get_inputs")    
    inputs = torch.tensor(token_idxs).unsqueeze(0).cuda()
    with torch.no_grad():
        attns = model(inputs)[-1]
        # Remove attention from <CLS> (first) and <SEP> (last) token
    print("trim attns")
    attns = [attn[:, :, 1:-1, 1:-1] for attn in attns]
    print("stack attns")
    attns = torch.stack([attn.squeeze(0) for attn in attns])
    attns = attns.tolist()
    print("attentions calculated")
    return(attns)





def load_model(model_path):
    '''
    Takes path to huggingface model directory
    Returns the model and the tokenizer
    '''
    print("load tokenizer")
    tokenizer = AutoTokenizer.from_pretrained(model_path)

    print("load model")
    model = AutoModel.from_pretrained(model_path, output_attentions=True)

    return(model, tokenizer)

def wrap_attns(model, tokenizer, tokens, min_attn = 0.1, mut = None):
    outlist = []
    tokens = format_tokens(tokens, mut)
    attns = get_attn_data(model, tokenizer, tokens, min_attn  = min_attn)
    num_layers = len(attns)
    num_heads = len(attns[0])
    for layer in range(1, num_layers + 1): # Max 31
         print("layer {} {} mutstatus {}".format(record.id, layer, mut ))
         for head in range(1, num_heads + 1): # Max 17
            if mut is not None:     
                identifier = "{}-{}-{}-{}".format(record.id, layer, head, mut)
            else:
                identifier = "{}-{}-{}".format(record.id, layer, head)
            attn_head = attns[layer -1][head - 1 ]
            #print(len(tokens))
            #print(len(attn_head))
            #print(len(attn_head[0]))
            complete = []
            for i in range(len(tokens)):
                if i in complete:
                   continue
                complete.append(i)
                for j in range(len(tokens)):
                    a = max(attn_head[i][j], attn_head[j][i])
                    if a is not None and a >= min_attn:
                          outlist.append([identifier, layer, head, tokens[i], i + 1, tokens[j], j + 1, a])
    return(outlist)

if __name__ == "__main__":

    args = get_attention_args()
    print(args)
  

    fasta_path = args.fasta_path
    model_path = args.model_path
    min_attn = args.min_attn
    mut = args.mut
    mutfile = args.mutfile
    attn_outfile = args.outfile
    model, tokenizer = load_model(model_path)

    mutlist = [None]
    if mutfile is not None:
         with open(mutfile, "r") as m:
            raw_muts = m.readlines()
         muts = [x.replace("\n", "") for x in raw_muts]
         mutlist = mutlist + muts
    if mut is not None:
        mutlist = mutlist + [mut]  # Always do wild-type first
    print(mutlist)

    if attn_outfile is None:
        if mut is not None: 
           attn_outfile = "{}-{}-attns.csv".format(fasta_path, mut)
        elif mutfile is not None:
           attn_outfile = "{}-muts-attns.csv".format(fasta_path)
        else:
            attn_outfile = "{}-attns.csv".format(fasta_path)

    with open(attn_outfile, "w") as o:
       o.write("identifier,proteinid,mut,layer,head,res1,res2,attention\n")
       with open(fasta_path) as handle:
           for record in SeqIO.parse(handle, "fasta"):
              tokens = list(str(record.seq))
              for mutation in mutlist:
                  outlist = wrap_attns(model, tokenizer, tokens, min_attn = min_attn, mut = mutation)
                  for identifier, layer, head, token1, pos1, token2, pos2, attn in outlist:
                     o.write("{},{},{},{},{},{}-{},{}-{},{}\n".format(identifier, record.id, mutation, layer, head, token1, pos1, token2, pos2, attn))




