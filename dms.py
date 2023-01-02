import pandas as pd
import argparse
from functools import partial
from igraph import *
import io
import sys 
from time import time
import torch
import numpy as np
from torch.multiprocessing import Pool
from Bio import SeqIO
import copy
from transformer_infrastructure.hf_embed import load_model
#from transformers import AutoConfig

import line_profiler
profile = line_profiler.LineProfiler()
import atexit
atexit.register(profile.print_stats)
#from numba import jit
#import numba as nb

@profile
def get_attn_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("-f", "--fasta", dest = "fasta_path", type = str, required = True,
                        help="Fasta file")
    parser.add_argument("-n", "--num_processes", dest = "num_processes", type = int, required = False, default= 5,
                        help="Number of threads, default: 5")

    parser.add_argument("-ma", "--min_attn", dest = "min_attn", type = float, required = False, default= 0.1,
                        help="Minimum attention to plot, default: 0.1")
    parser.add_argument("-ml", "--mutlimit", dest = "mutlimit", type = int, required = False,
                        help="Limit attention calculations to n mutations. For testing")
    parser.add_argument("-mo", "--model", dest = "model_path", type = str, required = True,
                        help="Model directory Ex. /path/to/model_dir")
    parser.add_argument("-mu", "--mut", dest = "mut", type = str, required = False,
                        help="mutate position x to y. ex. '102_w' or 'p_102_w' ")
    parser.add_argument("-mf", "--mutfile", dest = "mutfile", type = str, required = False,
                        help="File with mutate positions x to y. One per line ex. '102_w' or 'p_102_w' ")

    parser.add_argument("-o", "--out", dest = "outfile", type = str, required = True,
                        help="Outfile name")
    args = parser.parse_args()
    return(args)

#def format_df_filtered(name, tbl):
#    ungrouped = tbl.reset_index()
#    G = Graph.DataFrame(ungrouped[['res1', 'res2']], use_vids=False)
#    G.es["weight"] = ungrouped['attention']#.tolist()
#    return(G)  


@profile
def attndf_to_graphlist(df):
    '''
    Parameters
    ----------
    df : Attention dataframe.

    Returns
    -------
    G_list : Graph list of the attention dataframe.
    group_keys : Names of the grouped columns (layer, head).
    '''    
    #print(df)
    df['name'] = df['layer'].astype(str).str.cat(df['head'].astype(str), sep='-')

    #df['name'] = df.apply(lambda row: "{}-{}".format(row.layer, row.head), axis=1)
    

    df_grouped = df.groupby(['name'])
    #group_keys = list(df_grouped.groups.keys()) 
    df_filtered = df_grouped[['res1', 'res2', 'attention']]
    G_dict = {}    

    #start = time()
    #graphlist = [format_df_filtered(x, y) for x,y in df_filtered]
    #newdict = dict.fromkeys( group_keys, graphlist)
    #end = time()
    #print(end-start)
    #start = time()
    for name, tbl in df_filtered:
        ungrouped = tbl.reset_index()
        #print(ungrouped)
        G = Graph.DataFrame(ungrouped[['res1', 'res2']], use_vids=False)
        G.es["weight"] = ungrouped['attention']#.tolist()
        G_dict[name] = G
    #end = time()
    #print(end - start)
    return G_dict

@profile
def compare_attn_networks(g1list, g2list, summary = True):
    '''
    Parameters
    ----------
    g1list : Graph list of the sequence.
    g2list : Graph list of the mutated sequence.
    Returns
    -------
    outlist : List of the values nodes1, nodes2, edges1, edges2, distinct_edges1, distinct_edges2, edge_weight1, edge_weight2.
    '''
    outdict = {}
    dict_combo =dict(g1list, **g2list)
    for layer_head in dict_combo.keys():   #zip(g1list, g2list):
        if layer_head not in g1list.keys():
           outdict[layer_head] = 0
           continue
        elif layer_head not in g2list.keys():
           outdict[layer_head] = 0
           continue
        else:
           g1 = g1list[layer_head].simplify(combine_edges = max)
           g2 = g2list[layer_head].simplify(combine_edges = max)
        nodes1 = g1.vcount()
        nodes2 = g2.vcount()
        edges1 = g1.ecount()
        edges2 = g2.ecount()
        
        inter = intersection([g1,g2]).simplify()
        disconnected_nodes = inter.vs.select(_degree=0)
        inter.delete_vertices(disconnected_nodes)
        distinct_edges1 = edges1 - inter.ecount()
        distinct_edges2 = edges2 - inter.ecount()
        
        vs = g1.vs.select(name_in=inter.vs['name'])
        intersect_weight_g1 = sum(g1.es.select(_source_in=vs, _target_in=vs)['weight'])
        total_weight_g1 = sum(g1.es['weight'])
        distinct_weight_g1 = total_weight_g1 - intersect_weight_g1\
            
        intersect_weight_g2 = sum(g2.es.select(_source_in=vs, _target_in=vs)['weight'])
        total_weight_g2 = sum(g2.es['weight'])
        distinct_weight_g2 = total_weight_g2 - intersect_weight_g2
        

        if summary == False:
            outlist.append([nodes1, nodes2, edges1, edges2, distinct_edges1, distinct_edges2, distinct_weight_g1, distinct_weight_g2, total_weight_g1, total_weight_g2])
        else:

            if edges1 +  edges2 ==0:
              score = 1
            else:
              score = (inter.ecount() / (edges1 + edges2 - inter.ecount()))


            outdict[layer_head] = score
    return outdict

@profile
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

@profile
def format_tokens(tokens, mut = None):
    # mutation pos is indexed from 1
    #print("tokens at format tokens", tokens)
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

@profile
def get_attn_data(model, tokenizer, tokens, model_type, min_attn = 0.1, start_index=0, end_index=None, max_seq_len=1024):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    #print("tokens", tokens)
    token_idxs = tokenizer.encode(tokens, is_split_into_words=True)#.tolist()

    inputs = torch.tensor(token_idxs).unsqueeze(0).cuda()
    model.to(device)
    inputs.to(device)
    #print(inputs.get_device())
    #print(model.get_device())
    with torch.no_grad():
        attns = model(inputs)[-1]

    # attns = list of length #layers
    # attns[0] = array of dim (# seqs, #heads, #aas, #aas)

    if model_type == "bert":
        # Remove attention from <CLS> (first) and <SEP> (last) token
        attns = [attn[:, :, 1:-1, 1:-1] for attn in attns]
    if model_type == "t5":
        # T5 models don't have the classifier <CLS> token
        attns = [attn[:, :, :-1, :-1] for attn in attns]

    attns = torch.stack([attn.squeeze(0) for attn in attns])#.cpu()
   
    # Now attns has shape (#layers, #heads, #aas, #aas)
    return(attns)

#@profile
#def load_model(model_path):
#    '''
#    Takes path to huggingface model directory
#    Returns the model and the tokenizer
#    '''
#    print("load tokenizer")
#    tokenizer = AutoTokenizer.from_pretrained(model_path)
#    print("load model")
#    model = AutoModel.from_pretrained(model_path, output_attentions=True)
#    return(model, tokenizer)

@profile
def get_attn_df(model, tokenizer, tokens, model_type, min_attn = 0.1, mut = None):

    tokens = format_tokens(tokens, mut)

    attns = get_attn_data(model, tokenizer, tokens, model_type, min_attn  = min_attn)
    attns = attns.cpu()
    print("attns calculated")

    layers_list, heads_list, aa1_list, pos1_list, aa2_list, pos2_list, attns_list = wrap_attns(np.array(attns), tokens, min_attn)
    #layers_list, heads_list, aa1_list, pos1_list, aa2_list, pos2_list, attns_list = wrap_attns(np.array(attns, dtype = "float32"), nb.typed.List(num_layers_list), nb.typed.List(num_heads_list), nb.typed.List(tokens), nb.typed.List(tokens_len), min_attn)
    print("attns filtered")    
    res1_list = ["-".join([x, str(y)]) for x,y in zip(aa1_list,pos1_list)]
    res2_list = ["-".join([x, str(y)]) for x,y in zip(aa2_list,pos2_list)]
    print("got reslists")
    df = pd.DataFrame(list(zip(layers_list, heads_list, res1_list, res2_list, attns_list)), columns=['layer','head','res1','res2','attention'])
    return(df)

@profile
def wrap_attns(attns, tokens, min_attn):
   '''
   Get indices where attn above min_attn
   Returns lists of all amino acid pairs above threshold
   for all layers and heads
   '''
   # Attns have dimensions (layers, heads, tokens, tokens)
   attns_idx = np.argwhere(attns >= min_attn)
   attns_sel = attns[attns >= min_attn] 
   attns_idx_T = attns_idx.T
   layers = attns_idx_T[0] + 1
   heads = attns_idx_T[1] + 1
   pos1s = attns_idx_T[2] + 1
   pos2s = attns_idx_T[3] + 1 
   aa1s = [tokens[x] for x in attns_idx_T[2]] 
   aa2s = [tokens[x] for x in attns_idx_T[3]] 
   return(layers, heads, aa1s, pos1s, aa2s, pos2s, attns_sel) 


@profile
def get_score_dict(mutation, tokens, model, tokenizer, model_type, glist_wt, min_attn):
                print(mutation) 
                tokens_for_mut = copy.deepcopy(tokens) # Necessary 
               
                df_mut_attns = get_attn_df(model, tokenizer, tokens_for_mut, model_type = model_type, min_attn = min_attn, mut = mutation)
                print("got attns")
                glist_mut = attndf_to_graphlist(df_mut_attns)
                print("got graphlist")
                score_dict = compare_attn_networks(glist_wt, glist_mut)
                print("got score dict")
                score_dict['mutation']  = mutation    
                return(score_dict)

if __name__ == "__main__":
    torch.multiprocessing.set_start_method('spawn')# good solution !!!!
    args = get_attn_args()
    fasta_path = args.fasta_path
    model_path = args.model_path
    attn_outfile = args.outfile
    min_attn = args.min_attn
    mut = args.mut
    num_processes = args.num_processes
    mutfile = args.mutfile
    mutlimit = args.mutlimit
    model, tokenizer, model_config = load_model(model_path, 
                        output_hidden_states=False,
                        output_attentions = True, 
                        return_config = True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("CUDA available?", torch.cuda.is_available())
    model.to(device)
    # Set to inference mode
    model.eval()
    # Allow parallel
    model.share_memory()
    mutlist = []
    if mutfile is not None:
         with open(mutfile, "r") as m:
            raw_muts = m.readlines()
         muts = [x.replace("\n", "") for x in raw_muts]
         mutlist = mutlist + muts
    if mut is not None:
        mutlist = mutlist + [mut]  # Always do wild-type first
    print(mutlist)


    
    amino_acids = ['A', 'R', 'N', 'D', 'C', 'E', 'Q', 'G', 'H', 'I', 'L', 'K', 'M', 'F', 'P', 'S', 'T', 'W', 'Y', 'V']
     #g1list, group_names = attndf_to_graphlist(df1) 
    

    model_type = model_config.model_type
    print("This is a {} model".format(model_type))
    num_layers = model_config.num_hidden_layers
    num_heads = model_config.num_attention_heads

    print("num_layers:", num_layers)
    print("num_heads:", num_heads)
   
    headlist = []
    for layer in range(1, num_layers + 1):
        for head in range(1, num_heads + 1):
            headlist.append(str(layer) + "-" + str(head))
    
    #o.write("proteinID,mutation{}\n".format(headstr))#"edges1,edges2,distinct_edges1,distinct_edges2,distinct_edge_weight1,distinct_edge_weight2,total_edge_weight1,total_edge_weight2\n")
    print(fasta_path)
    with open(fasta_path) as handle:
        for record in SeqIO.parse(handle, "fasta"):
            print(record)
            if len(mutlist) ==  0:  # If havent provided a mutation list, do full dms scan 
              for seq_index, seq in enumerate(record.seq):
   
                for  mut_seq in amino_acids:
                    if (seq != mut_seq):
                        mutation = seq + str(seq_index + 1) + mut_seq
                        mutlist = mutlist + [mutation]
            if mutlimit:
                mutlist = mutlist[0:mutlimit]
            print("Mutlist", mutlist)

            tokens = list(str(record.seq))
            #wild type attention calc 

            print("start WT calculation")
            df_wt_attns = get_attn_df(model, tokenizer, tokens, model_type = model_type, min_attn = min_attn)
           
            glist_wt = attndf_to_graphlist(df_wt_attns)
            ## ['sp|Q92781|RDH5_HUMAN-30-16', 30, 16, 'I', 198, 'G', 156, 0.10365235060453415]i
            name = str(record.id)

            scoredict_partial = partial(get_score_dict, tokens = tokens, model = model, tokenizer= tokenizer, model_type = model_type, glist_wt = glist_wt, min_attn = min_attn)
            with Pool(processes = num_processes) as pool:
                list_of_scoredicts = pool.map(scoredict_partial, mutlist)

            #for mutation in mutlist:
            #    score_dict = get_score_dict(mutation, tokens, model, tokenizer, model_type, glist_wt, min_attn = min_attn)
            #    print(score_dict)
            #    list_of_scoredicts.append(score_dict)
            
            
                #print(mutation) 
                #tokens_for_mut = copy.deepcopy(tokens) # Necessary 
                #df_mut_attns = get_attn_df(name, model, tokenizer, tokens_for_mut, min_attn = min_attn, mut = mutation)
                #glist_mut, group_names = attndf_to_graphlist(df_mut_attns)
                #score_dict = compare_attn_networks(glist_wt, glist_mut)
                #score_dict['mutation']  = mutation    
                #print(score_dict)
            scores_tbl = pd.DataFrame(list_of_scoredicts)
            scores_tbl = scores_tbl.fillna(1) # If attentions missing from heads in both mut and wt, they're the same
            observed_heads = [x for x in headlist if x in scores_tbl.columns]
            scores_tbl['proteinID'] = record.id
            column_order = ['proteinID', 'mutation'] + observed_heads
            scores_tbl= scores_tbl[column_order]
            print(scores_tbl)
            scores_tbl.to_csv(attn_outfile, index = False, float_format='%.5f')
                       




#def wrap_attns_old(attns, num_layers_list, num_heads_list, tokens, tokens_len, min_attn):
#    layers = []
#    heads = []
#    attns_list = []
#    aa1s = []
#    pos1s = []
#    aa2s = []
#    pos2s = []
#    for layer in num_layers_list: # Max 31
#         print(layer)
#         for head in num_heads_list: # Max 17
#            attn_head = attns[layer - 1][head - 1]
#            print(attn_head.shape)
#            complete = []
#            for i in tokens_len:
#                complete.append(i)
#                for j in tokens_len:
#                    if j in complete:
#                       continue
#                    a = attn_head[i][j]
#                    #print(a.shape)
#                    if a >= min_attn and a is not None:
#                        
#                        pos1 = i + 1
#                        pos2 = j + 1
#                        aa1 = tokens[i]
#                        aa2 = tokens[j]
#                        aa1s.append(aa1)
#                        pos1s.append(pos1)
#                        aa2s.append(aa2)
#                        pos2s.append(pos2)
#                        attns_list.append(a)
#                        heads.append(head)
#                        layers.append(layer)
#    return(layers, heads, aa1s, pos1s, aa2s, pos2s, attns_list)
#
