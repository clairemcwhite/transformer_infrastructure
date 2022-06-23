import pandas as pd
import argparse
from igraph import *
import io
import sys 
import time
import torch
import numpy as np
from Bio import SeqIO
from transformers import AutoModel, AutoTokenizer, BertConfig
import copy
#import line_profiler
#profile = line_profiler.LineProfiler()
#import atexit
#atexit.register(profile.print_stats)
from numba import jit
import numba as nb
#@profile
def get_attn_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("-f", "--fasta", dest = "fasta_path", type = str, required = True,
                        help="Fasta file")
    parser.add_argument("-mo", "--model", dest = "model_path", type = str, required = True,
                        help="Model directory Ex. /path/to/model_dir")
    parser.add_argument("-o", "--out", dest = "outfile", type = str, required = True,
                        help="Outfile name")
    args = parser.parse_args()
    return(args)

#@profile
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
    df_grouped = df.groupby(['layer', 'head'])
    #print(df[['layer', 'head']].unique())
    #print(df_grouped)
    group_keys = list(df_grouped.groups.keys())
    df_filtered = df_grouped[['res1', 'res2', 'attention']]
    #for d in df_filtered:
    #    print(d)
    #print(df_filtered)
    #G_list = []
    G_dict = {}    
    for name, tbl in df_filtered:
        ungrouped = tbl.reset_index()
        G = Graph.DataFrame(ungrouped[['res1', 'res2']])
        #print("G_start", G)
        G.es["weight"] = ungrouped['attention'].tolist()
        #print("G w weight", G)
        name_str =  "-".join([str(x) for x in name]) # Join tuple (1,4) to "1-4"
        G_dict[name_str] = G
        #G_list.append(G)
    #print("g list")
    #print(G_dict[(1,10)])
    return G_dict, group_keys

#@profile
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
           g1 = g1list[layer_head]
           g2 = g2list[layer_head]
        nodes1 = g1.vcount()
        nodes2 = g2.vcount()
        edges1 = g1.ecount()
        edges2 = g2.ecount()
        
        inter = intersection([g1,g2]).simplify()
        disconnected_nodes = inter.vs.select(_degree=0)
        inter.delete_vertices(disconnected_nodes)
        #print("g1", g1)
        #print("g2", g2)
        #print("inter",inter)
        #for edge in inter.es():
        #    print(edge.source, edge.target)
        distinct_edges1 = edges1 - inter.ecount()
        distinct_edges2 = edges2 - inter.ecount()
        
        vs = g1.vs.select(name_in=inter.vs['name'])
        #print("vertex_len", len(g1.vs), len(vs))
        intersect_weight_g1 = sum(g1.es.select(_source_in=vs, _target_in=vs)['weight'])
        total_weight_g1 = sum(g1.es['weight'])
        distinct_weight_g1 = total_weight_g1 - intersect_weight_g1\
            
        intersect_weight_g2 = sum(g2.es.select(_source_in=vs, _target_in=vs)['weight'])
        total_weight_g2 = sum(g2.es['weight'])
        distinct_weight_g2 = total_weight_g2 - intersect_weight_g2
        
        print("shared_nodes, shared_edges, nodes1, nodes2, edges1, edges2, distinct_edges1, distinct_edges2, distinct_weight_g1, distinct_weight_g2, total_weight_g1, total_weight_g2") 
        print(inter.vcount(), inter.ecount(), nodes1, nodes2, edges1, edges2, distinct_edges1, distinct_edges2, distinct_weight_g1, distinct_weight_g2, total_weight_g1, total_weight_g2) 
        if summary == False:
            outlist.append([nodes1, nodes2, edges1, edges2, distinct_edges1, distinct_edges2, distinct_weight_g1, distinct_weight_g2, total_weight_g1, total_weight_g2])
        else:
            score = ( 2 * inter.vcount() + inter.ecount()) / (nodes1 + nodes2 + edges1 + edges2)
            print("node/edge overlap  :", score)
            score = (inter.ecount() / min(edges1, edges2))
            print("Overlap Coefficient:", score)
            score = (inter.ecount() / (edges1 + edges2 - inter.ecount()))
            print("Jaccard Coefficient:", score)

            outdict[layer_head] = score
    return outdict

#@profile
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

#@profile
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

#@profile
def get_attn_data(model, tokenizer, tokens, min_attn = 0.1, start_index=0, end_index=None, max_seq_len=1024):
    #if max_seq_len:
    #    tokens = tokens[:max_seq_len - 2]  # Account for SEP, CLS tokens (added in next step)
    #print("tokens", tokens)
    token_idxs = tokenizer.encode(tokens, is_split_into_words=True)#.tolist()
    #if max_seq_len:
    #    assert len(token_idxs) == min(len(tokens) + 2, max_seq_len)
    #else:
    #    assert len(token_idxs) == len(tokens) + 2
    inputs = torch.tensor(token_idxs).unsqueeze(0)
    with torch.no_grad():
        attns = model(inputs)[-1]
        # Remove attention from <CLS> (first) and <SEP> (last) token
    attns = [attn[:, :, 1:-1, 1:-1] for attn in attns]
    attns = torch.stack([attn.squeeze(0) for attn in attns])
    attns = attns.tolist()
    return(attns)

#@profile
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

#@profile
def wrap_attns_identifiers(name, model, tokenizer, tokens, min_attn = 0.1, mut = None):
    tokens = format_tokens(tokens, mut)
    attns = get_attn_data(model, tokenizer, tokens, min_attn  = min_attn)
    num_layers_list = list(range(1, len(attns) + 1)) # actual line
    #num_layers_list = list(range(1, 3))
    #num_heads = len(attns[0])
    num_heads_list = list(range(1, len(attns[0]) + 1))
    tokens_len = list(range(len(tokens)))
    layers_list, heads_list, aa1_list, pos1_list, aa2_list, pos2_list, attns_list = wrap_attns(np.array(attns, dtype = "float32"), nb.typed.List(num_layers_list), nb.typed.List(num_heads_list), nb.typed.List(tokens), nb.typed.List(tokens_len), min_attn)
    
    res1_list = ["-".join([x, str(y)]) for x,y in zip(aa1_list,pos1_list)]
    res2_list = ["-".join([x, str(y)]) for x,y in zip(aa2_list,pos2_list)]
    df = pd.DataFrame(list(zip(layers_list, heads_list, res1_list, res2_list, attns_list)), columns=['layer','head','res1','res2','attention'])
    
    #attn_indices= [idx for idx,val in enumerate(outlist_attn) if val >= min_attn]
    #outlist_attn = [val for idx,val in enumerate(outlist_attn) if idx in attn_indices]
    #print(len(outlist_attn), len(attn_indices), attn_indices[0:10])
    #print(outlist_attn[0:5])

    #outlist_id = wrap_attns(name, nb.typed.List(num_layers_list), nb.typed.List(num_heads_list), nb.typed.List(tokens), nb.typed.List(tokens_len), mut, attn_indices)

    #old code to zip list of lists and list
    #outlist = [i + [j] for i, j in zip(outlist_id, outlist_attn)]
    #print(outlist_id[0:5])
    #print(outlist_attn[0:5])
    #print(outlist[0:5])
    return(df)

@jit(nopython=False)
def wrap_attns(attns, num_layers_list, num_heads_list, tokens, tokens_len, min_attn):
    layers = []
    heads = []
    attns_list = []
    aa1s = []
    pos1s = []
    aa2s = []
    pos2s = []
    for layer in num_layers_list: # Max 31
         for head in num_heads_list: # Max 17
            attn_head = attns[layer - 1][head - 1]
            complete = []
            for i in tokens_len:
                complete.append(i)
                for j in tokens_len:
                    if j in complete:
                       continue
                    a = attn_head[i][j]
                    if a >= min_attn and a is not None:
                        pos1 = i + 1
                        pos2 = j + 1
                        aa1 = tokens[i]
                        aa2 = tokens[j]
                        aa1s.append(aa1)
                        pos1s.append(pos1)
                        aa2s.append(aa2)
                        pos2s.append(pos2)
                        attns_list.append(a)
                        heads.append(head)
                        layers.append(layer)
    return(layers, heads, aa1s, pos1s, aa2s, pos2s, attns_list)

if __name__ == "__main__":
    args = get_attn_args()
    fasta_path = args.fasta_path
    model_path = args.model_path
    attn_outfile = args.outfile
    min_attn = 0.1
    model, tokenizer = load_model(model_path)
    
    amino_acids = ['A', 'R', 'N', 'D', 'C', 'E', 'Q', 'G', 'H', 'I', 'L', 'K', 'M', 'F', 'P', 'S', 'T', 'W', 'Y', 'V']
    mutlist = []
    #g1list, group_names = attndf_to_graphlist(df1) 
    
    bert_config = BertConfig.from_pretrained(model_path)
    num_layers = bert_config.num_hidden_layers
    num_heads = bert_config.num_attention_heads
    
    headlist = []
    for layer in range(1, num_layers + 1):
        for head in range(1, num_heads + 1):
            headlist.append(str(layer) + "-" + str(head))
    
    #o.write("proteinID,mutation{}\n".format(headstr))#"edges1,edges2,distinct_edges1,distinct_edges2,distinct_edge_weight1,distinct_edge_weight2,total_edge_weight1,total_edge_weight2\n")
    with open(fasta_path) as handle:
        for record in SeqIO.parse(handle, "fasta"):
            for seq_index, seq in enumerate(record.seq):
                for aa_index, mut_seq in enumerate(amino_acids):
                    if (seq != mut_seq):
                        mutation = seq + str(seq_index+1) + mut_seq
                        #print(mutation) 
                        mutlist.append(mutation)
            tokens = list(str(record.seq))
            #wild type attention calc: 
            #outtable_wt = []
            df_wt_attns = wrap_attns_identifiers(name, model, tokenizer, tokens, min_attn = min_attn)
            glist_wt, group_names = attndf_to_graphlist(df_wt_attns)
            ## ['sp|Q92781|RDH5_HUMAN-30-16', 30, 16, 'I', 198, 'G', 156, 0.10365235060453415]i
            #df_wt_attns = pd.DataFrame.from_records(outlist_wt, columns=['ProteinID', 'layer', 'head', 'res1', 'pos1', 'res2', 'pos2', 'attention'])

            #outtable_wt = []
            #for layer, head, token1, pos1, token2, pos2, attn in outlist_wt:
            #    outtable_wt.append([layer, head, "-".join([token1, str(pos1)]), "-".join([token2, str(pos2)]), attn])
            #df_wt_attns = pd.DataFrame.from_records(outtable_wt, columns=['layer', 'head', 'res1', 'res2', 'attention'])               
            #print(glist_wt[0:5] )
            name = str(record.id)
            list_of_scoredicts = []
            for mutation in mutlist[0:3]:
                print(mutation) 
                tokens_for_mut = copy.deepcopy(tokens) # Necessary 
                df_mut_attns = wrap_attns_identifiers(name, model, tokenizer, tokens_for_mut, min_attn = min_attn, mut = mutation)
                glist_mut, group_names = attndf_to_graphlist(df_mut_attns)
                score_dict = compare_attn_networks(glist_wt, glist_mut)
                score_dict['mutation']  = mutation    
                list_of_scoredicts.append(score_dict)
            scores_tbl = pd.DataFrame(list_of_scoredicts)
            print(scores_tbl)
            scores_tbl = scores_tbl.fillna(1) # If attentions missing from heads in both mut and wt, they're the same
            observed_heads = [x for x in headlist if x in scores_tbl.columns]
            scores_tbl['proteinID'] = record.id
            column_order = ['proteinID', 'mutation'] + observed_heads
            scores_tbl= scores_tbl[column_order]
            print(scores_tbl)
            scores_tbl.to_csv(attn_outfile, index = False, float_format='%.3f')
                        
