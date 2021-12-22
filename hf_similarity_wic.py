from transformer_infrastructure.hf_utils import parse_fasta, get_hidden_states, build_index
from transformer_infrastructure.hf_embed import get_embeddings, parse_fasta_for_embed
import pandas as pd
import time
from sentence_transformers import util

from Bio import SeqIO
import pickle
import argparse

import numpy as np
import torch
from transformers import AutoTokenizer, AutoModel
import logging
import faiss
#fasta = '/scratch/gpfs/cmcwhite/quantest2/QuanTest2/Test/zf-CCHH.vie'

def get_seqsim_args():

    parser = argparse.ArgumentParser()
    parser.add_argument("-m", "--model", dest = "model_name", type = str, required = True,
                        help="Model directory Ex. /path/to/model_dir")

    parser.add_argument("-st", "-sequence_table", dest = "sequence_path", type = str, required = False,
                        help="Path to table of sequences to evaluate in csv (id,sequence) no header. Output of utils.parse_fasta")
    parser.add_argument("-f", "-fasta", dest = "fasta_path", type = str, required = False,
                        help="Path to fasta of sequences to evaluate")

    parser.add_argument("-n", "--dont_add_spaces" , action = "store_true",
                        help="Flag if sequences already have spaces")

    parser.add_argument("-o", "--outfile", dest = "outfile", type = str, required = True,
                        help="output csv for table of word attributions")
    parser.add_argument("-k", dest = "k", type = int, required = False,
                        help="If present, limit to k closest sequences")
    #parser.add_argument("-p", dest = "percent", type = float, required = False,
    #                    help="If present, limit to top percent similar")


    args = parser.parse_args()
    return(args)  

 
def get_sequence_similarity(layers, model_name, seqs, seqs_spaced, seq_names, outfile, logging, k):
    # Use last ten layers by default
    #layers = [-10, -9,-8,-7, -6, -5, -4, -3, -2, -1] if layers is None else layers
    # Add more cpus if memory error. 4cpus/1000 sequences

    logging.info("load tokenizer")
    #tokenizer = AutoTokenizer.from_pretrained(model_name)
    logging.info("load model")
    #model = AutoModel.from_pretrained(model_name, output_hidden_states=True)
    logging.info("model loaded")
    #x = [l[i:i + n] for i in range(0, len(l), n)] 
    #logging.info(x)
    #seqs_list = np.array_split(lst, 5)

    #enc_hidden_states = embed_sequences(model_name, seqs, False, False)
 
    seqlens = [len(x) for x in seqs]
    padding = 5
    embedding_dict = get_embeddings(seqs_spaced,
                                    model_name,
                                    seqlens = seqlens,
                                    get_sequence_embeddings = True,
                                    get_aa_embeddings = False,
                                    layers = layers,
                                    padding = padding)
    enc_hidden_states = embedding_dict['sequence_embeddings']
    #enc_hidden_states = embed_sequences(model_pname, seqs, False, False)
    logging.info(enc_hidden_states.shape)
    
    #n = 100
    #seqs_batches = [seqs[i:i + n] for i in range(0, len(seqs), n)]
    logging.info("start encoding")
    
    # Encoding uses lots of memory
    # Avoid by either increasing cpus or sequences in batches 
    # Or add to index in batches?
    #for i in range(len(seqs_batches)):
    #    logging.info(i)
    #    hidden_states = get_hidden_states(seqs_batches[i], model, tokenizer, layers)
     
        # Get cls embedding as proxy for whole sentence 
    #    logging.info("pre hidden")
    #    enc_hidden_states_batch = hidden_states[:,0,:]
    #    if i == 0:
    #         enc_hidden_states = enc_hidden_states_batch
    #    else:
    #         enc_hidden_states = torch.cat([enc_hidden_states, enc_hidden_states_batch])

    logging.info("post_hidden")
    logging.info("Start comparison")
    start = time.time()


    index = build_index(enc_hidden_states) 
    #
    k = len(seq_names)
    print(k)
    distance, index = index.search(enc_hidden_states, k)
    end = time.time()
    tottime = end - start
    logging.info("compare complete in {} s".format(tottime))

    #print(distance)
    pairs = []
    complete = []
    with open(outfile, "w") as o:
        for i in range(len(index)):
            complete.append(seq_names[i])
            row =index[i]
            for j in range(len(row)):
              if seq_names[row[j]] in complete:
                continue
              name1 = seq_names[i]
              name2 = seq_names[row[j]]
              
              D =  1 - round(distance[i,j], 2)
              pairs.append([name1, name2, D])
              #print(name1, name2,D)
              #print(i, row[j], seq_names[i], seq_names[row[j]], distance[i,j])
              o.write("{}\t{}\t{}\n".format(name1,name2,D))
 

    return 1
 
 
if __name__ == '__main__':
    # Embedding not good on short sequences without context Ex. HEIAI vs. HELAI, will select terminal I for middle I, instead of context match L
    # Potentially maximize local score? 
    # Maximize # of matches
    # How to get sequence info?
    log_format = "%(asctime)s::%(levelname)s::%(name)s::"\
             "%(filename)s::%(lineno)d::%(message)s"
    logging.basicConfig(filename= "seqsim_logger.txt", level='DEBUG', format=log_format)

    logging.info("is this running?")
    args = get_seqsim_args()
    logging.info("load sequences")
    fasta_path = args.fasta_path
    model_name = args.model_name
    outfile = args.outfile
    k = args.k
    #if args.fasta_path:
    #   fasta_tbl = args.fasta_path + ".txt"
    #   sequence_lols = parse_fasta(args.fasta_path, fasta_tbl, args.dont_add_spaces)
    #
    #   df = pd.DataFrame.from_records(sequence_lols,  columns=['id', 'sequence', 'sequence_spaced'])
    #logging.info("sequences loaded")
    seq_names, seqs, seqs_spaced = parse_fasta_for_embed(fasta_path, extra_padding = True)
    #seq_names = df['id'].tolist()
    #seqs = df['sequence_spaced'].tolist()

    max_length = 1024
    seqs_spaced = [x[:2*max_length-2] for x in seqs_spaced]

    layers = [-10, -9, -8, -7, -6, -5, -4, -3, -2, -1]
    #model_name = 'prot_bert_bfd'
    #sqs = ['A A H K C Q T C G K A F N R S S T L N T H A R I H Y A G N P', 'Y K C K Q C G K A F A R S G G L Q K H K R T H']
    #seqs = ['H E A L A I', 'H E A I A L', 'H E E L A H']

    #seq_names = ['seq1','seq2', 'seq3']
    get_sequence_similarity(layers, model_name, seqs, seqs_spaced, seq_names, outfile, logging, k)


