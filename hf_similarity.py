from transformer_infrastructure.hf_utils import parse_fasta, get_hidden_states
import pandas as pd

from sentence_transformers import util

from Bio import SeqIO
import pickle
import argparse

import numpy as np
import torch
from transformers import AutoTokenizer, AutoModel
import logging
print("are we loading libraries??")
#fasta = '/scratch/gpfs/cmcwhite/quantest2/QuanTest2/Test/zf-CCHH.vie'

def get_seqsim_args():

    parser = argparse.ArgumentParser()
    parser.add_argument("-m", "--model", dest = "model_path", type = str, required = True,
                        help="Model directory Ex. /path/to/model_dir")

    parser.add_argument("-st", "-sequence_table", dest = "sequence_path", type = str, required = False,
                        help="Path to table of sequences to evaluate in csv (id,sequence) no header. Output of utils.parse_fasta")
    parser.add_argument("-f", "-fasta", dest = "fasta_path", type = str, required = False,
                        help="Path to fasta of sequences to evaluate")

    parser.add_argument("-n", "--dont_add_spaces" , action = "store_true",
                        help="Flag if sequences already have spaces")

    parser.add_argument("-o", "--outfile", dest = "outfile", type = str, required = True,
                        help="output csv for table of word attributions")


    args = parser.parse_args()
    return(args)  

 
def get_sequence_similarity(layers, model_path, seqs, seq_names, outfile, logging):
    # Use last four layers by default
    #layers = [-4, -3, -2, -1] if layers is None else layers
    # Add more cpus if memory error. 4cpus/1000 sequences

    logging.info("load tokenizer")
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    logging.info("load model")
    model = AutoModel.from_pretrained(model_path, output_hidden_states=True)
    logging.info("model loaded")
    #x = [l[i:i + n] for i in range(0, len(l), n)] 
    #logging.info(x)
    #seqs_list = np.array_split(lst, 5)

    n = 100
    seqs_batches = [seqs[i:i + n] for i in range(0, len(seqs), n)]
    logging.info("start encoding")
    
    # Encoding uses lots of memory
    # Avoid by either increasing cpus or sequences in batches 
    for i in range(len(seqs_batches)):
        logging.info(i)
        hidden_states = get_hidden_states(seqs_batches[i], model, tokenizer, layers)
     
        # Get cls embedding as proxy for whole sentence 
        logging.info("pre hidden")
        cls_hidden_states_batch = hidden_states[:,0,:]
        if i == 0:
             cls_hidden_states = cls_hidden_states_batch
        else:
             cls_hidden_states = torch.cat([cls_hidden_states, cls_hidden_states_batch])

    print(cls_hidden_states)
    logging.info("post_hidden")
    
    cosine_scores =  util.pytorch_cos_sim(cls_hidden_states, cls_hidden_states)
    logging.info(cosine_scores)

    pairs = []
    complete = []
    with open(outfile, "w") as o:
        for i in range(len(cosine_scores)):
          complete.append(i)
          for j in range(len(cosine_scores)):
             if j in complete:
                 continue
    
             o.write("{}\t{}\t{}\n".format(seq_names[i], seq_names[j], cosine_scores[i,j]))

    

    #match_edges = get_wholeseq_similarities(hidden_states, seqs, seq_names)
    
    #seq_edges = get_seq_edges(seqs, seq_names)
    #all_edges = seq_edges + match_edges 
    #for x in all_edges:
    #  print(x)

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
    if args.fasta_path:
       fasta_tbl = args.fasta_path + ".txt"
       sequence_lols = parse_fasta(args.fasta_path, fasta_tbl, args.dont_add_spaces)

       df = pd.DataFrame.from_records(sequence_lols,  columns=['id', 'sequence', 'sequence_spaced'])
    logging.info("sequences loaded")
    seq_names = df['id'].tolist()
    seqs = df['sequence_spaced'].tolist()

    max_length = 1024
    seqs = [x[:max_length-2] for x in seqs]
    layers = [-4, -3, -2, -1]
    #model_path = 'prot_bert_bfd'
    model_path = args.model_path
    outfile = args.outfile
    #sqs = ['A A H K C Q T C G K A F N R S S T L N T H A R I H Y A G N P', 'Y K C K Q C G K A F A R S G G L Q K H K R T H']
    #seqs = ['H E A L A I', 'H E A I A L', 'H E E L A H']

    #seq_names = ['seq1','seq2', 'seq3']
    get_sequence_similarity(layers, model_path, seqs, seq_names, outfile, logging)


#input_ids = torch.tensor(tokenizer.encode("Hello, my dog is cute")).unsqueeze(0)  # Batch size 1
#outputs = model(input_ids)
#
