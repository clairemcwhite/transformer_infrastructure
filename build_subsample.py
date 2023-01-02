#!/usr/bin/env python3
import numpy as np
from random import sample

from transformer_infrastructure.hf_embed import parse_fasta_for_embed 

from Bio import SeqIO
#from Bio.Seq import Seq
from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord



import pickle
import argparse




def subsamp_sequences(seqs, seq_names, numseqs,  ref = False):

 
    print("Including reference: ", ref)
    if not ref:
       indexes = list(range(0, len(seqs)))
       indexes = sample(indexes, numseqs)
    else: 
       indexes = list(range(3, len(seqs)))
       indexes = sample(indexes, numseqs - 3)
       indexes = [0,1,2] + indexes
    indexes = np.sort(indexes)
    print(indexes)
    
    select_seqs = [seqs[i] for i in indexes]
    select_seq_names = [seq_names[i] for i in indexes]
    return(select_seqs, select_seq_names, indexes) 

def subsamp_embeddings(embedding_dict, indexes):

        select_embedding_dict = {}
        select_embedding_dict['sequence_embeddings'] = np.take(embedding_dict['sequence_embeddings'], indexes, 0)
        select_embedding_dict['aa_embeddings'] = np.take(embedding_dict['aa_embeddings'], indexes, 0)
       
        return(select_embedding_dict) 




# Make parameter actually control this
#def format_sequences(fasta, padding =  5):
#   
#    # What are the arguments to this? what is test.fasta? 
#    seq_names, seqs, seqs_spaced = parse_fasta_for_embed(fasta, extra_padding = True)
#    
#    return(seq_names, seqs, seqs_spaced)


def get_embeddings_args():

    parser = argparse.ArgumentParser("This is a function to subsample a set of embeddings")
    parser.add_argument("-i", "--in", dest = "fasta_path", type = str, required = True,
                        help="Path to fasta")
    parser.add_argument("-e", "--emb", dest = "embedding_path", type = str, required = False,
                        help="Path to embeddings")

    parser.add_argument("-n", "--numseqs", dest = "numseqs", type = int, required = True,
                        help="The number of total sequences to include after sampling")

    parser.add_argument("-r", "--ref", dest = "ref", action = "store_true",
                        help="If flagged, include first three sequences as the reference sequences")

    #parser.add_argument("-o", "--pkl_out", dest = "pkl_out", type = str, required = True,
    #                    help="Path to outfile")


    args = parser.parse_args()

    return(args)


 
if __name__ == '__main__':

    args = get_embeddings_args()

    embedding_path = args.embedding_path
    fasta_path = args.fasta_path
    ref = args.ref 
    numseqs = args.numseqs
    padding = 5 


    fasta_out = "{}.{}seqs.fasta".format(fasta_path, numseqs)

    seq_names, seqs, seqs_spaced = parse_fasta_for_embed(fasta_path, padding = 0)

    

    select_seqs, select_seq_names, indexes = subsamp_sequences(seqs, seq_names, numseqs, ref = ref)

    records = []
    for i in range(len(select_seqs)):
         newrecord = SeqRecord(
                       Seq(select_seqs[i]),
                       id=select_seq_names[i],
                       description = "")
         records.append(newrecord)
    with open(fasta_out, "w") as handle:

         SeqIO.write(records, handle, "fasta")

    if embedding_path:
        if "128pca" in embedding_path:
            pkl_out = "{}.128pca.pkl".format(fasta_out)
        else:
            pkl_out = "{}.pkl".format(fasta_out)
        pkl_log = "{}.description".format(pkl_out)
    
        with open(embedding_path, "rb") as f:
              embedding_dict = pickle.load(f)
        select_embedding_dict = subsamp_embeddings(embedding_dict, indexes)
   
        with open(pkl_out, "wb") as fOut:
               pickle.dump(select_embedding_dict, fOut, protocol=pickle.HIGHEST_PROTOCOL)
        with open(pkl_log, "w") as pOut:
                pOut.write("Object {} dimensions: {}\n".format('sequence_embeddings', select_embedding_dict['sequence_embeddings'].shape))
                pOut.write("Object {} dimensions: {}\n".format('aa_embeddings', select_embedding_dict['aa_embeddings'].shape))
    
    
                pOut.write("Contains sequences:\n")
                for x in select_seq_names:
                  pOut.write("{}\n".format(x))
    
    
