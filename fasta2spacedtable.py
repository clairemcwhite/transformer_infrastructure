from transformer_infrastructure.hf_utils import parse_fasta
from Bio import SeqIO
import argparse

    

def get_fasta_args():

    parser = argparse.ArgumentParser()
    parser.add_argument("-f", "--fasta", dest = "fasta_path", type = str, required = True,
                        help="fasta of protein sequences")

    parser.add_argument("-a", "--add_spaces" , action = "store_true",
                        help="Flag if sequences already have spaces")
    parser.add_argument("-os", "--outsequences", dest = "outsequences", type = str, required = True,
                        help="output csv for table of identified and spaced out sequences (for conversion after embedding)")


    args = parser.parse_args()
    return(args)

if __name__ == "__main__":
    args = get_fasta_args()
    sequences = parse_fasta(args.fasta_path, args.outsequences, args.add_spaces)




