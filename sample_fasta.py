from Bio import SeqIO
import random
import argparse

def get_samplefasta_args():

    parser = argparse.ArgumentParser()
    parser.add_argument("-n",  dest = "n", type = int, required = True,
                        help="Number of records to save")
    parser.add_argument("-f", "--fasta", dest = "fasta_path", type = str, required = True,
                        help="path to a fasta of protein sequences")
    parser.add_argument("-o", "--outfasta", dest = "outfasta_path", type = str, required = True,
                        help="path to output fasta")

    args = parser.parse_args()

    return(args)


def sample_fasta(fasta_path, outfasta_path, n = 10000):
   random.seed(42)
   count = 0

   with open(outfasta_path, "w") as handle:


       records = []
       for record in SeqIO.parse(fasta_path, "fasta"):
              records.append(record)

       # Select n records
       randsamp = random.sample(range(0, len(records) ), n) 
       for record in records:
          if count in randsamp:
             SeqIO.write(record, handle, "fasta")
          count = count + 1

if __name__ == "__main__":

    args = get_samplefasta_args()

    fasta_path = args.fasta_path
    outfasta_path = args.outfasta_path
    n = args.n
    #fasta_path = "/scratch/gpfs/cmcwhite/qfo_2020/qfo_2020.fasta"
    #outfasta_path = "/scratch/gpfs/cmcwhite/qfo_2020/qfo_sample10000.fasta"

    sample_fasta(fasta_path, outfasta_path, n = n)


