from Bio import SeqIO
from Bio import AlignIO

import argparse



def convert_to_fasta(infile, outfile, current_format = "msf", new_format = "fasta"):
       
    with open(infile, "r") as input_handle:
       with open(outfile, "w") as output_handle:

          alignments = AlignIO.parse(input_handle, current_format)
          AlignIO.write(alignments, output_handle, new_format)


def get_args():

    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--infile", dest = "infile", type = str, required = True,
                        help="Input multiple sequence alignment")
    parser.add_argument("-o", "--outfile", dest = "outfile", type = str, required = True,
                        help="Output name for multiple sequence alignment")
    parser.add_argument("-f1", "--current_format", dest = "current_format", type = str, required = False, default = 'msf',
                        help="Current file format, possible https://biopython.org/wiki/AlignIO. Default=msf ")
    parser.add_argument("-f2", "--new_format", dest = "new_format", type = str, required = False, default = 'fasta',
                        help="Desired file format, possible https://biopython.org/wiki/AlignIO. Default=fasta ")


    args = parser.parse_args()
    return(args)
 
if __name__ == '__main__':


    args = get_args()
    infile = args.infile
    outfile  = args.outfile
    current_format = args.current_format
    new_format = args.new_format
    print("Converting file {} from {} to {}".format(infile, current_format, new_format))
    convert_to_fasta(infile, outfile, current_format, new_format)
    print("Converted file saved at {}".format(outfile))


