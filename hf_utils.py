from Bio import SeqIO
def parse_fasta(fasta_path, sequence_out, no_spaces):

   sequences = []


   with open(sequence_out, "w") as outfile:

       for record in SeqIO.parse(fasta_path, "fasta"):
            #print("%s %i" % (record.id, record.seq))
            if no_spaces:
                seq_spaced = record.seq
            else:
                seq_spaced =  " ".join(record.seq)
            outstring = "{},{}\n".format(record.id, seq_spaced)
            outfile.write(outstring)
            sequences.append(seq_spaced)
   
   return(sequences)


