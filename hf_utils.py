from Bio import SeqIO

def format_sequence(sequence, no_spaces):
   if no_spaces:
       seq_spaced = sequence
   else:
       seq_spaced =  " ".join(sequence)

   return seq_spaced

def parse_fasta(fasta_path, sequence_out, no_spaces):

   sequences = []

   with open(sequence_out, "w") as outfile:

       for record in SeqIO.parse(fasta_path, "fasta"):
            #print("%s %i" % (record.id, record.seq))
            seq_spaced = format_sequence(record.seq, no_spaces)
            outstring = "{},{}\n".format(record.id, seq_spaced)
            outfile.write(outstring)
            sequences.append([record.id, record.seq, seq_spaced])
   
   return(sequences)


