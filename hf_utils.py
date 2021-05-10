from Bio import SeqIO
import torch
def format_sequence(sequence, no_spaces):
   if no_spaces:
       seq_spaced = sequence
   else:
       seq_spaced =  " ".join(sequence)

   return seq_spaced


def get_hidden_states(seqs, model, tokenizer, layers):
    # For a list of sequences, get list of hidden states
    encoded = tokenizer.batch_encode_plus(seqs, return_tensors="pt", padding=True)
    with torch.no_grad():
        output = model(**encoded)

    # Get all hidden states
    hidden_states = output.hidden_states
    #BramVanroy  https://github.com/huggingface/transformers/issues/1328#issuecomment-534956703
    #Concatenate final for hidden states into long vector
    pooled_output = torch.cat(tuple([hidden_states[i] for i in [-4, -3, -2, -1]]), dim=-1)


    return pooled_output

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


