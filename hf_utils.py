from Bio import SeqIO
import torch

### Sequence formatting
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


### Classification ###
def assemble_SS3_dataset(seqs, labels, tag2id, tokenizer, logging):

    labels_encodings = encode_tags(labels, tag2id)
    logging.info("labels encoded")


    seqs_encodings = seq_tokenizer(seqs, is_split_into_words=True, return_offsets_mapping=True, truncation=True, padding=True)

    _ = seqs_encodings.pop("offset_mapping")
    logging.info("offset_mapping popped")


    dataset = SS3Dataset(seqs_encodings, labels_encodings)
    logging.info("SS3 dataset constructed")

    return(dataset)


def get_sequencelabel_tags(labels):
    unique_tags = set(labels)
    unique_tags  = sorted(list(unique_tags))  # make the order of the labels unchanged
    tag2id = {tag: id for id, tag in enumerate(unique_tags)}
    id2tag = {id: tag for tag, id in tag2id.items()}
    return(tag2d, id2tag)


### AA relationships ###
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


