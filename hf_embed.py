#from sentence_transformers import SentenceTransformer, models
from transformers import AutoTokenizer, AutoModel
import torch
from Bio import SeqIO
import pickle
import argparse

'''
Get pickle(s) of embeddings for a fasta of protein sequences
with a huggingface transformer model

Can return aa-level, sequence-level, or both

To load a pre-computed embedding:

with open("embeddings.pkl", "rb") as f:
    cache_data = pickle.load(f)
    sequence_embeddings = cache_data['sequence_embeddings']
    aa_embeddings = cache_data['aa_embeddings']

Claire D. McWhite
'''


def parse_fasta_for_embed(fasta_path, truncate = "", extra_padding = False):
   ''' 
   Load a fasta of protein sequences and
     add a space between each amino acid in sequence (needed to compute embeddings)
   Takes: 
       str: Path of the fasta file
       truncate (int): Optional length to truncate all sequences to
       extra_padding (bool): Optional to add padding to each sequence to avoid start/end of sequence effects
                             Useful for when all sequences don't start/end at "same" amino acid 
 
   Returns: 
       list of sequences  
   '''
   sequences = []

   for record in SeqIO.parse(fasta_path, "fasta"):
       if truncate:
           seq = seq[0:truncate]
       else:
           seq = record.seq

       if extra_padding == True: 
           seq = "XXXXX{}XXXXX".format(seq)

       seq_spaced = seq_spaced =  " ".join(seq)
       sequences.append(seq_spaced)

   return(sequences)


def mean_pooling(model_output, attention_mask):
    '''
    Mean Pooling - Take attention mask into account for correct averaging

    This function is from sentence_transformers
    #https://www.sbert.net/examples/applications/computing-embeddings/README.html
    '''
    token_embeddings = model_output[0] #First element of model_output contains all token embeddings
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, 1)
    sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
    return sum_embeddings / sum_mask

def retrieve_aa_embeddings(model_output, layers = [-4, -3, -2, -1], padding = ""):
    '''
    Get the amino acid embeddings for each sequences
    Pool layers by concatenating selection of layers
    Return shape: (numseqs, length of longest sequence, 1024*numlayers)

    Takes: 
       model_output: From sequence encoding
       layers (list of ints): By default, pool final four layers of model
       padding (int): If padding was added, remove embeddings corresponding to extra padding before returning 

    Note: If the output shape of this function is [len(seqs), 3, x], make sure there are spaces between each amino acid
    The "3" corresponds to CLS,seq,END 
     '''
    
    # Get all hidden states
    hidden_states = model_output.hidden_states
    # Concatenate hidden states into long vector
    aa_embeddings = torch.cat(tuple([hidden_states[i] for i in layers]), dim=-1)
    if padding:
        aa_embeddings = hstates_list[:,padding:-padding,:]

    return(aa_embeddings)


def retrieve_sequence_embeddings(model_output, encoded):
    ''' 
    Get a sequence embedding by taking the mean of all amino acid embeddings
    Return shape (numseqs x 1024)
    '''
    sentence_embeddings = mean_pooling(model_output, encoded['attention_mask'])
    return(sentence_embeddings)


def load_model(model_path):
    '''
    Takes path to huggingface model directory
    Returns the model and the tokenizer
    '''
    print("load tokenizer")
    tokenizer = AutoTokenizer.from_pretrained(model_path)

    print("load model")
    model = AutoModel.from_pretrained(model_path, output_hidden_states=True)

    return(model, tokenizer)


def get_encodings(seqs, model_path):
    '''
    Encode sequences with a model and tokenizer

    Takes:
       model_path (str): Path to a particular transformer model
                         ex. "prot_bert_bfd"
       sequences (list): List of sequences with a space between each acids.  
                         ex ["M E T", "S E Q"]
 
   '''
 
    model, tokenizer = load_model(model_path)
    
    encoded = tokenizer.batch_encode_plus(seqs, return_tensors="pt", padding=True)
    with torch.no_grad():
        model_output = model(**encoded)

    return(model_output, encoded)



def embed_sequences(seqs, model_path, get_sequence_embeddings = True, get_aa_embeddings = True, layers = [-4, -3, -2, 1], padding = ""):
    '''
    Get a pkl of embeddings for a list of sequences using a particular model
    Embeddings

      pkl_out (str) : Filename of output pickle of embeddings
 
    '''
    model_output, encoded = get_encodings(seqs, model_path)

    embedding_dict = {}
    if get_sequence_embeddings:
         sequence_embeddings = retrieve_sequence_embeddings(model_output, encoded)
         embedding_dict['sequence_embeddings'] = sequence_embeddings

    if get_aa_embeddings:
         aa_embeddings = retrieve_aa_embeddings(model_output, layers = layers, padding = padding)
         embedding_dict['aa_embeddings'] = aa_embeddings
        
   
    return(embedding_dict)




    

def get_embed_args():

    parser = argparse.ArgumentParser()
    parser.add_argument("-m", "--model", dest = "model_path", type = str, required = True,
                        help="Model directory Ex. /path/to/model_dir")
    parser.add_argument("-f", "--fasta", dest = "fasta_path", type = str, required = True,
                        help="path to a fasta of protein sequences")
    parser.add_argument("-op", "--outpickle", dest = "pkl_out", type = str, required = True,
                        help="output .pkl filename")
    parser.add_argument("-s", "--sequence_embeddings", dest = "get_sequence_embeddings", type = bool, default = True,
                        help="Whether to get sequence embeddings, default: True")
    parser.add_argument("-a", "--aa_embeddings", dest = "get_aa_embeddings", type = bool, default = True,
                        help="Whether to get amino-acid embeddings, default: True")
    parser.add_argument("-p", "--extra_padding", dest = "extra_padding", type = bool, default = False,
                        help="Add if using unaligned sequence fragments (to reduce first and last character effects). Not needed for sets of complete sequences or domains.")


    args = parser.parse_args()
    return(args)

if __name__ == "__main__":

    args = get_embed_args()
    sequences = parse_fasta_for_embed(args.fasta_path, args.extra_padding)
    embedding_dict = embed_sequences(sequences, 
                                    args.model_path, 
                                    get_sequence_embeddings = args.get_sequence_embeddings, 
                                    get_aa_embeddings = args.get_aa_embeddings, 
                                    padding = args.extra_padding)
  
   
    #Store sequences & embeddings on disk
    with open(args.pkl_out, "wb") as fOut:
       pickle.dump(embedding_dict, fOut, protocol=pickle.HIGHEST_PROTOCOL)

    pkl_log = "{}.description".format(args.pkl_out)
    with open(pkl_log, "w") as pOut:
        for key, value in embedding_dict.items():
             print(key)
             print(value)
             pOut.write("Object {} dimensions: {}\n".format(key, value.shape))



# Would like to use SentenceTransformers GPU parallelization, but only currently can do sequence embeddings
#def embed_sequences(model_path, sequences, extra_padding,  pkl_out):
#    '''
#    
#    Get a pkl of embeddings for a list of sequences using a particular model
#    Embeddings will have shape xx
#
#    Takes:
#       model_path (str): Path to a particular transformer model
#                         ex. "prot_bert_bfd"
#       sequences (list): List of sequences with a space between each acids.  
#                         ex ["M E T", "S E Q"]
#       pkl_out (str)   : Filename of output pickle of embeddings
# 
#    '''
#    print("Create word embedding model")
#    word_embedding_model = models.Transformer(model_path)
#
#    # Default pooling strategy
#    pooling_model = models.Pooling(word_embedding_model.get_word_embedding_dimension())
#
#    model = SentenceTransformer(modules=[word_embedding_model, pooling_model])
#    print("SentenceTransformer model created")
#    
#    pool = model.start_multi_process_pool()
#
#    # Compute the embeddings using the multi-process pool
#    # about 1.5 hours to this step with 4 GPU and 1.4 million sequences 
#    print("Computing embeddings")
#
#    e = model.encode(sequences, output_value = 'token_embeddings')
#    print(e)
#
#    embeddings = model.encode_multi_process(sequences, pool, output_value = 'token_embeddings')
#
#    print("Embeddings computed. Shape:", embeddings.shape)
#
#    #Optional: Stop the proccesses in the pool
#    model.stop_multi_process_pool(pool)
#
#    return(embeddings)    
   


