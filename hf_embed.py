#from sentence_transformers import SentenceTransformer, models
from transformers import AutoTokenizer, AutoModel
#from transformer_infrastructure.pca_embeddings import xxx
import torch
import torch.nn as nn
from Bio import SeqIO
import pickle
import argparse
from torch.utils.data import Dataset, DataLoader
import numpy as np
import numba as nb
import awkward as ak
import time

'''
Get pickle of embeddings for a fasta of protein sequences
with a huggingface transformer model

Can return aa-level, sequence-level, or both

Optional to do PCA on embeddings prior to saving, or use pre-trained PCA matrix on them. 

#### Default pickle shapes
pickle['aa_embeddings']:  (numseqs x longest seqlength x (1024 * numlayers)
pickle['sequence_embeddings']: (numseqs x 1024)


### --use_ragged_arrays
Amino acid embeddings are by default save in a numpy array of dimensions 
   pickle['aa_embeddings']:  (numseqs x longest seqlength x (1024 * numlayers)

The package `awkward` allows saving of arrays of different lengths. 
If all sequences are around the same lengths, there's not much different
However, one long sequence can greatly increase file sizes.
For a set of 50 ~300aa sequence + one 5000aa sequence, there's a tenfold difference in file size. 

3.9G test_np.pkl
369M test_awkward.pkl

### PCA
Another route to smaller file size is training a PCA transform to reduced dimensionality.
It can either be applied to sequence or amino acid embeddings. 

Previously trained PCA matrices can be used as well.



#### Example command
 python transformer_infrastructure/hf_embed.py -m /scratch/gpfs/cmcwhite/prot_bert_bfd/ -f tester.fasta -o test.pkl

#### To load a pre-computed embedding:

 with open("embeddings.pkl", "rb") as f:
     cache_data = pickle.load(f)
     sequence_embeddings = cache_data['sequence_embeddings']
     aa_embeddings = cache_data['aa_embeddings']


#### extra_padding argument
Adding 5 X's to the beginning and end of each sequence seems to improve embeddings
I'd be interested in feedback with this parameter set to True or False

#### To download a huggingface model locally:

from transformers import AutoModel, AutoTokenizer

sourcename = "Rostlab/prot_bert_bfd"
modelname = "prot_bert_bfd"
outdir = "/scratch/gpfs/cmcwhite/hfmodels/" + modelname

tokenizer = AutoTokenizer.from_pretrained(sourcename)
tokenizer.save_pretrained(outdir)
model = AutoModel.from_pretrained(sourcename)
model.save_pretrained(outdir)

#### Minimal anaconda environment
conda create --name hf-transformers -c conda-forge -c pytorch transformers pytorch::pytorch numpy biopython

Claire D. McWhite
7/8/20
'''

def get_embed_args():

    parser = argparse.ArgumentParser()
    parser.add_argument("-m", "--model", dest = "model_path", type = str, required = True,
                        help="Model directory Ex. /path/to/model_dir")
    parser.add_argument("-f", "--fasta", dest = "fasta_path", type = str, required = True,
                        help="path to a fasta of protein sequences")
    parser.add_argument("-o", "--outpickle", dest = "pkl_out", type = str, required = True,
                        help="output .pkl filename")
    parser.add_argument("-s", "--sequence_embeddings", dest = "get_sequence_embeddings", type = bool, default = True,
                        help="Whether to get sequence embeddings, default: True")
    parser.add_argument("-a", "--aa_embeddings", dest = "get_aa_embeddings", type = bool, default = True,
                        help="Whether to get amino-acid embeddings, default: True")
    parser.add_argument("-p", "--extra_padding", dest = "extra_padding", type = bool, default = True,
                        help="Add if using unaligned sequence fragments (to reduce first and last character effects). Potentially not needed for sets of complete sequences or domains that start at the same character, default: True")
    parser.add_argument("-t", "--truncate", dest = "truncate", type = int, required = False,
                        help= "Optional: Truncate all sequences to this length")
    parser.add_argument("-d", "--pca_target_dim", dest = "target_dim", type = int, required = False,
                        help= "Optional: Run a PCA on all embeddings with target n dimensions prior to saving")
    parser.add_argument("-pm", "--pretrained_pcamatrix_pkl", dest = "pretrained_pcamatrix", type = str, required = False,
                        help= "Optional: Use a pretrained PCA matrix to reduce dimensions of embeddings (pickle file with objects pcamatrix and bias")
    parser.add_argument("-r", "--use_ragged_arrays", dest = "ragged_arrays", action = "store_true", required = False,
                        help= "Optional: Use package 'awkward' to save ragged arrays fo amino acid embeddings")

    args = parser.parse_args()
    
    return(args)

def parse_fasta_for_embed(fasta_path, truncate = None, extra_padding = True):
   ''' 
   Load a fasta of protein sequences and
     add a space between each amino acid in sequence (needed to compute embeddings)
   Takes: 
       str: Path of the fasta file
       truncate (int): Optional length to truncate all sequences to
       extra_padding (bool): Optional to add padding to each sequence to avoid start/end of sequence effects
                             Useful for when all sequences don't start/end at "same" amino acid. 
 
   Returns: 
       [ids], [sequences], [sequences with spaces and any extra padding] 
   '''
   sequences = []
   sequences_spaced = []
   ids = []
   for record in SeqIO.parse(fasta_path, "fasta"):
       
       seq = record.seq

       #if extra_padding == True: 
       #    seq = "XXXXX{}XXXXX".format(seq)
       if truncate:
          print("truncating to {}".format(truncate))
          seq = seq[0:truncate]
 
       seq_spaced =  " ".join(seq)
       # 5 X's seems to be a good amount of neutral padding
       if extra_padding == True:
            padding_aa = " X" * 5
            padding_left = padding_aa.strip(" ")
    
            # To do: Figure out why embedding are better with removed space between last X and first AA?
            seq_spaced = padding_left + seq_spaced  
            seq_spaced = seq_spaced + padding_aa
  
       ids.append(record.id)
       sequences.append(seq)
       sequences_spaced.append(seq_spaced)
   return(ids, sequences, sequences_spaced)


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
    If adding padding, it's usually 5. 

    Takes: 
       model_output: From sequence encoding
       layers (list of ints): By default, pool final four layers of model
       padding (int): If padding was added, remove embeddings corresponding to extra padding before returning 

    Return shape (numseqs x longest seqlength x (1024 * numlayers)

    Note: If the output shape of this function is [len(seqs), 3, x], make sure there are spaces between each amino acid
    The "3" corresponds to CLS,seq,END 
     '''
    
    # Get all hidden states
    hidden_states = model_output.hidden_states
    # Concatenate hidden states into long vector
    aa_embeddings = torch.cat(tuple([hidden_states[i] for i in layers]), dim=-1)
    if padding:
        aa_embeddings = aa_embeddings[:,padding:-padding,:]
        print('aa_embeddings.shape: {}'.format(aa_embeddings.shape))
    return(aa_embeddings, aa_embeddings.shape)


def retrieve_sequence_embeddings(model_output, encoded):
    ''' 
    Get a sequence embedding by taking the mean of all amino acid embeddings
    Return shape (numseqs x 1024)
    '''
    sentence_embeddings = mean_pooling(model_output, encoded['attention_mask'])
    print('sentence_embeddings.shape: {}'.format(sentence_embeddings.shape))
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

# ? 
class Collate:
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer

    def __call__(self, batch):
        encoding = self.tokenizer.batch_encode_plus(
            list(batch),
            return_tensors  = 'pt', 
            padding = True
            
        )
        return(encoding)

# ?
# Start trying to reduce these
class ListDataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __getitem__(self, index):
        return self.data[index]

    def __len__(self):
        return len(self.data)


@nb.jit
def make_direct(input, lengths, builder):
    '''
    Allows trimming ragged array to the actual sequence lengths
    No unnecessary embedding vectors saved
    From jpivarski
    https://github.com/scikit-hep/awkward-1.0/issues/480#issuecomment-703740986
    ''' 

    for i in range(len(lengths)):
         builder.append(input[i][:lengths[i]])
    return builder


def get_embeddings(seqs, model_path, seqlens, get_sequence_embeddings = True, get_aa_embeddings = True, layers = [-4, -3, -2, -1], padding = 5, ragged_arrays = False):
    '''
    Encode sequences with a transformer model

    Takes:
       model_path (str): Path to a particular transformer model
                         ex. "prot_bert_bfd"
       sequences (list): List of sequences with a space between each amino acid.  
                         ex ["M E T", "S E Q"]
 
   '''

    ak.numba.register()
    print("CUDA available?", torch.cuda.is_available())

    model, tokenizer = load_model(model_path)



    aa_shapes = [] 
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("device", device) 
    device_ids =list(range(0, torch.cuda.device_count()))
    print("device_ids", device_ids)
    if torch.cuda.device_count() > 1:
       print("Let's use", torch.cuda.device_count(), "GPUs!")
       #model = nn.DataParallel(model)
       model = nn.DataParallel(model, device_ids=device_ids).cuda()
       #model.to(device_ids) 

    else:
       model = model.to(device)

    # Definitly needs to be batched, otherwise GPU memory errors
    batch_size = torch.cuda.device_count()

    collate = Collate(tokenizer=tokenizer)

    data_loader = DataLoader(dataset=ListDataset(seqs),
                      batch_size=batch_size,
                      shuffle=False,
                      collate_fn=collate,
                      pin_memory=False)
    start = time.time()

    # Need to concatate output of each chunk
    sentence_array_list = []
    aa_array_list = []
#
#    aa_builder = ak.ArrayBuilder()

    # Using awkward arrays for amino acids because sequence lengths are variable
    # If 10,000 long sequence was used, all sequences would be padded to 10,000
    # Awkward arrays allow concatenating ragged arrays
    count = 0
    maxlen = max(seqlens)
    with torch.no_grad():

        # For each chunk of data
        for data in data_loader:
            input = data.to(device)
            # DataParallel model splits data to the different devices and gathers back
            # nvidia-smi shows 4 active devices (when there are 4 GPUs)
            model_output = model(**input)
 
            # Do final processing here. 
            if True:
                sentence_embeddings = mean_pooling(model_output, data['attention_mask'])
                sentence_embeddings = sentence_embeddings.to('cpu')
                sentence_array_list.append(sentence_embeddings)
 
            if True:
                aa_embeddings, aa_shape = retrieve_aa_embeddings(model_output, layers = layers, padding = padding)
                aa_embeddings = aa_embeddings.to('cpu')
                aa_embeddings = np.array(aa_embeddings)
                # Trim each down to just its sequence length
                if ragged_arrays == True:
                    for j in range(len(aa_embeddings)):
                         seqindex = (20 * count) + j
                         print(count, j, seqindex, seqlens[seqindex])
                         print(aa_embeddings[j])
                         aa_embed_trunc = aa_embeddings[j][:seqlens[seqindex], :]
                    
                         aa_embed_ak = ak.Array(aa_embed_trunc)
                         aa_array_list.append(aa_embed_ak)
               
                else:
                    # If not using ragged arrays, must pad to same dim as longest sequence
                    print(maxlen - (aa_embeddings.shape[1] - 1))
                    npad = ((0,0), (0, maxlen - (aa_embeddings.shape[1] - 1)), (0,0))
                    aa_embeddings = np.pad(aa_embeddings, npad)
                    aa_array_list.append(aa_embeddings)

        end = time.time() 
        print("Total time to embed = {}".format(end - start))
  
        count = count + 1
        ak_aa =np.concatenate(aa_array_list)
 
        
        lengths = np.array(seqlens)

        # Trim each aa embedding to only the aa's in the original sequences
        #ak_aa = make_direct(ak_aa, lengths, ak.ArrayBuilder()).snapshot()

        embedding_dict = {}
        embedding_dict['sequence_embeddings'] = np.concatenate(sentence_array_list)
        embedding_dict['aa_embeddings'] = ak_aa


        return(embedding_dict)

    

if __name__ == "__main__":

    args = get_embed_args()
    #print(args.fasta_path, args.extra_padding)
    ids, sequences, sequences_spaced = parse_fasta_for_embed(fasta_path = args.fasta_path, 
                                                             truncate = args.truncate, 
                                                             extra_padding = args.extra_padding)

    seqlens = [len(x) for x in sequences]
 
    if args.extra_padding:
       padding = 5

    else: 
       padding = 0
    #print(sequences_spaced)
    embedding_dict = get_embeddings(sequences_spaced, 
                                    args.model_path, 
                                    get_sequence_embeddings = args.get_sequence_embeddings, 
                                    get_aa_embeddings = args.get_aa_embeddings, 
                                    padding = padding, 
                                    seqlens = seqlens,
                                    ragged_arrays = args.ragged_arrays)
  
   
    #Store sequences & embeddings on disk
    with open(args.pkl_out, "wb") as fOut:
       pickle.dump(embedding_dict, fOut, protocol=pickle.HIGHEST_PROTOCOL)

    pkl_log = "{}.description".format(args.pkl_out)
    with open(pkl_log, "w") as pOut:
        pOut.write("Object {} dimensions: {}\n".format('sequence_embeddings', embedding_dict['sequence_embeddings'].shape))

        if args.ragged_arrays == True:
            pOut.write("Object {} dimensions: {}\n".format('aa_embeddings', embedding_dict['aa_embeddings'].type))
            pOut.write("aa_embeddings are an `awkward` arrays with dimensions\n")   
            pOut.write("aa_embeddings are an `awkward` arrays with dimensions\n")   
    
            pOut.write("{}".format(ak.num(embedding_dict['aa_embeddings'], axis=0)))
            pOut.write("{}".format(ak.num(embedding_dict['aa_embeddings'], axis=1)))
        # Else it's a square numpy array
        else:
            pOut.write("Object {} dimensions: {}\n".format('aa_embeddings', embedding_dict['aa_embeddings'].shape))
           

        pOut.write("Contains sequences:\n")
        for x in ids:
          pOut.write("{}\n".format(x))




# Would like to use SentenceTransformers GPU parallelization, but only currently can do sequence embeddings. Need to do adapt it
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
   


