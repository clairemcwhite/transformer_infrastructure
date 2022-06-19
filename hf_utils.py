from Bio import SeqIO

import torch

import faiss
import numpy as np
from sklearn.preprocessing import normalize

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


### AA class
class AA:
   def __init__(self):
       self.seqnum = ""
       self.seqname = ""
       self.seqindex = ""
       self.seqpos = ""
       self.seqaa = ""
       self.index = ""
       self.clustid = ""
   #__str__ and __repr__ are for pretty #printing
   def __str__(self):
        return("{}-{}-{}".format(self.seqnum, self.seqpos, self.seqaa))
   def __repr__(self):
    return str(self)


### Sequence formatting

def format_sequence(sequence, add_spaces = True):

   if add_spaces == False:
       seq_spaced = sequence

   else:
       seq_spaced = " ".join(list(sequence))

   return seq_spaced

def parse_fasta(fasta_path, sequence_out = "", add_spaces = True):

   sequences = []
   with open(sequence_out, "w") as outfile:

       for record in SeqIO.parse(fasta_path, "fasta"):
            seq_spaced = format_sequence(record.seq, add_spaces)
            
            if sequence_out:
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



### Similarity
def build_index_flat(hidden_states, scoretype = "cosinesim", index = None,  normalize_l2 = True, return_norm = False):


    if not index:
        d = hidden_states.shape[1]
        if scoretype == "euclidean":
           # Not working right, do tests before using
           #index = faiss.index_factory(d, "Flat", faiss.METRIC_L2)
           index = faiss.index_factory(d, "Flat")
           #index = faiss.IndexFlatL2(d)
        else:
            index = faiss.index_factory(d, "Flat", faiss.METRIC_INNER_PRODUCT)

    if normalize_l2 == True:
        #hidden_states, norm = normalize(hidden_states, norm='l2', axis=1, copy=True, return_norm=True)
        #print(hidden_states.shape)
        faiss.normalize_L2(hidden_states)
        #xtsq = hidden_states ** 2
        #print(np.sum(xtsq, axis = 1))
        #print(np.sum(xtsq, axis = 1).shape) 
        #xtsq_norm = hidden_states_norm ** 2
        #print(np.sum(xtsq_norm, axis = 1))
        #print(np.sum(xtsq_norm, axis = 1).shape) 

    index.add(hidden_states)

    if return_norm == True:
       return(index, norm)

    if return_norm == False:
       return(index)

def build_index_voronoi(hidden_states, seqlens):

    d = hidden_states.shape[1]

    nlist = int(max(seqlens)/20)
    nprobe = 3 # 
    print("nlist = ", nlist)
    print("nprobe = ", nprobe)

    ##PQ(quantizer, d, nlist, m, bits)
      #quantizer = faiss.IndexHNSWFlat(d, 32)
    quantizer = faiss.IndexFlatL2(d)  # the other index
    #index = faiss.IndexIVFPQ(quantizer, d, nlist, 8,8, faiss.METRIC_INNER_PRODUCT)
    index = faiss.IndexIVFFlat(quantizer, d, nlist, faiss.METRIC_INNER_PRODUCT)
    assert not index.is_trained
    
    faiss.normalize_L2(hidden_states)
    index.train(hidden_states)
    assert index.is_trained
    index.add(hidden_states)
    index.nprobe = nprobe

    return(index)


def get_knn(index, hidden_states, k):
    distance, index = index.search(hidden_states, k)
    return(distance, index)


def compare_hidden_states(hidden_states_a, hidden_states_b, k):
    '''
    Get cosine similarity(?) between sets of embeddings
    If k, limit to k closest matches for each embedding
    Returns distances and indices
    About 10x faster than sentence_transformer util.pytorch_cos_sim
    '''
    #if hidden_states_a == hidden_states_b:
    #   all_hidden_states = hidden_states_a 
       

    index = build_index(all_hidden_states)

    logging.info("start comparison")
    if k:
        # Return k closests seqs. Add one, because best hit will be self
        distance, index = index.search(hidden_states, k + 1)
    else:
        num_seqs = hidden_states.shape[0]
        D, I = index.search(hidden_states, num_seqs)

    return(D, I)

#Mean Pooling - Take attention mask into account for correct averaging
def mean_pooling(model_output, attention_mask):
    '''
    This function is from sentence_transformers
    #https://www.sbert.net/examples/applications/computing-embeddings/README.html
    '''
    token_embeddings = model_output[0] #First element of model_output contains all token embeddings
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, 1)
    sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
    return sum_embeddings / sum_mask


def kmeans_hidden_states_aas(hidden_states_list, k):
    '''
    Return kmeans clusters for set of embeddings 
    D = distance to centroid
    I = index of cluster
    '''
    hidden_states = np.concatenate([hidden_states_list])
    d = hidden_states.shape[1]
    kmeans = faiss.Kmeans(d = d, k = k, niter = 10)
    kmeans.train(hidden_states)
    D, I = kmeans.index.search(hidden_states, 1)
    return(D, I)

    # Do this per sequence, will get seq-specific path w/o keeping track 
    #for prot in hidden_states:
    #    print("prolurm-6359369.out")
    #    print(prot)
    #    D, I = kmeans.index.search([prot], 1)
    #    print(I)
    #return(D, I)


def kmeans_hidden_states(hidden_states, k):
    '''
    Return kmeans clusters for set of embeddings 
    D = distance to centroid
    I = index of cluster
    '''
    #print(hidden_states)
    print(hidden_states.shape)
    d = hidden_states.shape[1]
    kmeans = faiss.Kmeans(d = d, k = k, niter = 10)
    kmeans.train(hidden_states)
   
    # Do this per sequence, will get seq-specific path w/o keeping track

    D, I = kmeans.index.search(hidden_states, 1)
    return(D, I)

### AA relationships ###
def get_hidden_states(seqs, model, tokenizer, layers, return_sentence = False):
    
    # For a list of sequences, get list of hidden states
    encoded = tokenizer.batch_encode_plus(seqs, return_tensors="pt", padding=True)
    with torch.no_grad():
        output = model(**encoded)

    # Get all hidden states
    hidden_states = output.hidden_states

  
   
    #BramVanroy  https://github.com/huggingface/transformers/issues/1328#issuecomment-534956703
    #Concatenate final for hidden states into long vector
    pooled_output = torch.cat(tuple([hidden_states[i] for i in layers]), dim=-1)

    if return_sentence == True:
       sentence_embeddings = mean_pooling(output, encoded['attention_mask'])
       return(pooled_output, sentence_embeddings)
    # If output shape is [len(seqs), 3, x], make sure there are spaces between each amino acid
    # 3 - CLS,seq,END
    else:
        return pooled_output

#
