from transformer_notebooks.hf_utils import parse_fasta
from sentence_transformers import SentenceTransformer, models

from Bio import SeqIO
import pickle
import argparse

def embed_sequences(model_path, sequences, pkl_out, pre_embedded):

    
    if pre_embedded == True:
        model = SentenceTransformer(model_path)
        print("model loaded")
    else:
        word_embedding_model = models.Transformer(model_path)
        # Default pooling strategy
        pooling_model = models.Pooling(word_embedding_model.get_word_embedding_dimension())

        model = SentenceTransformer(modules=[word_embedding_model, pooling_model])
        print("SentenceTransformer model created")
    
    #embeddings = model.encode(sequences)

    pool = model.start_multi_process_pool()

    #Compute the embeddings using the multi-process pool
    embeddings = model.encode_multi_process(sequences, pool)
    # about 1.5 hours to this step with 4 GPU and 1.4 million sequences 
    print("Embeddings computed. Shape:", embeddings.shape)

    #Optional: Stop the proccesses in the pool
    model.stop_multi_process_pool(pool)

    if pkl_out:
    #Store sequences & embeddings on disk
        with open(pkl_out, "wb") as fOut:
            pickle.dump({'sequences': sequences, 'embeddings': embeddings}, fOut, protocol=pickle.HIGHEST_PROTOCOL)
    return(embeddings)    
    #Load sequences & embeddings from disc
    #with open('embeddings.pkl', "rb") as fIn:
    #    stored_data = pickle.load(fIn)
    #    stored_sequences = stored_data['sequences']
    #    stored_embeddings = stored_data['embeddings']
    

def get_embed_args():

    parser = argparse.ArgumentParser()
    parser.add_argument("-m", "--model", dest = "model_path", type = str, required = True,
                        help="Model directory Ex. /path/to/model_dir")
    parser.add_argument("-f", "--fasta", dest = "fasta_path", type = str, required = True,
                        help="fasta of protein sequences")

    parser.add_argument("-n", "--dont_add_spaces" , action = "store_true",
                        help="Flag if sequences already have spaces")
    parser.add_argument("-p", "--pre_embedded" , action = "store_true",
                        help="Flag if model has already been saved as an embedding model")

    parser.add_argument("-op", "--outpickle", dest = "outpickle", type = str, required = True,
                        help="output .pkl filename")
    parser.add_argument("-os", "--outsequences", dest = "outsequences", type = str, required = True,
                        help="output csv for table of identified and spaced out sequences (for conversion after embedding)")


    args = parser.parse_args()
    return(args)

if __name__ == "__main__":
    args = get_embed_args()
    sequences = parse_fasta(args.fasta_path, args.outsequences, args.dont_add_spaces)
    embed_sequences(args.model_path, sequences, args.outpickle, args.pre_embedded)


