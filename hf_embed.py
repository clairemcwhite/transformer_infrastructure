from sentence_transformers import SentenceTransformer
from Bio import SeqIO
import pickle
import argparse

def embed_sequences(model_path, sequences, pkl_out):
    model = SentenceTransformer(model_path)
    
    
    #embeddings = model.encode(sequences)

    pool = model.start_multi_process_pool()

    #Compute the embeddings using the multi-process pool
    embeddings = model.encode_multi_process(sequences, pool)
    print("Embeddings computed. Shape:", emb.shape)

    #Optional: Stop the proccesses in the pool
    model.stop_multi_process_pool(pool)

    
    #Store sequences & embeddings on disk
    with open(pkl_out, "wb") as fOut:
        pickle.dump({'sequences': sequences, 'embeddings': embeddings}, fOut, protocol=pickle.HIGHEST_PROTOCOL)
    
    #Load sequences & embeddings from disc
    #with open('embeddings.pkl', "rb") as fIn:
    #    stored_data = pickle.load(fIn)
    #    stored_sequences = stored_data['sequences']
    #    stored_embeddings = stored_data['embeddings']
    
def parse_fasta(fasta_path, sequence_out, no_spaces):

   sequences = []


   with open(sequence_out, "w") as outfile:

       for record in SeqIO.parse(fasta_path, "fasta"):
            #print("%s %i" % (record.id, record.seq))
            if no_spaces:
                seq_spaced = record.seq
            else:
                seq_spaced =  " ".join(record.seq)
            outstring = "%s,%s\n".format(record.id, seq_spaced)
            outfile.write(outstring)
            sequences.append(seq_spaced)
   return(sequences)

def get_embed_args():

    parser = argparse.ArgumentParser()
    parser.add_argument("-m", "--model", dest = "model_path", type = str, required = True,
                        help="Model directory Ex. /path/to/model_dir")
    parser.add_argument("-f", "--fasta", dest = "fasta_path", type = str, required = True,
                        help="fasta of protein sequences")

    parser.add_argument("-n", "--dont_add_spaces" , action = "store_true",
                        help="Flag if sequences already have spaces")
    parser.add_argument("-op", "--outpickle", dest = "outpickle", type = str, required = True,
                        help="output .pkl filename")
    parser.add_argument("-os", "--outsequences", dest = "outsequences", type = str, required = True,
                        help="output csv for table of identified and spaced out sequences (for conversion after embedding)")


    args = parser.parse_args()
    return(args)

if __name__ == "__main__":
    args = get_embed_args()
    sequences = parse_fasta(args.fasta_path, args.outsequences, args.dont_add_spaces)
    embed_sequences(args.model_path, sequences, args.outpickle)


