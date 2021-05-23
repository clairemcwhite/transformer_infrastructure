module load anaconda3/2020.11
#pytorch has to be from pytorch channel, otherwise CUDA available is False
# Likely drop networkx
conda create --name hf-transformers -c conda-forge -c pytorch transformers pytorch::pytorch sentence-transformers pandas requests tqdm numpy seqeval cudatoolkit=10.2  biopython faiss networkx iteration_utilities igraph
conda activate hf-transformers
