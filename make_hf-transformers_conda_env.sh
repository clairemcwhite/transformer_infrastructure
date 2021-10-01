module load anaconda3/2020.11

#pytorch has to be from pytorch channel, otherwise CUDA available is False
conda create --name hf-transformers -c conda-forge -c pytorch transformers pytorch::pytorch pandas numpy biopython faiss seqeval cudatoolkit=10.2 python-igraph matplotlib sentence-transformers

conda activate hf-transformers


# With extras
#conda create --name hf-transformers -c conda-forge -c pytorch transformers pytorch::pytorch sentence-transformers pandas requests tqdm numpy seqeval cudatoolkit=10.2  biopython faiss networkx iteration_utilities python-igraph protobuf #jonga pygraphviz

