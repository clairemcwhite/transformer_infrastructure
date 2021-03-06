module load anaconda3/2020.11

#pytorch has to be from pytorch channel, otherwise CUDA available is False
#conda create --name hf-transformers -c conda-forge -c pytorch transformers pytorch::pytorch pandas numpy biopython faiss seqeval cudatoolkit=10.2 python-igraph matplotlib sentence-transformers  pytorch-lightning  torchmetrics

conda create --name hf-transformers -c conda-forge -c pytorch transformers pytorch::pytorch pandas numpy=1.21.2 biopython faiss seqeval cudatoolkit=11.1 python-igraph matplotlib  pytorch-lightning torchmetrics huggingface_hub numba

conda create --name hf-transformers2 -c conda-forge -c pytorch transformers pytorch::pytorch pandas numpy=1.21.2 biopython faiss seqeval cudatoolkit=11.1 python-igraph matplotlib  pytorch-lightning torchmetrics huggingface_hub numba rpy2



conda create --prefix /scratch/gpfs/cmcwhite/base_utils -c conda-forge numpy=1.21.2 pandas biopython rpy2 numba matplotlib scikit-learn r-igraph python-igraph numba r-dendextend r-dplyr

#conda create --name hf-transformers -c conda-forge -c pytorch transformers pytorch::pytorch pandas numpy=1.21.2 biopython faiss seqeval cudatoolkit=11.1 python-igraph matplotlib sentence-transformers  pytorch-lightning  torchmetrics huggingface_hub numba

#conda install pytorch torchvision torchaudio cudatoolkit=11.1 -c pytorch -c nvidia


conda activate hf-transformers


# With extras
#conda create --name hf-transformers -c conda-forge -c pytorch transformers pytorch::pytorch sentence-transformers pandas requests tqdm numpy seqeval cudatoolkit=10.2  biopython faiss networkx iteration_utilities python-igraph protobuf #jonga pygraphviz

