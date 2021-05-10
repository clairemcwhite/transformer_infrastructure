module load anaconda3/2020.11
#pytorch has to be from pytorch channel, otherwise CUDA available is False
conda create --name hf-transformers -c conda-forge -c pytorch transformers pytorch::pytorch sentence-transformers pandas requests tqdm numpy seqeval cudatoolkit=10.2  biopython
conda activate hf-transformers
