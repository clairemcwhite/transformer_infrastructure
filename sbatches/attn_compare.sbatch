#!/bin/bash
#SBATCH --job-name=index_sam       # create a short name for your job
#SBATCH --nodes=1                # node count
#SBATCH --ntasks=1               # total number of tasks across all nodes
#SBATCH --cpus-per-task=1        # cpu-cores per task (>1 if multi-threaded tasks)
#SBATCH --mem-per-cpu=64G         # memory per cpu-core (4G is default)
#SBATCH --time=144:00:00          # total run time limit (HH:MM:SS)
#SBATCH --gres=gpu:1             # number of gpus per node
#SBATCH --mail-type=begin        # send email when job begins
#SBATCH --mail-type=end          # send email when job ends
#SBATCH --mail-user=cmcwhite@princeton.edu

module purge
module load anaconda3/2020.11
module load cudatoolkit

conda activate hf-transformers


nvidia-smi

python -m torch.utils.collect_env

python /home/cmcwhite/transformer_infrastructure/dms.py -f /scratch/gpfs/vc8200/capra_bioinf08_data/fastas/MTHR_HUMAN.fasta -mo /scratch/gpfs/vc8200/prot_bert_bfd -o /scratch/gpfs/cmcwhite/MTHR_HUMAN.attn_compare_jaccard.gpu.csv
