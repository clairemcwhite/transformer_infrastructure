#!/bin/bash
#SBATCH --job-name=hf_clusterseqs        # create a short name for your job
#SBATCH --nodes=1                # node count
#SBATCH --ntasks=1               # total number of tasks across all nodes
#SBATCH --cpus-per-task=1        # cpu-cores per task (>1 if multi-threaded tasks)
#SBATCH --mem-per-cpu=32G         # memory per cpu-core (4G is default)
#SBATCH --gres=gpu:1             # number of gpus per node
#SBATCH --time=1:00:00          # total run time limit (HH:MM:SS)
#SBATCH --mail-type=begin        # send email when job begins
#SBATCH --mail-type=end          # send email when job ends
#SBATCH --mail-user=cmcwhite@princeton.edu

module purge
module load anaconda3/2020.11
module load cudatoolkit

conda activate hf-transformers


nvidia-smi

python -m torch.utils.collect_env

python transformer_infrastructure/hf_cluster.py 

#-m /home/cmcwhite/prot_bert_bfd -f /scratch/gpfs/cmcwhite/qfo_2020/qfo_2020.fasta -op /scratch/gpfs/cmcwhite/qfo_2020/qfo_2020.pkl -os /scratch/gpfs/cmcwhite/qfo_2020/qfo_2020.csv


