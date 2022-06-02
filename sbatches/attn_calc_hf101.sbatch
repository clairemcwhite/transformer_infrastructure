#!/bin/bash
#SBATCH --job-name=hf_clusterseqs        # create a short name for your job
#SBATCH --nodes=1                # node count
#SBATCH --ntasks=1               # total number of tasks across all nodes
#SBATCH --cpus-per-task=1        # cpu-cores per task (>1 if multi-threaded tasks)
#SBATCH --mem-per-cpu=16G         # memory per cpu-core (4G is default)
#SBATCH --gres=gpu:1             # number of gpus per node
#SBATCH --time=00:30:00          # total run time limit (HH:MM:SS)
#SBATCH --mail-type=begin        # send email when job begins
#SBATCH --mail-type=end          # send email when job ends
#SBATCH --mail-user=cmcwhite@princeton.edu

module purge
module load anaconda3/2020.11
module load cudatoolkit

conda activate hf-transformers


nvidia-smi

python -m torch.utils.collect_env


python /scratch/gpfs/cmcwhite/attentionview_provis/attn_scripts/attn_calc.py  -f HF101_ARATH.fasta -mo /scratch/gpfs/cmcwhite/prot_bert_bfd -ma 0.1 -mf mutlist_HF101_ARATH.txt -o HF101_ARATH_w_muts.attn.csv
#python /scratch/gpfs/cmcwhite/attentionview_provis/attn_scripts/attn_calc.py  -f best1_human_only.fasta -mo /scratch/gpfs/cmcwhite/prot_bert_bfd -ma 0.1 -mf best_human_mutlist.txt -o best1_human_w_muts-attn.csv


