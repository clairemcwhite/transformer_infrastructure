#!/bin/bash
#SBATCH --job-name=seqs_fa        # create a short name for your job
#SBATCH --nodes=1                # node count
#SBATCH --ntasks=1               # total number of tasks across all nodes
#SBATCH --cpus-per-task=1        # cpu-cores per task (>1 if multi-threaded tasks)
#SBATCH --mem-per-cpu=32G         # memory per cpu-core (4G is default)
#SBATCH --gres=gpu:1             # number of gpus per node
#SBATCH --time=24:00:00          # total run time limit (HH:MM:SS)
#SBATCH --mail-type=begin        # send email when job begins
#SBATCH --mail-type=end          # send email when job ends
#SBATCH --mail-user=cmcwhite@princeton.edu

module purge
module load anaconda3/2020.11
module load cudatoolkit

conda activate hf-transformers


nvidia-smi

python -m torch.utils.collect_env


python transformer_infrastructure/st_similarity_train.py -m /scratch/gpfs/cmcwhite/prot_bert_bfd/ -tr /scratch/gpfs/cmcwhite/scope/allscope_sz15groups_train.csv -d /scratch/gpfs/cmcwhite/scope/allscope_sz15groups_dev.csv  -te /scratch/gpfs/cmcwhite/scope/allscope_sz15groups_test.csv  -o /scratch/gpfs/cmcwhite/allscope_sz15groups_256 -maxl 256 -n allscope_sz15groups_256 -e 10 -tbsize 10 -vbsize 10 -l label_fa
