#!/bin/bash
#SBATCH --job-name=hf_ukbb_ev    # create a short name for your job
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


python transformer_infrastructure/hf_evaluation_aa.py -m /scratch/gpfs/cmcwhite/ukbb_geneticvariation_model/ -l  ukbb_geneticvariation/test_ukbb_geneticvariation.csv -s Test -maxl 1024 -p G

python transformer_infrastructure/hf_evaluation_aa.py -m /scratch/gpfs/cmcwhite/ukbb_geneticvariation_model/ -l  ukbb_geneticvariation/train_ukbb_geneticvariation.csv -s Train -maxl 1024 -p G


python transformer_infrastructure/hf_evaluation_aa.py -m /scratch/gpfs/cmcwhite/ukbb_geneticvariation_model/ -l  ukbb_geneticvariation/val_ukbb_geneticvariation.csv -s Val -maxl 1024 -p G
