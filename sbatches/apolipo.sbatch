#!/bin/bash
#SBATCH --job-name=hf_apolipo        # create a short name for your job
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

python transformer_infrastructure/hf_classification.py -m /scratch/gpfs/cmcwhite/prot_t5_xl_uniref50 -tr /scratch/gpfs/cmcwhite/apolipo_social_model/apolipo_labeledsetTrain.csv -v /scratch/gpfs/cmcwhite/apolipo_social_model/apolipo_labeledsetTest.csv -te /scratch/gpfs/cmcwhite/apolipo_social_model/apolipo_labeledsetTest.csv -o /scratch/gpfs/cmcwhite/apolipo_social_model_t5/ -maxl 1024 -n apolipo_sociality_t5 -e 10 -tbsize 1 -vbsize 1

python transformer_infrastructure/hf_classification.py -m /scratch/gpfs/cmcwhite/prot_bert_bfd/ -tr /scratch/gpfs/cmcwhite/apolipo_social_model/apolipo_labeledsetTrain.csv -v /scratch/gpfs/cmcwhite/apolipo_social_model/apolipo_labeledsetTest.csv -te /scratch/gpfs/cmcwhite/apolipo_social_model/apolipo_labeledsetTest.csv -o /scratch/gpfs/cmcwhite/apolipo_social_model/ -maxl 1024 -n apolipo_sociality_bfd -e 10 -tbsize 1 -vbsize 1
