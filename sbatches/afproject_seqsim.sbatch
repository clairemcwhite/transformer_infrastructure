#!/bin/bash
#SBATCH --job-name=hf_seqsim        # create a short name for your job
#SBATCH --nodes=1                # node count
#SBATCH --ntasks=1               # total number of tasks across all nodes
#SBATCH --cpus-per-task=4        # cpu-cores per task (>1 if multi-threaded tasks)
#SBATCH --mem-per-cpu=32G         # memory per cpu-core (4G is default)
#SBATCH --gres=gpu:1             # number of gpus per node
#SBATCH --time=04:00:00          # total run time limit (HH:MM:SS)
#SBATCH --mail-type=begin        # send email when job begins
#SBATCH --mail-type=end          # send email when job ends
#SBATCH --mail-user=cmcwhite@princeton.edu

module purge
module load anaconda3/2020.11
module load cudatoolkit

conda activate hf-transformers


nvidia-smi

python -m torch.utils.collect_env


# Low identity sequences
python transformer_infrastructure/hf_similarity2.py -m /scratch/gpfs/cmcwhite/prot_bert_bfd/ -o /scratch/gpfs/cmcwhite/afproject_output/simtest_lowident_untrained.txt -f /scratch/gpfs/cmcwhite/afproject/protein-low-ident/protein_low_ident.fasta

python transformer_infrastructure/hf_similarity2.py -m /scratch/gpfs/cmcwhite/afproject_fa_hilo_noQT_model/ -o /scratch/gpfs/cmcwhite/afproject_output/simtest_lowident_fa_hilo_noQT.txt -f /scratch/gpfs/cmcwhite/afproject/protein-low-ident/protein_low_ident.fasta


#python transformer_infrastructure/hf_similarity2.py -m /scratch/gpfs/cmcwhite/afproject_model/0_Transformer/ -o /scratch/gpfs/cmcwhite/afproject_output/simtest_lowident_trained_smallerset_fa.txt -f /scratch/gpfs/cmcwhite/afproject/protein-low-ident/protein_low_ident.fasta

#python transformer_infrastructure/hf_similarity2.py -m /scratch/gpfs/cmcwhite/afproject_px_model/0_Transformer/ -o /scratch/gpfs/cmcwhite/afproject_output/simtest_lowident_trained_smallerset_px.txt -f /scratch/gpfs/cmcwhite/afproject/protein-low-ident/protein_low_ident.fasta

#python transformer_infrastructure/hf_similarity2.py -m /scratch/gpfs/cmcwhite/afproject_fa_hilo_model/ -o /scratch/gpfs/cmcwhite/afproject_output/simtest_lowident_trained_largerset_fa.txt -f /scratch/gpfs/cmcwhite/afproject/protein-low-ident/protein_low_ident.fasta

#python transformer_infrastructure/hf_similarity2.py -m /scratch/gpfs/cmcwhite/afproject_px_hilo_model/ -o /scratch/gpfs/cmcwhite/afproject_output/simtest_lowident_trained_largerset_px.txt -f /scratch/gpfs/cmcwhite/afproject/protein-low-ident/protein_low_ident.fasta


# High identity sequences
python transformer_infrastructure/hf_similarity2.py -m  /scratch/gpfs/cmcwhite/prot_bert_bfd/ -o /scratch/gpfs/cmcwhite/afproject_output/simtest_highident_untrained.txt -f /scratch/gpfs/cmcwhite/afproject/protein-high-ident/protein_high_ident.fasta

python transformer_infrastructure/hf_similarity2.py -m /scratch/gpfs/cmcwhite/afproject_fa_hilo_noQT_model/ -o /scratch/gpfs/cmcwhite/afproject_output/simtest_highident_fa_hilo_noQT.txt -f /scratch/gpfs/cmcwhite/afproject/protein-high-ident/protein_high_ident.fasta

#python transformer_infrastructure/hf_similarity2.py -m /scratch/gpfs/cmcwhite/afproject_model/0_Transformer/ -o /scratch/gpfs/cmcwhite/afproject_output/simtest_highident_trained_smallerset_fa.txt -f /scratch/gpfs/cmcwhite/afproject/protein-high-ident/protein_high_ident.fasta

#python transformer_infrastructure/hf_similarity2.py -m /scratch/gpfs/cmcwhite/afproject_px_model/0_Transformer/ -o /scratch/gpfs/cmcwhite/afproject_output/simtest_highident_trained_smallerset_px.txt -f /scratch/gpfs/cmcwhite/afproject/protein-high-ident/protein_high_ident.fasta

#python transformer_infrastructure/hf_similarity2.py -m /scratch/gpfs/cmcwhite/afproject_fa_hilo_model/ -o /scratch/gpfs/cmcwhite/afproject_output/simtest_highident_trained_largerset_fa.txt -f /scratch/gpfs/cmcwhite/afproject/protein-high-ident/protein_high_ident.fasta

#python transformer_infrastructure/hf_similarity2.py -m /scratch/gpfs/cmcwhite/afproject_px_hilo_model/ -o /scratch/gpfs/cmcwhite/afproject_output/simtest_highident_trained_largerset_px.txt -f /scratch/gpfs/cmcwhite/afproject/protein-high-ident/protein_high_ident.fasta


