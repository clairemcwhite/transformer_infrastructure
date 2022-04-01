#!/bin/bash
#SBATCH --job-name=index_sam       # create a short name for your job
#SBATCH --nodes=1                # node count
#SBATCH --ntasks=1               # total number of tasks across all nodes
#SBATCH --cpus-per-task=1        # cpu-cores per task (>1 if multi-threaded tasks)
#SBATCH --mem-per-cpu=64G         # memory per cpu-core (4G is default)
#SBATCH --gres=gpu:1             # number of gpus per node
#SBATCH --time=48:00:00          # total run time limit (HH:MM:SS)
#SBATCH --mail-type=begin        # send email when job begins
#SBATCH --mail-type=end          # send email when job ends
#SBATCH --mail-user=cmcwhite@princeton.edu

module purge
module load anaconda3/2020.11
module load cudatoolkit

conda activate hf-transformers


nvidia-smi




python -m torch.utils.collect_env


mkdir human_protemee_segmented

echo "Break proteome into groups of 1000"
python $HOME/MS_grouped_lookup/scripts/proteome_utils/proteome_breaker.py human_proteome.fasta 1000 fasta_segmented 

echo "Add each fasta to the index"
python $HOME/transformer_infrastructure/hf_buildindex.py -i fasta_segmented/human_proteome*seg[1-9]*.fasta -o radicalsam_13.faissindex --layers -13 -12 -11 -10 -9 -8 -7 -6 -5 -4 -3 -2 -1  -m /scratch/gpfs/cmcwhite/prot_bert_bfd -t 12000

 
