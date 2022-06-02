#!/bin/bash
#SBATCH --job-name=index_sam       # create a short name for your job
#SBATCH --nodes=1                # node count
#SBATCH --ntasks=1               # total number of tasks across all nodes
#SBATCH --cpus-per-task=1        # cpu-cores per task (>1 if multi-threaded tasks)
#SBATCH --mem-per-cpu=64G         # memory per cpu-core (4G is default)
#SBATCH --gres=gpu:1             # number of gpus per node
#SBATCH --time=2:00:00          # total run time limit (HH:MM:SS)
#SBATCH --mail-type=begin        # send email when job begins
#SBATCH --mail-type=end          # send email when job ends
#SBATCH --mail-user=cmcwhite@princeton.edu

module purge
module load anaconda3/2020.11
module load cudatoolkit

conda activate hf-transformers


nvidia-smi

python -m torch.utils.collect_env
for f in fastas/radicalsam_xa[ijklmnopqrst].fasta
do 
   python $HOME/transformer_infrastructure/hf_seqsim.py -i $f -o ${f}.k100.st90.seqsims.csv -dx radicalsam_13d4.faissindex -dxn radicalsam_13d4.faissindex.names --layers -13 -12 -11 -10 -9 -8 -7 -6 -5 -3 -2 -1  -m /scratch/gpfs/cmcwhite/prot_bert_bfd -st .9 -k 100
done
