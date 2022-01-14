#!/bin/bash
#SBATCH --job-name=aasimtrain       # create a short name for your job
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


#nvidia-smi

#python -m torch.utils.collect_env




#DIR=/scratch/gpfs/cmcwhite/
#
# Tester
#python transformer_infrastructure/aa_simtrain2.py -maxl 256 -trfasta testing_fastas.txt -traln testing_refs.txt  -dvfasta testing_fastas.txt -dvaln testing_refs.txt -m /scratch/gpfs/cmcwhite/prot_bert_bfd -e 10 -trbsize 10 -dvbsize 10 -o /scratch/gpfs/cmcwhite/aasimtrain_tester/ -l -5 -4 -3 -2 -1 -fs .vie -as .aln
 

#python transformer_infrastructure/aa_simtrain2.py -maxl 256 -trfasta testing_fastas.txt -traln testing_refs.txt  -dvfasta testing_fastas.txt -dvaln testing_refs.txt -m /scratch/gpfs/cmcwhite/prot_bert_bfd -e 10 -trbsize 10 -dvbsize 10 -o /scratch/gpfs/cmcwhite/aasimtrain_tester/ -l 1 2 3 4 5 6 7 8 9 10 11 12 13

#python transformer_infrastructure/aa_simtrain2.py -maxl 256 -trfasta testing_fastas.txt -traln testing_refs.txt  -tsfasta testing_fastas.txt -tsaln testing_refs.txt -dvfasta testing_fastas.txt -dvaln testing_refs.txt -m /scratch/gpfs/cmcwhite/prot_bert_bfd -e 10 -trbsize 10 -dvbsize 10 -o /scratch/gpfs/cmcwhite/prot_bert_bfd_aa_simtrain_quick/

#python transformer_infrastructure/aa_simtrain2.py -maxl 256 -trfasta testing_fastas.txt -traln testing_refs.txt  -tsfasta testing_fastas.txt -tsaln testing_refs.txt -dvfasta testing_fastas.txt -dvaln testing_refs.txt -m /scratch/gpfs/cmcwhite/afproject_fa_hilo_noQT_model/ -e 10 -trbsize 10 -dvbsize 10 -o /scratch/gpfs/cmcwhite/prot_bert_bfd_seq_aa_simtrain_quick/


# Change the sbatch time  
DIR=/scratch/gpfs/cmcwhite/aln_datasets/homstrad_trainsets/

python transformer_infrastructure/aa_simtrain2.py -maxl 256 -trfasta $DIR/homstrad_train_fastas.txt -traln $DIR/homstrad_train_refs.txt  -dvfasta $DIR/homstrad_dev_fastas.txt -dvaln $DIR/homstrad_dev_refs.txt -m /scratch/gpfs/cmcwhite/prot_bert_bfd/ -e 20 -trbsize 20 -dvbsize 20 -o /scratch/gpfs/cmcwhite/prot_bert_bfd_aa_simtrain_homstrad_4layertrain/ -as .filt.aln -fs .filt.fasta -l -4 -3 -2 -1 

#python transformer_infrastructure/aa_simtrain2.py -maxl 256 -trfasta $DIR/homstrad_train_fastas.txt -traln $DIR/homstrad_train_refs.txt  -dvfasta $DIR/homstrad_dev_fastas.txt -dvaln $DIR/homstrad_dev_refs.txt -m /scratch/gpfs/cmcwhite/prot_bert_bfd/ -e 20 -trbsize 20 -dvbsize 20 -o /scratch/gpfs/cmcwhite/prot_bert_bfd_aa_simtrain_homstrad_13layertrain/ -as .filt.aln -fs .filt.fasta -l -13 -12 -11 -10 -9 -8 -7 -6 -5 -4 -3 -2 -1 
 


#python transformer_infrastructure/aa_simtrain2.py -maxl 256 -trfasta $DIR/homstrad_train_fastas.txt -traln $DIR/homstrad_train_refs.txt  -dvfasta $DIR/homstrad_dev_fastas.txt -dvaln $DIR/homstrad_dev_refs.txt -m /scratch/gpfs/cmcwhite/afproject_fa_hilo_noQT_model/ -e 20 -trbsize 20 -dvbsize 20 -o /scratch/gpfs/cmcwhite/prot_bert_bfd_seq_aa_simtrain_homstrad/





#python transformer_infrastructure/aa_simtrain2.py -maxl 256 -trfasta $DIR/TrainFasta.csv -traln $DIR/TrainRef.csv -dvfasta $DIR/DevFasta.csv -dvaln $DIR/DevRef.csv -m /scratch/gpfs/cmcwhite/afproject_fa_hilo_noQT_model/ -e 20 -trbsize 10 -dvbsize 10 -o /scratch/gpfs/cmcwhite/prot_bert_bfd_seq_aa_simtrain_shuffle/ 
#

 

