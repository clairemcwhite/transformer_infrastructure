#!/bin/bash
#SBATCH --job-name=run_mcl       # create a short name for your job
#SBATCH --nodes=1                # node count
#SBATCH --ntasks=1               # total number of tasks across all nodes
#SBATCH --cpus-per-task=1        # cpu-cores per task (>1 if multi-threaded tasks)
#SBATCH --mem-per-cpu=64G         # memory per cpu-core (4G is default)
#SBATCH --time=48:00:00          # total run time limit (HH:MM:SS)
#SBATCH --mail-type=begin        # send email when job begins
#SBATCH --mail-type=end          # send email when job ends
#SBATCH --mail-user=cmcwhite@princeton.edu

module purge
module load anaconda3/2020.11

conda activate hf-transformers

ST=95
for i in 5 #5 10 15
do echo $f
   mcl most_seqsims_k100_st${ST}_randomized_nonred.tab -te 20 -I $i --abc 
   python mcl_to_edges2.py out.most_seqsims_k100_st${ST}_randomized_nonred.tab.I${i}0  most_seqsims_k100_st${ST}_randomized_nonred.tab 
   python radical_lgls/format_ncol.py -i out.most_seqsims_k100_st${ST}_randomized_nonred.tab.I${i}0.network_randomized_nonred.tab -o radical_lgls/out.most_seqsims_k100_st${ST}_randomized_nonred.tab.I${i}0.network_randomized_nonred.tab.ncol
 done

