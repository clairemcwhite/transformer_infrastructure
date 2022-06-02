#!/bin/bash
#SBATCH --job-name=hf_clusterseqs        # create a short name for your job
#SBATCH --nodes=1                # node count
#SBATCH --ntasks=1               # total number of tasks across all nodes
#SBATCH --cpus-per-task=1        # cpu-cores per task (>1 if multi-threaded tasks)
#SBATCH --mem-per-cpu=16G         # memory per cpu-core (4G is default)
#SBATCH --gres=gpu:1             # number of gpus per node
#SBATCH --time=08:00:00          # total run time limit (HH:MM:SS)
#SBATCH --mail-type=begin        # send email when job begins
#SBATCH --mail-type=end          # send email when job ends
#SBATCH --mail-user=cmcwhite@princeton.edu

module purge
module load anaconda3/2020.11
module load cudatoolkit

conda activate hf-transformers


nvidia-smi

python -m torch.utils.collect_env

python  /home/cmcwhite/Downloads/chimerax-1.3/lib/python3.9/site-packages/ChimeraX_main.py --nogui --script "/scratch/gpfs/cmcwhite/attentionview_provis/attn_scripts/chart_attentions.py  -st /scratch/gpfs/cmcwhite/attentionview_provis/AF-Q9FX53-F1-model_v2.pdb -mo /scratch/gpfs/cmcwhite/prot_bert_bfd -ma 0.1"

python  /home/cmcwhite/Downloads/chimerax-1.3/lib/python3.9/site-packages/ChimeraX_main.py --nogui --script "/scratch/gpfs/cmcwhite/attentionview_provis/attn_scripts/chart_attentions.py  -st /scratch/gpfs/cmcwhite/attentionview_provis/AF-Q9FX53-F1-model_v2.pdb -mo /scratch/gpfs/cmcwhite/prot_bert_bfd -ma 0.1 -mu S198A"

python  /home/cmcwhite/Downloads/chimerax-1.3/lib/python3.9/site-packages/ChimeraX_main.py --nogui --script "/scratch/gpfs/cmcwhite/attentionview_provis/attn_scripts/chart_attentions.py  -st /scratch/gpfs/cmcwhite/attentionview_provis/AF-Q9FX53-F1-model_v2.pdb -mo /scratch/gpfs/cmcwhite/prot_bert_bfd -ma 0.1 -mu S198D"

python  /home/cmcwhite/Downloads/chimerax-1.3/lib/python3.9/site-packages/ChimeraX_main.py --nogui --script "/scratch/gpfs/cmcwhite/attentionview_provis/attn_scripts/chart_attentions.py  -st /scratch/gpfs/cmcwhite/attentionview_provis/AF-Q9FX53-F1-model_v2.pdb -mo /scratch/gpfs/cmcwhite/prot_bert_bfd -ma 0.1 -mu S175A"

python  /home/cmcwhite/Downloads/chimerax-1.3/lib/python3.9/site-packages/ChimeraX_main.py --nogui --script "/scratch/gpfs/cmcwhite/attentionview_provis/attn_scripts/chart_attentions.py  -st /scratch/gpfs/cmcwhite/attentionview_provis/AF-Q9FX53-F1-model_v2.pdb -mo /scratch/gpfs/cmcwhite/prot_bert_bfd -ma 0.1 -mu S175D"

python  /home/cmcwhite/Downloads/chimerax-1.3/lib/python3.9/site-packages/ChimeraX_main.py --nogui --script "/scratch/gpfs/cmcwhite/attentionview_provis/attn_scripts/chart_attentions.py  -st /scratch/gpfs/cmcwhite/attentionview_provis/AF-Q9FX53-F1-model_v2.pdb -mo /scratch/gpfs/cmcwhite/prot_bert_bfd -ma 0.1 -mu T133A"

python  /home/cmcwhite/Downloads/chimerax-1.3/lib/python3.9/site-packages/ChimeraX_main.py --nogui --script "/scratch/gpfs/cmcwhite/attentionview_provis/attn_scripts/chart_attentions.py  -st /scratch/gpfs/cmcwhite/attentionview_provis/AF-Q9FX53-F1-model_v2.pdb -mo /scratch/gpfs/cmcwhite/prot_bert_bfd -ma 0.1 -mu T133D"

python  /home/cmcwhite/Downloads/chimerax-1.3/lib/python3.9/site-packages/ChimeraX_main.py --nogui --script "/scratch/gpfs/cmcwhite/attentionview_provis/attn_scripts/chart_attentions.py  -st /scratch/gpfs/cmcwhite/attentionview_provis/AF-Q9FX53-F1-model_v2.pdb -mo /scratch/gpfs/cmcwhite/prot_bert_bfd -ma 0.1 -mu T89D"

python  /home/cmcwhite/Downloads/chimerax-1.3/lib/python3.9/site-packages/ChimeraX_main.py --nogui --script "/scratch/gpfs/cmcwhite/attentionview_provis/attn_scripts/chart_attentions.py  -st /scratch/gpfs/cmcwhite/attentionview_provis/AF-Q9FX53-F1-model_v2.pdb -mo /scratch/gpfs/cmcwhite/prot_bert_bfd -ma 0.1 -mu T89A"








#python  /home/cmcwhite/Downloads/chimerax-1.3/lib/python3.9/site-packages/ChimeraX_main.py --nogui --script "/scratch/gpfs/cmcwhite/attentionview_provis/attn_scripts/chart_attentions.py  -st /scratch/gpfs/cmcwhite/attentionview_provis/6vx6.pdb -mo /scratch/gpfs/cmcwhite/prot_bert_bfd -ma 0.1"


#python  /home/cmcwhite/Downloads/chimerax-1.3/lib/python3.9/site-packages/ChimeraX_main.py --nogui --script "/scratch/gpfs/cmcwhite/attentionview_provis/attn_scripts/chart_attentions.py  -st /scratch/gpfs/cmcwhite/attentionview_provis/6vx6.pdb -mo /scratch/gpfs/cmcwhite/prot_bert_bfd -ma 0.1 -mu D302H"

