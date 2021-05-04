
### General
save_hf_model_locally.py

 - Huggingface models must be saved locally to run on the cluster
 - argparse: TODO

fasta2spacedtable.py
 
 - Convert a fasta to table of ID,S E Q U E N C E 
 - argparse


hf_util.py
 - Common utilies
 - More to go in here, commonalities between sequence/aa level classification

### Classify whole sequences
hf_classification.py

 - Classify labeled sequences (ex. chloroplast localization) 
 - argparse
 - logger
 - evaluator: hf_evaluation.py

hf_evaluation.py 

 - Evaluate performance of sequence classification
 - argparse
 - logger: TODO

hf_predict.py

 - Apply classification model to new sequences: TODO
 - argparse: TODO
 - logger: TODO

hf_embed.py       

 - Convert fasta to embedding 
 - argparse
 - logger: TODO


hf_cluster.py     

 - Cluster sequence embeddings
 - argparse: TODO
 - logger:TODO


hf_interpret.py

 - Get word importances for each sequence classification
 - argparse: TODO
 - logger:TODO


### Classify amino acids

hf_classification_aa.py
 - Classify individual amino acids
 - argparse
 - logger
 - evaluator: TODO

hf_evaluate_aa.py : TODO
 - Evaluate performance of sequence classification
 - argparse: TODO
 - logger: TODO

hf_predict_aa.py: TODO

 - Apply classification model to new sequences: TODO
 - argparse: TODO
 - logger: TODO

Figure out what to do for interpretibility

### Motifs from pretrained prot-bert

for each amino acid...
Get embedding
Get saliency map of each sequence 
https://colab.research.google.com/github/AndreasMadsen/python-textualheatmap/blob/master/notebooks/huggingface_bert_example.ipynb#scrollTo=X8GJbpoUmYdT

for each aa record saliency interactions, and embeddings of those interactions

compare similarity of embeddings.....










