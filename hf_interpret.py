from transformers import AutoModelForSequenceClassification, AutoTokenizer
from transformers_interpret import SequenceClassificationExplainer
from transformer_notebooks.hf_utils import *

import argparse
import pandas as pd
import numpy as np

def get_interpret_args():

    parser = argparse.ArgumentParser()
    parser.add_argument("-m", "--model", dest = "model_path", type = str, required = True,
                        help="Model directory Ex. /path/to/model_dir")
    
    parser.add_argument("-s", "--sequence", dest = "sequence", type = str, required = False,
                        help="Single sequence as a string")
  
    parser.add_argument("-st", "-sequence_table", dest = "sequence_path", type = str, required = False,
                        help="Path to table of sequences to evaluate in csv (id,sequence) no header. Output of utils.parse_fasta")
    parser.add_argument("-f", "-fasta", dest = "fasta_path", type = str, required = False,
                        help="Path to fasta of sequences to evaluate")

    parser.add_argument("-n", "--dont_add_spaces" , action = "store_true",
                        help="Flag if sequences already have spaces")

    parser.add_argument("-o", "--outfile", dest = "outfile", type = str, required = True,
                        help="output csv for table of word attributions")
    parser.add_argument("-a", "--attribfile", dest = "attribfile", type = str, required = False,
                        help="For single sequences, output an attrib coloring file for chimera, .defattr suffix suggested")


    args = parser.parse_args()
    return(args)

def get_explainer(model_path):
    model = AutoModelForSequenceClassification.from_pretrained(model_path)
    tokenizer = AutoTokenizer.from_pretrained(model_path)

    # With both the model and tokenizer initialized we are now able to get explanations on an example text.

    cls_explainer = SequenceClassificationExplainer(
       model, 
       tokenizer)

    return(cls_explainer)

def explain_a_pred(sequence,cls_explainer):
    word_attributions = cls_explainer(sequence)

    pred_index = cls_explainer.predicted_class_index
    pred_prob = cls_explainer.pred_probs
    pred_name = cls_explainer.predicted_class_name

  

    # First and last are CLS and SEP, with value zero. remove thos
    #zip* converts tuple (aa, value) to two lists [aas], [values]
    aas, attributions = zip(*word_attributions[1:-1])

    return " ".join([str(x) for x in aas]), " ".join([str(x) for x in attributions]), pred_index, pred_prob, pred_name
if __name__ == "__main__":

    args = get_interpret_args()
    model_name = args.model_path

    explainer = get_explainer(model_name)


    if args.sequence:
       sequence = format_sequence(args.sequence, args.dont_add_spaces)
       print(sequence)
       word_attributions, pred_index, pred_prob, pred_name = explain_a_pred(sequence, explainer)
       #print(word_attributions.predicted_class_name)
       print(word_attributions)
       df = pd.DataFrame.from_records(word_attributions, columns = ['aa', 'contribution'])
       df['aa_position'] = np.arange(1, len(df) + 1)
       print(df)
       if args.attribfile:
           with open(args.attribfile, "w") as aoutfile:
                 aoutfile.write("attribute: percentExposed\n")
                 aoutfile.write("match mode: 1-to-1\n")
                 aoutfile.write("recipient: residues\n")
                 [aoutfile.write("\t:{}\t{}\n".format(x,y)) for x, y in zip(df['aa_position'], df['contribution'])]

    if args.fasta_path:
       sequence_lols = parse_fasta(args.fasta_path, "x.txt", args.dont_add_spaces)

       df = pd.DataFrame.from_records(sequence_lols,  columns=['id', 'sequence']) 

       df['output'] = df.apply(lambda row: explain_a_pred(row.sequence, explainer), axis = 1)
       
       # Split tuble to multiple columns
       df['aa'], df['word_attributions'], df['pred_index'], df['pred_prob'], df['pred_name'] = zip(*df.output)

       # Remove redundant columns
       df = df.drop(['sequence','output'], axis = 1)

       print(df)
       df.to_csv(args.outfile, index = False)










