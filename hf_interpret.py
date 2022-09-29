from transformers import AutoModelForSequenceClassification, AutoTokenizer
from transformers_interpret import SequenceClassificationExplainer
from transformer_infrastructure.hf_utils import *

import argparse
import pandas as pd
import numpy as np

def get_interpret_args():

    parser = argparse.ArgumentParser()
    parser.add_argument("-m", "--model", dest = "model_path", type = str, required = True,
                        help="Model directory Ex. /path/to/model_dir")
    
    parser.add_argument("-s", "--sequence", dest = "sequence", type = str, required = False,
                        help="Single sequence as a string")
  
    #parser.add_argument("-st", "-sequence_table", dest = "sequence_path", type = str, required = False,
    #                    help="Path to table of sequences to evaluate in csv (id,sequence) no header. Output of utils.parse_fasta")
    parser.add_argument("-f", "-fasta", dest = "fasta_path", type = str, required = False,
                        help="Path to fasta of sequences to evaluate")

    parser.add_argument("-n", "--dont_add_spaces" , action = "store_true",
                        help="Flag if sequences already have spaces")

    parser.add_argument("-o", "--outfile", dest = "outfile", type = str, required = True,
                        help="output csv for table of word attributions")
    parser.add_argument("-a", "--attribfile", dest = "attribfile", type = str, required = False,
                        help="For single sequences, output an attrib coloring file for chimera, .defattr suffix suggested")
    parser.add_argument("-l", "--labels", dest = "labels", type = str, required = False,
                        help="csv of labels, (id,sequence,label) for annotating. Optional.")
    parser.add_argument("-ml", "--maxlength", dest = "maxlength", type = int, required = False,
                        help="Truncate sequences to this length. Optional.")
    args = parser.parse_args()
    return(args)

def get_explainer(model_path):
    model = AutoModelForSequenceClassification.from_pretrained(model_path)
    tokenizer = AutoTokenizer.from_pretrained(model_path)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    print(model.device)

    # With both the model and tokenizer initialized we are now able to get explanations on an example text.

    cls_explainer = SequenceClassificationExplainer(
       model, 
       tokenizer)
    print("got explainer")

    print(model.config.id2label)
    return(cls_explainer)

def explain_a_pred(sequence,cls_explainer):

    print(sequence[0:5])
    word_attributions = cls_explainer(text = sequence)


    print("got word attributions")
    pred_index = cls_explainer.predicted_class_index
    pred_prob = cls_explainer.pred_probs.cpu()
    pred_name = cls_explainer.predicted_class_name

  

    # First and last are CLS and SEP, with value zero. remove thos
    #zip* converts tuple (aa, value) to two lists [aas], [values]
    aas, attributions = zip(*word_attributions[1:-1])
    positions =  np.arange(1, len(aas) + 1)
    print("formatted")
    # round to 8 digits
    return " ".join([str(x) for x in aas]), " ".join([str(round(x, 8)) for x in attributions]), " ".join([str(x) for x in positions]), pred_index, pred_prob, pred_name


if __name__ == "__main__":

    args = get_interpret_args()
    model_name = args.model_path
    maxlength = args.maxlength
    explainer = get_explainer(model_name)


    if args.sequence:
       sequence = args.sequence
       # This needs to be fixed
       if maxlength: 
           sequence = sequence[0:maxlength]
       sequence = format_sequence(sequence, add_spaces = True)
       print(sequence)



       # Do something to remove long sequences
       aas, word_attributions, pos,  pred_index, pred_prob, pred_name = explain_a_pred(sequence, explainer)
       #print(word_attributions.predicted_class_name)
       print("wa", word_attributions)
       print("aas", aas)
       print("pos", pos)
       print("pred_index", pred_index)
       print("pred_prob", pred_prob)
       print("pred_name", pred_name)
       print(word_attributions)
       info =  list(zip(aas.split(" "), word_attributions.split(" ")))
       print(info)
       df = pd.DataFrame.from_records(info, columns = ['aa', 'contribution'])
       #df = pd.DataFrame.from_records(info, columns = ['pred_prob', 'pred_name', 'pred_index', 'word_attributions', 'aas', 'pos'])
       print(df)
       df['aa_position'] = np.arange(1, len(df) + 1)
       print(df)
       if args.attribfile:
           with open(args.attribfile, "w") as aoutfile:
                 aoutfile.write("attribute: percentExposed\n")
                 aoutfile.write("match mode: 1-to-1\n")
                 aoutfile.write("recipient: residues\n")
                 [aoutfile.write("\t:{}\t{}\n".format(x,y)) for x, y in zip(df['aa_position'], df['contribution'])]

    if args.fasta_path:
       fasta_tbl = args.fasta_path + ".txt"
       sequence_lols = parse_fasta(args.fasta_path, fasta_tbl, True, maxlength)
       print(len(sequence_lols))
       #sequence_lols = [x for x in sequence_lols if len(x[1]) < 250]
       print(len(sequence_lols))
       print(sequence_lols)
       df = pd.DataFrame.from_records(sequence_lols,  columns=['id', 'sequence', 'sequence_spaced']) 

       #df = df.head(1)
       print(df)
       pd.set_option('display.max_colwidth', None)
       
       df['output'] = df.apply(lambda row: explain_a_pred(row.sequence_spaced, explainer), axis = 1)
       print(df['output'])
       
       # Split tuble to multiple columns
       df['aa'], df['word_attributions'], df['positions'], df['pred_index'], df['pred_prob'], df['pred_name'] = zip(*df.output)

       # Remove redundant columns
 
       df = df.sort_values(by=['pred_prob'],  ascending=False)
       if args.labels:
           labels = pd.read_csv(args.labels)
           labels.columns =['id', 'sequence_spaced', 'label']
         
           df = labels.merge(df, on = ['id','sequence_spaced'], how='right')
           df = df[['id','pred_prob', 'pred_name', 'label', 'pred_index','sequence_spaced','word_attributions', 'positions', 'sequence']]

       df.to_csv(args.outfile, index = False)










