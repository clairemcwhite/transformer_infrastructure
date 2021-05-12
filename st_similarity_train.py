#!/usr/bin/env python

from transformer_infrastructure.hf_utils import get_sequencelabel_tags, SS3Dataset
from torch.utils.data import DataLoader
from sentence_transformers import losses, util
from sentence_transformers import LoggingHandler, SentenceTransformer, evaluationfrom sentence_transformers.readers import InputExample
import logging
from datetime import datetime
import csv
import os
from zipfile import ZipFile
import random
import argparse

def get_args():

    parser = argparse.ArgumentParser()
    parser.add_argument("-m", "--model", dest = "model_name", type = str, required = True, 
                        help="Model directory path or name on huggingface. Ex. /path/to/model_dir Rostlab/prot_bert_bfd")
    parser.add_argument("-tr", "--train", dest = "train_path", type = str, required = True, 
                        help="Path to training set, containing columns named sequence,label (csv)")
    parser.add_argument("-d", "--dev", dest = "dev_path", type = str, required = True, 
                        help="Path to dev/validation set (used during training), containing columns named sequence,label (csv)")
    parser.add_argument("-te", "--test", dest = "test_path", type = str, required = True, 
                        help="Path to withheld test set (used after training), containing columns named sequence,label (csv)")
    parser.add_argument("-o", "--outdir", dest = "outdir", type = str, required = True, 
                        help="Name of output directory")
    parser.add_argument("-maxl", "--maxseqlength", dest = "max_length", type = int, required = False, default = 1024, 
                        help="Truncate all sequences to this length (default 1024). Reduce if memory errors")
    parser.add_argument("-n", "--expname", dest = "expname", type = str, required = False, default = "transformer_run",
                        help="Experiment name, used for logging, default = transformer_run")
    parser.add_argument("-e", "--epochs", dest = "epochs", type = int, required = False, default = 10, 
                        help="Number of epochs. Increasing can help if memory error")
    parser.add_argument("-tbsize", "--train_batchsize", dest = "train_batchsize", type = int, required = False, default = 10, 
                        help="Per device train batchsize. Reduce along with val batch size if memory error")
    parser.add_argument("-vbsize", "--dev_batchsize", dest = "dev_batchsize", type = int, required = False, default = 10, 
                        help="Per device validation batchsize. Reduce if memory error")

  
    args = parser.parse_args()
    return(args)


def load_dataset(path, max_length, label_column):
        df  = pd.read_csv(path)
      
        df['seq1_fixed'] = ["".join(seq.split()) for seq in df['sequence1']]
        df['seq1_fixed'] = [re.sub(r"[UZOB]", "X", seq) for seq in df['seq1_fixed']]
        seqs1 = [ list(seq)[:max_length-2] for seq in df['seq1_fixed']]

        df['seq2_fixed'] = ["".join(seq.split()) for seq in df['sequence2']]
        df['seq2_fixed'] = [re.sub(r"[UZOB]", "X", seq) for seq in df['seq2_fixed']]
        seqs2 = [ list(seq)[:max_length-2] for seq in df['seq2_fixed']]

        labels = list(df[label_column]) # ex. label

        id1s = list(df['id1'])
        id2s = list(df['id1'])
        assert len(seqs1) == len(seqs2) == len(labels) == len(ids1) == len(ids2)
        return seqs1, seqs2, labels, ids1, ids2


def encode_tags(tags, tag2id):

    encoded_labels = [tag2id[tag] for tag in tags]
    return encoded_labels


if __name__ == "__main__":


 

    #max_length = 1024
    #train_path = '/home/jupyter/chloro_loc/chloro_labeledsetTrain.csv'
    #test_path = '/home/jupyter/chloro_loc/chloro_labeledsetTest.csv'
    #dev_path = '/home/jupyter/chloro_loc/chloro_labeledsetVal.csv'
    #   if args.n_gpu > 1#:
    #    model = torch.nn.DataParallel(model)

    args = get_args()

    model_name = args.model_name
    max_length = args.max_length
    train_path = args.train_path
    test_path = args.test_path
    dev_path = args.dev_path

    log_format = "%(asctime)s::%(levelname)s::%(name)s::"\
             "%(filename)s::%(lineno)d::%(message)s"

    expname = args.expname
    outdir = args.outdir
    logname = outdir + expname + "_" + model_name.strip("/") + ".log"
    print("logging at ", logname)
    logging.basicConfig(filename=logname, level='DEBUG', format=log_format)

    logging.info("Check for torch")
    logging.info(torch.cuda.is_available())

    epochs = args.epochs
    checkpoint = args.checkpoint
    train_batchsize = args.train_batchsize
    dev_batchsize = args.dev_batchsize
 
    #return seqs1, seqs2, labels, ids1, ids2
    train_seqs1, train_seqs2, train_labels, train_ids1, train_ids2 = load_dataset(train_path, max_length)
    dev_seqs1, dev_seqs2, dev_labels, dev_ids1, dev_ids2 = load_dataset(dev_path, max_length)
    test_seqs1, test_seqs2, test_labels, test_ids1, test_ids2 = load_dataset(test_path, max_length)

    logging.info("datasets loaded")


    # Consider each label as a tag for each token
    
    #tag2id, id2tag = get_sequencelabel_tag(train_labels)

    #logging.info("id2tag")
    #logging.info(id2tag)
    #logging.info("tag2id")
    #logging.info(tag2id)
     

    #labels_encodings = encode_tags(labels, tag2id)
    train_samples = []
    for i in range(len(train_seqs1)):
         if train_label_encodings[i] == 1:
            train_samples.append(InputExample(texts=[train_seqs1[i], train_seqs2[i], train_labels[i]))

    train_dataloader = DataLoader(train_samples, shuffle=True, batch_size=train_batch_size)  


    train_loss = losses.MultipleNegativesRankingLoss(model)

    # Set up a set of different performatnce evaluators
 
    evaluators = []
   
    ###### Classification ######
    # Given (quesiton1, question2), is this a duplicate or not?
    # The evaluator will compute the embeddings for both questions and then compute
    # a cosine similarity. If the similarity is above a threshold, we have a duplicate.
    binary_acc_evaluator = evaluation.BinaryClassificationEvaluator(dev_seqs1, dev_seqs2, dev_labels)
    evaluators.append(binary_acc_evaluator)
 
    binary_acc_evaluator = evaluation.BinaryClassificationEvaluator(train_sequences1, train_sequences2, train_labels)
   evaluators.append(binary_acc_evaluator)
   
   logging.info("binary acc evaluator added")
   dev_seq_dict = {}
   dev_duplicates = []

   # create dict of id:seq
   for i in range(len(dev_seqs1)):
      dev_seq_dict[dev_ids1] = dev_seqs1
      dev_seq_dict[dev_ids2] = dev_seqs2
   # Create pairs list of duplicate ids
   for i in range(len(dev_seqs1)):
      if dev_labels[i] == 1:
        dev_duplicates.append([dev_ids1[i], dev_ids2[i])


    # The ParaphraseMiningEvaluator computes the cosine similarity between all sentences and
    # extracts a list with the pairs that have the highest similarity. Given the duplicate
    # information in dev_duplicates, it then computes and F1 score how well our duplicate mining worked
    paraphrase_mining_evaluator = evaluation.ParaphraseMiningEvaluator(dev_seq_dict, dev_duplicates, name='dev', show_progress_bar = True)
    evaluators.append(paraphrase_mining_evaluator)

    logging.info("paraphrase evaluator added")
   
    seq_evaluator = evaluation.SequentialEvaluator(evaluators, main_score_function=lambda scores: scores[-1])


    logger.info("Evaluate model without training")
    seq_evaluator(model, epoch=0, steps=0, output_path=model_save_path)



    logging.info("training complete")
    trainer.save_model(outdir)
    seq_tokenizer.save_pretrained(outdir)
  
    logging.info("model saved")
    logging.info(outdir)


    get_predictions(outdir, test_path, len(tag2id), max_length, "withheldtest")
    get_predictions(outdir, dev_path, len(tag2id), max_length, "val")
    get_predictions(outdir, train_path, len(tag2id), max_length, "train")
    #test_predictions, test_label_ids, test_metrics = trainer.predict(test_dataset)
    #logging.info("test metrics (withheld)")
    #logging.info(test_metrics)
    #outtest  = outdir + "/" + expname + "_test_predictions.csv"
    #np.savetxt(outtest, test_predictions, delimiter=',')

    #train_predictions, train_label_ids, train_metrics = trainer.predict(train_dataset)
    #logging.info("train metrics")
    #logging.info(train_metrics)
    #outtrain  = outdir + "/" + expname + "_train_predictions.csv"
    #np.savetxt(outtrain, train_predictions, delimiter=',')


    #dev_predictions, dev_label_ids, dev_metrics = trainer.predict(dev_dataset)
    #logging.info("val metrics (seen during training)")
    #logging.info(dev_metrics)
    #outval  = outdir + "/" + expname + "_dev_predictions.csv"
    #np.savetxt(outval, dev_predictions, delimiter=',')

 



    #idx = 2
    #sample_ground_truth = test_dataset[idx]['labels']
    #sample_predictions =  np.argmax(predictions[idx])
    #
    #
    ## In[57]:
    #
    #
    #sample_sequence = seq_tokenizer.decode(list(test_dataset[idx]['input_ids']), skip_special_tokens=True)
    #
    #
    ## In[58]:
    #
    #
    #
    #print(sample_ground_truth)
    #print(sample_predictions)
    #
    #
    ## In[41]:
    #
    #
    #print("Sequence       : {} \nGround Truth is: {}\nprediction is  : {}".format(#sample_sequence,
    #                                                                      sample_ground_truth,
    #                                                                      # Remove the first token on prediction becuase its CLS token
    #                                                                      # and only show up to the input length
    #                                                                      sample_predictions))
    #      
    #
    #
    ## **14. Save the model**
    #
    ## **15. Check Tensorboard**
    #
    ## In[ ]:
    #
    #
    #get_ipython().run_line_magic('load_ext', 'tensorboard')
    #get_ipython().run_line_magic('tensorboard', '--logdir logs')
    #
