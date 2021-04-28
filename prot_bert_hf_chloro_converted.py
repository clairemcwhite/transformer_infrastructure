#!/usr/bin/env python
# coding: utf-8

# **1. Check the GPU device**
# 

# In[4]:


get_ipython().system('nvidia-smi')


# **2. Load necessry libraries including huggingface transformers**

# In[5]:


get_ipython().system('pip -q install transformers seqeval')


# In[ ]:


torch os pandas requests tqdm numpy seqeval re


# In[6]:


import torch
from transformers import AutoTokenizer, Trainer, TrainingArguments, AutoModelForSequenceClassification, BertTokenizerFast, EvalPrediction
from torch.utils.data import Dataset
import os
import pandas as pd
import requests
from tqdm.auto import tqdm
import numpy as np
from seqeval.metrics import accuracy_score, f1_score, precision_score, recall_score
import re


# **3. Select the model you want to fine-tune**

# In[7]:


model_name = 'Rostlab/prot_bert_bfd'


# **5. Load dataset into memory**

# In[8]:


def load_dataset(path, max_length):
        df  = pd.read_csv(path)
        #df = pd.read_csv(path,names=['input','label','disorder'],skiprows=1)
      
        df['seq_fixed'] = ["".join(seq.split()) for seq in df['sequence']]
        df['seq_fixed'] = [re.sub(r"[UZOB]", "X", seq) for seq in df['seq_fixed']]
        seqs = [ list(seq)[:max_length-2] for seq in df['seq_fixed']]

        labels = list(df['label'])
        #df['label_fixed'] = ["".join(label.split()) for label in df['label']]
        #labels = [ list(label)[:max_length-2] for label in df['label_fixed']

        assert len(seqs) == len(labels)
        return seqs, labels


# In[9]:


max_length = 1024


# In[10]:


train_seqs, train_labels = load_dataset('/home/jupyter/chloro_loc/chloro_labeledsetTrain.csv', max_length)
val_seqs, val_labels = load_dataset('/home/jupyter/chloro_loc/chloro_labeledsetVal.csv', max_length)
test_seqs, test_labels = load_dataset('/home/jupyter/chloro_loc/chloro_labeledsetTest.csv', max_length)
#casp12_test_seqs, casp12_test_labels, casp12_test_disorder = load_dataset('dataset/CASP12_HHblits.csv', max_length)
#cb513_test_seqs, cb513_test_labels, cb513_test_disorder = load_dataset('dataset/CB513_HHblits.csv', max_length)
#ts115_test_seqs, ts115_test_labels, ts115_test_disorder = load_dataset('dataset/TS115_HHblits.csv', max_length)


# In[11]:


print(train_seqs[0][10:30], train_labels[0], sep='\n')


# **6. Tokenize sequences**

# In[12]:


seq_tokenizer = BertTokenizerFast.from_pretrained(model_name, do_lower_case=False)


# In[13]:


train_seqs_encodings = seq_tokenizer(train_seqs, is_split_into_words=True, return_offsets_mapping=True, truncation=True, padding=True)
val_seqs_encodings = seq_tokenizer(val_seqs, is_split_into_words=True, return_offsets_mapping=True, truncation=True, padding=True)
test_seqs_encodings = seq_tokenizer(test_seqs, is_split_into_words=True, return_offsets_mapping=True, truncation=True, padding=True)

#casp12_test_seqs_encodings = seq_tokenizer(casp12_test_seqs, is_split_into_words=True, return_offsets_mapping=True, truncation=True, padding=True)
#cb513_test_seqs_encodings = seq_tokenizer(cb513_test_seqs, is_split_into_words=True, return_offsets_mapping=True, truncation=True, padding=True)
#ts115_test_seqs_encodings = seq_tokenizer(ts115_test_seqs, is_split_into_words=True, return_offsets_mapping=True, truncation=True, padding=True)


# **7. Tokenize labels**

# In[14]:


# Consider each label as a tag for each token
unique_tags = set(train_labels)
unique_tags  = sorted(list(unique_tags))  # make the order of the labels unchanged
tag2id = {tag: id for id, tag in enumerate(unique_tags)}
id2tag = {id: tag for tag, id in tag2id.items()}


# In[15]:


id2tag
tag2id


# In[16]:


def encode_tags(tags, tag2id):

    encoded_labels = [tag2id[tag] for tag in tags]
    return encoded_labels


# In[17]:


train_labels_encodings = encode_tags(train_labels, tag2id)
val_labels_encodings = encode_tags(val_labels,  tag2id)
test_labels_encodings = encode_tags(test_labels,  tag2id)
#casp12_test_labels_encodings = encode_tags(casp12_test_labels, casp12_test_seqs_encodings)
#cb513_test_labels_encodings = encode_tags(cb513_test_labels, cb513_test_seqs_encodings)
#ts115_test_labels_encodings = encode_tags(ts115_test_labels, ts115_test_seqs_encodings)


# **8. Mask disorder tokens**

# **9. Create SS3 Dataset**

# In[18]:


class SS3Dataset(Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels)


# In[19]:


# we don't want to pass this to the model
_ = train_seqs_encodings.pop("offset_mapping")
_ = val_seqs_encodings.pop("offset_mapping")
_ = test_seqs_encodings.pop("offset_mapping")
#_ = casp12_test_seqs_encodings.pop("offset_mapping")
#_ = cb513_test_seqs_encodings.pop("offset_mapping")
#_ = ts115_test_seqs_encodings.pop("offset_mapping")


# In[20]:


train_dataset = SS3Dataset(train_seqs_encodings, train_labels_encodings)
val_dataset = SS3Dataset(val_seqs_encodings, val_labels_encodings)
test_dataset = SS3Dataset(test_seqs_encodings, test_labels_encodings)
#casp12_test_dataset = SS3Dataset(casp12_test_seqs_encodings, casp12_test_labels_encodings)
#cb513_test_dataset = SS3Dataset(cb513_test_seqs_encodings, cb513_test_labels_encodings)
#ts115_test_dataset = SS3Dataset(ts115_test_seqs_encodings, ts115_test_labels_encodings)


# **10. Define the evaluation metrics**

# In[21]:



def compute_metrics(pred):
  labels = pred.label_ids
  preds = pred.predictions.argmax(-1)
  # calculate accuracy using sklearn's function
  return {
  "accuracy" : accuracy_score(labels, preds),
  #"precision" : precision_score(labels, preds),
  #"recall" : recall_score(labels, preds),
  #"f1"  : f1_score(labels, preds)

  }

#def compute_metrics(p: EvalPrediction):
#    preds_list = p.predictions
#    out_label_list = p.label_ids
#    return {
#        "accuracy": accuracy_score(out_label_list, preds_list),
#        "precision": precision_score(out_label_list, preds_list),
#        "recall": recall_score(out_label_list, preds_list),
#        "f1": f1_score(out_label_list, preds_list),
#    }


# **11. Create the model**

# In[22]:


def model_init():
  return AutoModelForSequenceClassification.from_pretrained(model_name,
                                                         num_labels=len(unique_tags),
                                                         id2label=id2tag,
                                                         label2id=tag2id,
                                                         gradient_checkpointing=False)


# **12. Define the training args and start the trainer**

# In[23]:


training_args = TrainingArguments(
    output_dir='/home/jupyter/chloro_loc/',          # output directory
    num_train_epochs=5,              # total number of training epochs
    per_device_train_batch_size=1,   # batch size per device during training
    per_device_eval_batch_size=1,   # batch size for evaluation
    warmup_steps=200,                # number of warmup steps for learning rate scheduler
    learning_rate=3e-05,             # learning rate
    weight_decay=0.0,                # strength of weight decay
    logging_dir='./logs',            # directory for storing logs
    logging_steps=200,               # How often to print logs
    do_train=True,                   # Perform training
    do_eval=True,                    # Perform evaluation
    evaluation_strategy="epoch",     # evalute after each epoch
    gradient_accumulation_steps=32,  # total number of steps before back propagation
    fp16=True,                       # Use mixed precision
    fp16_opt_level="02",             # mixed precision mode
    run_name="prot_bert_bfd_chloro_loc_w_val",      # experiment name
    seed=3,                         # Seed for experiment reproducibility
    load_best_model_at_end=True,
    metric_for_best_model="eval_accuracy",
    greater_is_better=True,

)

import gc 

# Your code with pytorch using GPU

gc.collect() 

trainer = Trainer(
    model_init=model_init,                # the instantiated 🤗 Transformers model to be trained
    args=training_args,                   # training arguments, defined above
    train_dataset=train_dataset,          # training dataset
    eval_dataset=val_dataset,             # evaluation dataset
    compute_metrics = compute_metrics,    # evaluation metrics
)


# In[24]:


#trainer.train()


# In[25]:


trainer.train("/home/jupyter/chloro_loc/checkpoint-124")


# In[26]:


trainer.save_model('/home/jupyter/chloro_loc')


# In[27]:


seq_tokenizer.save_pretrained('/home/jupyter/chloro_loc')


# **13. Make predictions and evaluate**

# In[28]:


predictions, label_ids, metrics = trainer.predict(test_dataset)


# In[30]:


metrics


# In[56]:


idx = 2
sample_ground_truth = test_dataset[idx]['labels']
sample_predictions =  np.argmax(predictions[idx])


# In[57]:


sample_sequence = seq_tokenizer.decode(list(test_dataset[idx]['input_ids']), skip_special_tokens=True)


# In[58]:



print(sample_ground_truth)
print(sample_predictions)


# In[41]:


print("Sequence       : {} \nGround Truth is: {}\nprediction is  : {}".format(#sample_sequence,
                                                                      sample_ground_truth,
                                                                      # Remove the first token on prediction becuase its CLS token
                                                                      # and only show up to the input length
                                                                      sample_predictions))
      


# **14. Save the model**

# **15. Check Tensorboard**

# In[ ]:


get_ipython().run_line_magic('load_ext', 'tensorboard')
get_ipython().run_line_magic('tensorboard', '--logdir logs')

