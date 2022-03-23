#from transformers import AutoModelForSequenceClassification, BertTokenizer
#
#sourcename = "Rostlab/prot_t5_xxl_bfd"
#modelname = "prot_t5_xxl_bfd"
#outdir = "/scratch/gpfs/cmcwhite/hfmodels/" + modelname
#print(sourcename, modelname, outdir)
#
#tokenizer = BertTokenizer.from_pretrained(sourcename)
#tokenizer.save_pretrained(outdir)
#model = AutoModelForSequenceClassification.from_pretrained(sourcename)
#model.save_pretrained(outdir)

>>> from transformers import T5Tokenizer, T5Model
>>> tokenizer = T5Tokenizer.from_pretrained("Rostlab/prot_t5_xl_uniref50")
>>> model = T5Model.from_pretrained("Rostlab/prot_t5_xl_uniref50", cache_dir="/scratch/gpfs/cmcwhite/cache/")                                                                                         | 0.00/24.0 [00:00<?, ?B/s]
Downloading:   0%|                                                                                                                                                                                                                  | 0.00/10.5G [00:00<?, ?B/s]
Some weights of the model checkpoint at Rostlab/prot_t5_xl_uniref50 were not used when initializing T5Model: ['lm_head.weight']
- This IS expected if you are initializing T5Model from the checkpoint of a model trained on another task or with another architecture (e.g. initializing a BertForSequenceClassification model from a BertForPreTraining model).
- This IS NOT expected if you are initializing T5Model from the checkpoint of a model that you expect to be exactly identical (initializing a BertForSequenceClassification model from a BertForSequenceClassification model).
>>> 
>>> 
>>> 
>>> 
>>> 
>>> 
>>> 
>>> 
>>> tokenizer = T5Tokenizer.from_pretrained("Rostlab/prot_t5_xl_uniref50", cache_dir="/scratch/gpfs/cmcwhite/cache/")
>>> model.save_pretrained('/scratch/gpfs/cmcwhite/hfmodels/prot_t5_xl_uniref50')     



from transformers import AutoTokenizer, AutoModelForTokenClassification
  
tokenizer = AutoTokenizer.from_pretrained("Rostlab/prot_bert_bfd_ss3")
tokenizer.save_pretrained('/scratch/gpfs/cmcwhite/hfmodels/prot_bert_bfd_ss3')
model = AutoModelForTokenClassification.from_pretrained("Rostlab/prot_bert_bfd_ss3")
model.save_pretrained("/scratch/gpfs/cmcwhite/hfmodels/prot_bert_bfd_ss3")

  
#tokenizer = AlbertTokenizer.from_pretrained("Rostlab/prot_t5_xxl_bfd")
#tokenizer.save_pretrained('/scratch/gpfs/cmcwhite/hfmodels/prot_t5_xxl_bfd')
#model = AutoModel.from_pretrained("Rostlab/prot_t5_xxl_bfd")
#model.save_pretrained('/scratch/gpfs/cmcwhite/hfmodels/prot_t5_xxl_bfd')

##from transformers import AutoModelForSequenceClassification, BertTokenizer
#tokenizer = BertTokenizer.from_pretrained('Rostlab/prot_t5_xxl_bfd')
#tokenizer.save_pretrained('/scratch/gpfs/cmcwhite/hfmodels/prot_t5_xxl_bfd')
#model = AutoModelForSequenceClassification.from_pretrained('Rostlab/prot_t5_xxl_bfd')
##model.save_pretrained('/scratch/gpfs/cmcwhite/hfmodels/prot_t5_xxl_bfd')
