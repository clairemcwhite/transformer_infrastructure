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
