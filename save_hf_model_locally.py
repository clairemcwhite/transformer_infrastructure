from transformers import AutoModelForSequenceClassification, BertTokenizer
tokenizer = BertTokenizer.from_pretrained('Rostlab/prot_bert_bfd')
tokenizer.save_pretrained('/home/cmcwhite/prot_bert_bfd')
model = AutoModelForSequenceClassification.from_pretrained('Rostlab/prot_bert_bfd')
model.save_pretrained('/home/cmcwhite/prot_bert_bfd')
