import torch
from transformers import AutoModelForTokenClassification, AutoModelForSeq2SeqLM

model_name_path = 'translator-04-vatican-finetuned-from-350k'

if 'tagger' in model_name_path:
    model = AutoModelForTokenClassification.from_pretrained(model_name_path)
else:
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name_path)

torch.save(model, model_name_path + '.pt')