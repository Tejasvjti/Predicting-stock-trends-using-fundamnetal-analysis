"""Prediction"""
'''
pip install PyPDF2==2.10.8
pip install transformers==4.22.0
pip install datasets==2.4.0
'''
import numpy as np
import pandas as pd
import transformers
import torch
from transformers import BertTokenizer, Trainer, BertForSequenceClassification, TrainingArguments, AutoModelForSequenceClassification
from sklearn.model_selection import train_test_split
from datasets import Dataset
from sklearn.metrics import accuracy_score
torch.__version__, transformers.__version__
torch.cuda.is_available()

def testing(text):
  data=pd.DataFrame()
  data['text'] = text
  print(data)
  label_id = {0:'neutral', 1:'positive',2:'negative'}
  #tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True)
  tokenizer = BertTokenizer.from_pretrained('yiyanghkust/finbert-pretrain')
  model_loaded = AutoModelForSequenceClassification.from_pretrained("./Trained model")
  datatesting = Dataset.from_pandas(data)
  datatesting = datatesting.map(lambda e: tokenizer(e['text'], truncation=True, padding='max_length', max_length=128), batched=True)
  datatesting.set_format(type='torch', columns=['input_ids', 'token_type_ids', 'attention_mask'])
  print(datatesting)
  trainer_model = Trainer(model = model_loaded)
  testing_results = trainer_model.predict(datatesting)
  pred = testing_results
  pred_class = np.argmax(pred[0],axis=1)
  print(pred_class)
  pred_label=[]
  if pred_class[-1]==0:
    pred_label.append('Stock Price is Neutral')
  elif pred_class[-1]==1:
    pred_label.append('Stock Price will go Up')
  else:
    pred_label.append('Stock Price will go Down')
  pred_sentiment = pred_label[-1]
  return pred_sentiment

'''
text = ['cfra maintains hold on agilent technologies lowers price target to 85.']
predictions = testing(text)
print(predictions)
'''

