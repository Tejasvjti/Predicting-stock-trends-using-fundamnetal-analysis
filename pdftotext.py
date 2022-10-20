from PyPDF2 import PdfReader
from transformers import pipeline
from prediction import * 

def remove_unnecessay_lines(text):

  final_text = ""

  for line in text.split('\n'):
    if len(line.split(' ')) > 1:
      final_text += line
  return final_text

summarizer = pipeline("summarization")

def extract_summary(reader, summarizer):
  count = reader.numPages
  for i in range(count):
      page = reader.getPage(i)
      text = page.extract_text()
      final_text = remove_unnecessay_lines(text)
  summary_ =  summarizer(final_text, max_length = 0.2 * len(final_text), min_length= 30, do_sample=False)[0]['summary_text']
  summary = []
  summary = summary_
  return summary

'''
reader = PdfReader("./Data/q1-2023.pdf")
summary1 = extract_summary(reader, summarizer)
summary2=[]
summary2.append(summary1)
text = summary2
sentiment = testing(text)
print(sentiment)
'''