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
  summary = []
  for pgs in range(reader.getNumPages()):
    page = reader.pages[pgs]
    text = page.extract_text()
    final_text = remove_unnecessay_lines(text)
    summary.append(summarizer(final_text, max_length= 500, min_length=30, do_sample=False)[0]['summary_text'])
  summary = ' '.join(summary)
  return summary

'''
reader = PdfReader("./Stockpdf/GODREJ.pdf")
summary1 = extract_summary(reader, summarizer)
print(summary1)
sentiment = testing(text)
print(sentiment)
'''