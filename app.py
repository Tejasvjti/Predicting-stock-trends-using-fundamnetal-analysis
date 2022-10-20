from datetime import datetime
import os
from flask import render_template,request,redirect,url_for,Flask
from pdftotext import *
from prediction import *

app = Flask(__name__,template_folder='template')

summary2 = []
@app.route('/')

@app.route('/home',methods = ['GET','POST'])
def home():
  return render_template('index.html')

@app.route("/summarize", methods = ['POST'])
def summarize():
  select = request.form.get('select1')
  select = str(select)
  reader = PdfReader("./Stockpdf/"+select+".pdf")
  summary1 = extract_summary(reader, summarizer)
  summary2.append(summary1)
  return render_template('result.html',
                               text_summary=summary1)

@app.route("/prediction_sentiment", methods = ['GET','POST'])
def prediction_sentiment():
  text = summary2
  sentiment = testing(text)
  return render_template('result.html',
                               text_sentiment=sentiment,
                               text_summary=summary2[-1])

if __name__=='__main__':
    app.run(port=8080, debug=False)

