import numpy as np
import pandas as pd
from flask import Flask, render_template, request
import pickle

app = Flask(__name__,template_folder="template")

model = pickle.load(open('Thyroid Disease Detection.pkl', 'rb'))


@app.route('/', methods=['GET'])
def home():
    return render_template('index.html')


@app.route('/prediction', methods=['POST'])
def prediction():
    if request.method=='POST':
        TSH = (request.form["TSH"])
        FTI = (request.form["FTI"])
        TT4 = (request.form["TT4"])
        T3 = (request.form["T3"])
        query_hypothyroid = (request.form["query_hypothyroid"])
        on_thyroxine = (request.form["on_thyroxine"])
        sex = (request.form["sex"])
        pregnant = (request.form["pregnant"])
        psych = (request.form["psych"])
        thyroid_surgery = (request.form["thyroid_surgery"])
        goitre = (request.form['goitre'])
        arr=np.array([[TSH,FTI,TT4,T3,query_hypothyroid,on_thyroxine,sex,pregnant,psych,thyroid_surgery,goitre]])
        prediction = model.predict(arr)
    return render_template('after.html', data=prediction)


if __name__ == "__main__":
    app.run(host='127.0.0.1', port='8000')
