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
        sex = (request.form["sex"])
        on_thyroxine = (request.form["on_thyroxine"])
        pregnant = (request.form["pregnant"])
        thyroid_surgery = (request.form["thyroid_surgery"])
        query_hypothyroid = (request.form["query_hypothyroid"])
        goitre = (request.form['goitre'])
        psych = (request.form["psych"])
        TSH = (request.form["TSH"])
        T3 = (request.form["T3"])
        TT4 = (request.form["TT4"])
        FTI = (request.form["FTI"])
        arr=np.array([[sex,on_thyroxine,pregnant,thyroid_surgery,query_hypothyroid,goitre,psych,TSH,T3,TT4,FTI]])
        prediction = model.predict(arr)
    return render_template('after.html', data=prediction)


if __name__ == "__main__":
    app.run(host='127.0.0.1', port='8000')
