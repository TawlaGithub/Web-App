import pandas as pd
import numpy as np
from flask import Flask, request, jsonify, render_template
import pickle

app = Flask(__name__)
model = pickle.load(open('classifier.pkl', 'rb'))

b = [0,1,2,3,4,5,6,7,8]
a = ['apple','banana','chickpea','kidneybeans','maize','mango',
     'mothbeans','mungbean','rice']

a = pd.DataFrame(a,columns=['label'])
b = pd.DataFrame(b,columns=['encoded'])
classes = pd.concat([a,b],axis=1).sort_values('encoded').set_index('label')

@app.route('/')
def welcome():
    return render_template('croppredict.html')

@app.route('/predict',methods=['POST'])
def predict():
    temp = request.form.get('Temperature')
    humid = request.form.get('Humidity')
    ph = request.form.get('PH')
    rain = request.form.get('rain_fall')
    n = request.form.get('n')
    p = request.form.get('p')
    k = request.form.get('k')
    data = [[n,p,k,temp,humid,ph,rain]]
    pred = model.predict(data)

    for i in range(0,len(classes)):
        if(classes.encoded[i]==pred):
            output = classes.index[i].upper()
    return render_template('results.html', prediction_text='Your Recomended Crop is {}'.format(output))

if __name__ == "__main__":
    app.run(debug=True)
