from flask import Flask, render_template,request, jsonify

import pandas as pd
import pickle 
import numpy as np

app = Flask(__name__,template_folder='templates')
model = pickle.load(open('model.pkl','rb'))

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict',method=['POST'])
def prediction():
    int_features = [int(x) for x in request.form.values()]
    final_features = [np.array(int_features)]
    prediction = model.predict(final_features)

    output = round(prediction[0],2)

    return render_template('predict.html', prediction_text='whether the user subscribed or not: $ {}'.format(output))

if __name__ == '__main__':
    app.run(debug=True)