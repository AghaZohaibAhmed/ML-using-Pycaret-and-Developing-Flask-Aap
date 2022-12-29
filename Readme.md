Create Flask App
Step1:
Create new Virtual Environment
pip install virtualenv
activate on windows .\abc\Scripts\activate
activate on linux source abc/bin/activate
check packages names
pip freeze > requirements.txt
Install required packages
pip install flask numpy pandas gunicorn scikit-learn pickle5
Step2:
run ML_class3.ipynb -> save model file my_final_pipline
Step3: Create your Flask App
create app.py file
create folder static
create folder templates
open app.py

import pickle
from flask import Flask, render_template, request
import numpy as np
import pandas as pd
from pycaret.classification import *

app = Flask(__name__)
loaded_model = load_model("static/my_final_pipline")

@app.route("/")
def index():
    return render_template("index.html")


@app.route("/model", methods=['POST'])
def about():
    Pregnancies = np.int64(request.form['Pregnancies'])
    Glucose = np.int64(request.form['Glucose'])
    BloodPressure = np.int64(request.form['BloodPressure'])
    SkinThickness = np.int64(request.form['SkinThickness'])
    Insulin = np.int64(request.form['Insulin'])
    BMI = np.float(request.form['BMI'])
    DiabetesPedigreeFunction = np.float(request.form['DiabetesPedigreeFunction'])
    Age	= np.int64(request.form['Age'])
    
    

    df1 = pd.DataFrame([[Pregnancies, Glucose, BloodPressure, SkinThickness, Insulin, BMI, DiabetesPedigreeFunction, Age, 10]], columns=['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin','BMI', 'DiabetesPedigreeFunction', 'Age', 'Outcome'])
    print(df1)
    df2 = predict_model(loaded_model,df1)
    result = df2['Label'].values[0]

    return f"""
    Client Pregnancies {Pregnancies}<br>
    Client Glucose {Glucose}<br>
    Client BloodPressure {BloodPressure}<br>
    Client SkinThickness {SkinThickness}<br>
    Client Insulin {Insulin}<br>
    Client BMI {BMI}<br>
    Client DiabetesPedigreeFunction {DiabetesPedigreeFunction}<br>
    Client Age {Age}<br>
    <h1>Prediction of Diabetes {result}</h1>
    """  


if __name__ == '__main__':
    app.run(debug=True)
    