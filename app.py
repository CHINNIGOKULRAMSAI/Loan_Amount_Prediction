from flask import Flask,request,render_template, flash, redirect, url_for, session
from models import db,User
import os
import sys
import pandas as pd
import numpy as np

from src.pipeline.predict_pipeline import CustomData,PredictPipeline

application = Flask(__name__)
app = application

@app.route('/')
def index():
    return render_template('index.html')

app.config

@app.route('/predictdata',methods = ['GET','POST'])
def predict_datapoint():
    if request.method == 'GET':
        return render_template('home.html', confidence=None, input_summary=None)
    else:
        data = CustomData(
            loan_id = request.form.get('loan_id'),
            no_of_dependents = request.form.get('no_of_dependents'),
            education = request.form.get('education'),
            self_employed = request.form.get('self_employed'),
            income_annum = request.form.get('income_annum'),
            loan_amount = request.form.get('loan_amount'),
            loan_term = request.form.get('loan_term'),
            cibil_score = request.form.get('cibil_score'),
            residential_assets_value = request.form.get('residential_assets_value'),
            commercial_assets_value = request.form.get('commercial_assets_value'),
            luxury_assets_value = request.form.get('luxury_assets_value'),
            bank_asset_value = request.form.get('bank_asset_value'))
        
        pred_df = data.get_data_as_dataframe()
        predict_pipeline = PredictPipeline()
        results, explanation = predict_pipeline.predict(pred_df)

        return render_template('home.html', results=results[0], confidence=None, input_summary=None, explanation=explanation)
    

if __name__ == '__main__':
    app.run(host='0.0.0.0',debug=True)