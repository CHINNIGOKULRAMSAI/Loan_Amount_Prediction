import os
import sys
import pandas as pd
import numpy as np
import shap

from src.exception import CustomException
from src.logger import logging
from src.utils import load_object

class PredictPipeline:
    def __init__(self):
        self.model = (load_object(os.path.join("artifacts","model.pkl")))
        self.preprocessor = (load_object(os.path.join("artifacts","preprocessor.pkl")))
    def predict(self,features):
        try:
            data_scaled = self.preprocessor.transform(features)
            pred = self.model.predict(data_scaled)

            explainer = shap.TreeExplainer(self.model)
            shap_values = explainer.shap_values(data_scaled)

            feature_names = features.columns.tolist()

            explanation = sorted(
                zip(feature_names,shap_values[0]),
                key= lambda x: abs(x[1]),
                reverse=True,
            )

            return pred, explanation
        except Exception as e:
            raise CustomException(e,sys)
        
class CustomData:
    def __init__(self, loan_id, no_of_dependents, education, self_employed, income_annum, loan_amount, loan_term, cibil_score, residential_assets_value, commercial_assets_value, luxury_assets_value, bank_asset_value):
        self.loan_id = loan_id
        self.no_of_dependents = int(no_of_dependents)
        self.education = education
        self.self_employed = self_employed
        self.income_annum = float(income_annum)
        self.loan_amount = float(loan_amount)
        self.loan_term = int(loan_term)
        self.cibil_score = int(cibil_score)
        self.residential_assets_value = float(residential_assets_value)
        self.commercial_assets_value = float(commercial_assets_value)
        self.luxury_assets_value = float(luxury_assets_value)
        self.bank_asset_value = float(bank_asset_value)

    def get_data_as_dataframe(self):
        data = {
            "loan_id": [self.loan_id],
            "no_of_dependents": [self.no_of_dependents],
            "education": [self.education],
            "self_employed": [self.self_employed],
            "income_annum": [self.income_annum],
            "loan_amount": [self.loan_amount],
            "loan_term": [self.loan_term],
            "cibil_score": [self.cibil_score],
            "residential_assets_value": [self.residential_assets_value],
            "commercial_assets_value": [self.commercial_assets_value],
            "luxury_assets_value": [self.luxury_assets_value],
            "bank_asset_value": [self.bank_asset_value]
        }
        return pd.DataFrame(data)
