import pandas as pd
import json

from datetime import datetime
from joblib import load

class Transaction():

    def __init__(self, category: str, gender: str, state: str, job: str, amt: float, city_pop: int, 
                birth_date: datetime, date_of_transaction: datetime.date, time_of_transaction: datetime.time):

        self.category = category
        self.gender = gender
        self.state = state
        self.job = job
        self.amt = amt
        self.city_pop = city_pop
        self.age_days = (date_of_transaction - birth_date).days
        self.time_in_sec = (time_of_transaction.hour * 60 + time_of_transaction.minute) * 60 + time_of_transaction.second
    
    def set_up_model(self, type_of_model):
        self.type_of_model = type_of_model
        model_random_forest = load("rsc/models/random_forest.joblib")
        model_xgboost = load("rsc/models/xgb.joblib")
        self.model = model_xgboost if type_of_model == "xgboost" else model_random_forest
    
    def to_pandas_dataframe(self):
        return pd.DataFrame([
            {
                'category': self.category,
                'gender': self.gender,
                'state': self.state,
                'job': self.job,
                'amt': self.amt,
                'city_pop': self.city_pop,
                'age_days': self.age_days,
                'time_in_sec': self.time_in_sec
            }
        ])
    
    def predict(self):
        return self.model.predict(self.to_pandas_dataframe())
    
    def predict_proba(self):
        self.predict_proba = self.model.predict_proba(self.to_pandas_dataframe()).tolist()
        return pd.DataFrame(self.predict_proba, columns=["not fraud", "fraud"])

    def create_output(self):
        dict_self = self.__dict__
        dict_self.pop("model")
        return json.dumps(dict_self, indent=4)
    