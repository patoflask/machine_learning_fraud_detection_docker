# -*- coding: utf-8 -*-
#!usr/bin/env python3
'''
=================================================================>--Version:v1
=========================================================>Compiègne:20/01/2021
=======================================================>Fraud predictive model
================================================>done by @Manfo Satana Patrice
'''
import os
import pickle
import numpy as np

CLES =[
    "Time", "V1", "V2", "V3", "V4", "V5",  "V6",
    "V7", "V8", "V9","V10",  "V11", "V12", "V13",
    "V14", "V15",    "V16", "V17", "V18",  "V19",
    "V20",    "V21", "V22", "V23", "V24",  "V25",
    "V26", "V27", "V28", "Amount"
       ]

    
VALEURS = [0,
           -1.3598071336738,  -0.0727811733098497,
           2.53634673796914,     1.37815522427443,
           -0.338320769942518,  0.462387777762292,
           0.239598554061257,  0.0986979012610507,
           0.363786969611213,  0.0907941719789316,
           -0.551599533260813, -0.617800855762348,
           -0.991389847235408, -0.311169353699879,
           1.46817697209427,   -0.470400525259478,
           0.207971241929242,  0.0257905801985591,
           0.403992960255733,   0.251412098239705,
           -0.018306777944153,  0.277837575558899,
           -0.110473910188767, 0.0669280749146731,
           0.128539358273528,  -0.189114843888824,
           0.133558376740387, -0.0210530534538215,
           149.62]

CLES_LENGTH = len(CLES)

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))

class FraudPredictor(object):
    """This class predict the transaction class
    from the user transaction data
        @==> data : list on lenght 30
        @==>out_put:
              "1" -> frudulent transastion
              "0" -> non fraudulent transaction
    """

    def __init__(self,data):
        self.input_data_len = CLES_LENGTH
        self.MODEL_PATH = os.path.join(ROOT_DIR, 'model.pkl')
        self.prediction = self.get_prediction(data)

    def get_prediction(self,data):
        loaded_model = pickle.load(open(self.MODEL_PATH, 'rb'))
        return loaded_model.predict(self.transform_data(data))[0]
    @staticmethod
    def transform_data(data):
        return np.array(data).reshape(-1, CLES_LENGTH)
    @property
    def model_path(self):
        return self.MODEL_PATH
    @property
    def model_pred(self):
        return self.prediction
    @property
    def input_data_legnt(self):
        return self.input_data_len


if __name__ == "__main__":
    Prediction = FraudPredictor(VALEURS)
    print(Prediction.model_path)
    print(ROOT_DIR)
    print(Prediction.prediction)
    print(Prediction.input_data_len)