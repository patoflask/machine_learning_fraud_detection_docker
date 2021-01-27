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
from testdata import CLES, VALEURS

CLES_LENGTH = len(CLES)

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))

class FraudPredictor(object):
    """
    This class predict the transaction class
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
