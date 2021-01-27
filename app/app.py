# -*- coding: utf-8 -*-
#!usr/bin/env python3
'''
=================================================================>--Version:v1
=========================================================>Compiègne:20/01/2021
=======================================================>Fraud predictive model
================================================>done by @Manfo Satana Patrice
'''
# app.py
from flask import Flask, jsonify, request
from Projet.app.fraud_utils import FraudPredictor

app = Flask(__name__)


@app.route('/predict_fraud', methods=['POST'])
def predict():
    """
    this function predict the transaction class based on json
    data which is containings the user transaction data :
        Example:
            given the data set
                data = {
                  'Time': 0, 'V1': -1.35, 'V2': -0.072,'V3': 2.5,
                  'V4': 1.37,'V5': -0.33, 'V6': 0.46, 'V7': 0.23,
                  'V8': 0.09, 'V9': 0.36, 'V10': 0.09, 'V11': -0.55,
                  'V12': -0.617, 'V13': -0.99, 'V14': -0.31, 'V15': 1.46,
                  'V16': -0.47,'V17': 0.20, 'V18': 0.02, 'V19': 0.40,
                  'V20': 0.25, 'V21': -0.018, 'V22': 0.27, 'V23': -0.11,
                  'V24': 0.06,'V25': 0.12, 'V26': -0.18,'V27': 0.13,
                  'V28': -0.021, 'Amount': 149.62
                }

        :return 0 or 1
    """
    if request.method == 'POST':
        donnes = request.json
        if donnes:
            donnes_list = [float(elt) for elt in donnes.values()]
            try:
                prediction = FraudPredictor(donnes_list).prediction
                data = {'prediction': int(prediction)}
                return jsonify(data)

            except:
                if len(donnes_list) < 30:
                    return jsonify({"error": "the lenght of the input data \
                                    isn'tcorrect!"})
                else:
                    return jsonify({"error": "the server is updadting,\
                                    tryagain later !"})
        else:

            return jsonify({"error": "no data"})

    else:

        return jsonify({"error": "only post method is accepted"})


if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port='5004')
