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
    this fucntion predict the transaction class based on json
    containing the user transaction data set:
        Example:
            given the data set
                data = {
                  'Time': 0, 'V1': -1.3598071336738, 'V2': -0.0727811733098497,
                  'V3': 2.53634673796914, 'V4':1.37815522427443,
                  'V5': -0.338320769942518, 'V6': 0.462387777762292,
                  'V7': 0.239598554061257, 'V8': 0.0986979012610507,
                  'V9': 0.363786969611213,'V10': 0.0907941719789316,
                  'V11': -0.551599533260813, 'V12': -0.617800855762348,
                  'V13': -0.991389847235408, 'V14': -0.311169353699879,
                  'V15': 1.46817697209427, 'V16': -0.470400525259478,
                  'V17': 0.207971241929242, 'V18': 0.0257905801985591,
                  'V19': 0.403992960255733, 'V20': 0.251412098239705,
                  'V21': -0.018306777944153, 'V22': 0.277837575558899,
                  'V23': -0.110473910188767, 'V24': 0.0669280749146731,
                  'V25': 0.128539358273528, 'V26': -0.189114843888824,
                  'V27': 0.133558376740387, 'V28': -0.0210530534538215,
                  'Amount': 149.62
                }

        :the model should return: 0 or 1
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