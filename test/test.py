# -*- coding: utf-8 -*-
#!usr/bin/env python3
'''
=================================================================>--Version:v1
=========================================================>Compiègne:20/01/2021
=======================================================>Fraud predictive model
================================================>done by @Manfo Satana Patrice
'''
import requests
from Projet.app.fraud_utils import CLES, VALEURS

CLES_VALEURS = dict(zip(CLES,VALEURS))

resp = requests.post("http://localhost:5004/predict_fraud", json=CLES_VALEURS)

print(resp.text)
