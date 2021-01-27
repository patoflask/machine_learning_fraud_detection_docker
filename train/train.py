# -*- coding: utf-8 -*-
#!usr/bin/env python3
'''
=================================================================>--Version:v1
=========================================================>Compiègne:20/01/2021
=======================================================>Fraud predictive model
================================================>done by @Manfo Satana Patrice
'''
import os
import pandas as pd
import pickle
import numpy as np

# import the models libraries
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

from sklearn.metrics import accuracy_score, precision_score, f1_score
from sklearn.metrics import plot_confusion_matrix, recall_score

import matplotlib.pyplot as plt

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(os.path.dirname(ROOT_DIR), 'creditcard.csv')
MODEL_PATH = os.path.join(ROOT_DIR, '../app/model.pkl')

data  = pd.read_csv(DATA_PATH )

X_train, X_test, y_train, y_test = train_test_split(
    data.iloc[:,0:30], data.iloc[:,30], test_size=0.33, random_state=42)

# center and reduce training data set
scaler = StandardScaler()

x_train_scaled = scaler.fit_transform(X_train.astype(np.float64))
X_test_scaled = scaler.fit_transform(X_test)

log_clf = LogisticRegression()
rnd_clf = RandomForestClassifier(random_state=42)
svm_clf = SVC()
tree_clf = DecisionTreeClassifier()

#Let seet which is the best model in terms of accuracy
acuracy_values = []
for clf in (log_clf, rnd_clf, svm_clf, tree_clf):
    clf.fit(x_train_scaled, y_train)
    y_pred = clf.predict(X_test_scaled)
    print(clf.__class__.__name__, accuracy_score(y_test, y_pred))
    acuracy_values.append(accuracy_score(y_test, y_pred)) 

#  for consistency I create a series from the list.
acc_series = pd.Series(np.array(acuracy_values))
 
x_labels = ["log_clf", "rnd_clf ", "svm_clf" ,"tree_clf"]
# Plot the figure.
plt.figure(figsize=(12, 8))
ax = acc_series.plot(kind='bar')
ax.set_title('Acuracy Values VS Model Name')
ax.set_xlabel('Model Name')
ax.set_ylabel('Acuracy Value')
ax.set_xticklabels(x_labels)

rects = ax.patches

# Make some labels
for rect, label in zip(rects, acuracy_values):
    height = rect.get_height()
    ax.text(rect.get_x() + rect.get_width() / 2, height , label,
            ha='center', va='bottom')
    
plt.rcParams.update({'font.size': 10})
plt.show()

# Performances measurement on the default random forest Model
plot_confusion_matrix(rnd_clf, X_test_scaled, y_test)
plt.rcParams.update({'font.size':10})
plt.show()

print(precision_score(y_test, y_pred))
print(recall_score(y_test, y_pred))
print(f1_score(y_test, y_pred))

#let make some feature engineering
param_grid = { 
    'max_features': ['auto', 'sqrt', 'log2'],
    'criterion' :['gini', 'entropy'],
    'max_depth':[4,6,8]
}

CV_rfc = GridSearchCV(estimator=rnd_clf, param_grid=param_grid, cv=5)
CV_rfc.fit(x_train_scaled, y_train)

# best estimator {'criterion': 'gini', 'max_depth': 8, 'max_features':
# 'log2'}

#Let build the model with the optimum paramters

RandomF = RandomForestClassifier(random_state=42, max_features='log2',
                                 max_depth=8, criterion='gini'
                                 )

pipeline = Pipeline([
                    ('scale', StandardScaler()),
                    ('clf', RandomF)
                     ])

if __name__ == "__main__":
    model = pipeline.fit(X_train.astype(np.float64), y_train)
    
    pickle.dump(model, open(MODEL_PATH,'wb'))