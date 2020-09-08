#!/usr/bin/env python
# coding: utf-8

import joblib
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

def get_sepsis_score(data, model):

    data = np.nan_to_num(data, nan=0.0)
    labels = model.predict(data)
    score = model.predict_proba(data)

    scores = list()
    for idx, each in enumerate(score):
        index = int(labels[idx])
        scores.append(each[index])

    # This is will only be called for one row at a time so driver.py
    # only expects 1 score and 1 label

    return scores[-1], labels[-1]


def load_sepsis_model():
    filename = 'rf_model.sav'
    model = joblib.load(filename)
    return model