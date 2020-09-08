#!/usr/bin/env python
# coding: utf-8


import joblib
import pandas as pd

from sklearn.ensemble import RandomForestClassifier

class ModelGenerator:
    def __init__(self, filename, output_filename):
        self.filename = filename
        self.output_filename = output_filename

    def read_data(self):
        train_a_df = pd.read_csv(self.filename, sep=',')
        train_a_df.fillna(0, inplace=True)
        x_train = train_a_df.drop(['SepsisLabel'], axis=1)
        x_train = x_train.values

        y_train = train_a_df['SepsisLabel']
        y_train = y_train.values

        return x_train, y_train

    def fit_to_model(self):
        x_train, y_train = self.read_data()
        model = RandomForestClassifier(
            n_estimators=100, criterion='gini',
            bootstrap=True, n_jobs=-1
        )

        model.fit(x_train, y_train)
        joblib.dump(model, self.output_filename)
