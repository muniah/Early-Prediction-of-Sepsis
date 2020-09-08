#!/usr/bin/env python
# coding: utf-8

import os
import numpy as np
import pandas as pd

class DataPreperator:
    def __init__(self, training_ds_dir, training_ds_output_dir):
        self.current_working_dir = os.getcwd()
        # self.training_ds_dir = "training_setA"
        self.training_ds_dir = training_ds_dir
        self.training_ds_output_dir = training_ds_output_dir
        self.training_setA_path = os.path.join(self.current_working_dir, self.training_ds_dir)
        self.files = []

    def list_files(self):
        for f in os.listdir(self.training_setA_path):
            if os.path.isfile(os.path.join(self.training_setA_path, f)):
                self.files.append(f)

    def prepare_data(self):
        self.list_files()
        dfs_a = [pd.read_csv(os.path.join(self.training_setA_path, f), sep='|') for f in self.files]
        df_a = pd.concat(dfs_a, sort=False, ignore_index=True)
        df_a.drop(['Unnamed: 0'], axis=1, inplace=True)
        df_a.to_csv(self.training_ds_output_dir, index=False)

