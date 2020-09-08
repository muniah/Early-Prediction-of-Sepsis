#!/usr/bin/env python
# coding: utf-8

from data_preparation import DataPreperator
from model import ModelGenerator


if __name__ == '__main__':
    training_ds_dir = "training_setA"
    training_ds_output_dir = 'train_a_combined.csv'
    model_output_dir = 'rf_model.sav'

    print("Prepare data for generating model.")
    dp = DataPreperator(
        training_ds_dir=training_ds_dir,
        training_ds_output_dir=training_ds_output_dir)
    dp.prepare_data()
    print("Prepared data for generating model.")

    print("Starting model generation...")
    mg = ModelGenerator(training_ds_output_dir, model_output_dir)
    mg.fit_to_model()
    print("Generated model and saved into file name: ", model_output_dir)