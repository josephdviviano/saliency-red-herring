#!/usr/bin/env python

import pickle
import os

def get_data(metric_file):
    with open (metric_file, "rb") as f:
        data = pickle.load(f)

    return data

if __name__ == "__main__":

    # Get all files available.
    files = []
    pwd = os.path.dirname(os.path.realpath(__file__))
    for r, d, f in os.walk(pwd):
        for fname in f:
            if "metrics.pkl" in fname:
                files.append(os.path.join(r, fname))

    sorted(files)
    # Print the results.
    for fname in files:
        data = get_data(fname)
        exp_name = data[-1]['experiment_name']
        train_auc = data[-1]['trainauc']
        valid_auc = data[-1]['validauc']
        best_valid = data[-1]['best_metric']
        test_auc = data[-1]['testauc_for_best_validauc']

        print("{}: train={}, valid={}, best_valid={}, test={}".format(
            exp_name, train_auc, valid_auc, best_valid, test_auc))

