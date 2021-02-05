#!/usr/bin/env python
from activmask.results import *
import os
import pandas as pd
import sys
import time


def get_synth(df_performance, nsamples=None):
    start_time = time.time()
    df_synth = df_experiment_filter(df_performance, 'synth-seeds')
    df_synth['Dataset'] = "Synth"
    df_synth = add_loc_scores(df_synth, 'synth', img_size=28)
    df_synth.to_pickle('notebooks/results/synth.pkl')
    print('finished SYNTH in {} MINS'.format((time.time() - start_time)/60.0))


def get_xray(df_performance, nsamples=None):
    start_time = time.time()
    df_xray = df_experiment_remover(df_experiment_filter(df_performance, 'xray-seeds_resnet'), 'resnet-bal')
    df_xray['Dataset'] = "XRay SPC"
    df_xray = add_loc_scores(df_xray, 'xray', img_size=224, nsamples=nsamples)
    df_xray.to_pickle('notebooks/results/xray.pkl')
    print('finished XRAY in {} MINS'.format((time.time() - start_time)/60.0))


def get_xray_bal(df_performance, nsamples=None):
    start_time = time.time()
    df_xray_bal = df_experiment_filter(df_performance, 'xray-seeds_resnet-bal')
    df_xray_bal['Dataset'] = "XRay No SPC"
    df_xray_bal = add_loc_scores(df_xray_bal, 'xray_bal', img_size=224, nsamples=nsamples)
    df_xray_bal.to_pickle('notebooks/results/xray_bal.pkl')
    print('finished XRAY BAL in {} MINS'.format((time.time() - start_time)/60.0))


def get_rsna(df_performance, nsamples=None):
    start_time = time.time()
    df_rsna = df_experiment_remover(df_experiment_filter(df_performance, 'rsna-seeds_resnet'), 'resnet-bal')
    df_rsna['Dataset'] = "RSNA VPC"
    df_rsna = add_loc_scores(df_rsna, 'rsna', img_size=224, nsamples=nsamples)
    df_rsna.to_pickle('notebooks/results/rsna.pkl')
    print('finished RSNA in {} MINS'.format((time.time() - start_time)/60.0))


def get_rsna_bal(df_performance, nsamples=None):
    start_time = time.time()
    df_rsna_bal = df_experiment_filter(df_performance, 'rsna-seeds_resnet-bal')
    df_rsna_bal['Dataset'] = "RSNA No VPC"
    df_rsna_bal = add_loc_scores(df_rsna_bal, 'rsna_bal', img_size=224, nsamples=nsamples)
    df_rsna_bal.to_pickle('notebooks/results/rsna_bal.pkl')
    print('finished RSNA BAL in {} MINS'.format((time.time() - start_time)/60.0))


if __name__ == "__main__":
    df_performance = get_performance_metrics(RESULTS_DIR, best=True)
    NSAMPLES = 100

    get_synth(df_performance, nsamples=NSAMPLES)
    get_xray(df_performance, nsamples=NSAMPLES)
    get_xray_bal(df_performance, nsamples=NSAMPLES)
    get_rsna(df_performance, nsamples=NSAMPLES)
    get_rsna_bal(df_performance, nsamples=NSAMPLES)

