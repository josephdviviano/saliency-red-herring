#!/usr/bin/env python
import os
import sys
#WDIR = '/home/jdv/code/activmask/activmask'
#sys.path.insert(0, os.path.dirname(WDIR))
#os.chdir(WDIR)

import pandas as pd
from activmask.results import *
import time

def plot_incorrect_correct_split():
    NSAMPLES = 100

    render_mean_grad_wrapper(
        'synth', NSAMPLES, 'synth-seeds', 'Synthetic', size=28,
        model_name=BEST_MODEL_NAME, grad_type='integrated', plot='correct', notebook=False)

    render_mean_grad_wrapper(
        'synth', NSAMPLES, 'synth-seeds', 'Synthetic', size=28,
        model_name=BEST_MODEL_NAME, grad_type='integrated', plot='incorrect', notebook=False)

    render_mean_grad_wrapper(
        'xray', NSAMPLES,'xray-seeds', 'With SPC', size=224,
        model_name=BEST_MODEL_NAME, model_type='resnet', grad_type='integrated',
        absoloute=True,  plot='correct', notebook=False)

    render_mean_grad_wrapper(
        'xray', NSAMPLES,'xray-seeds', 'With SPC', size=224,
        model_name=BEST_MODEL_NAME, model_type='resnet', absoloute=True,
        grad_type='integrated', plot='incorrect', notebook=False)

    render_mean_grad_wrapper(
        'xray_bal', NSAMPLES,'xray-seeds', 'No SPC', size=224,
        model_name=BEST_MODEL_NAME, model_type='resnet-bal', absoloute=True,
        grad_type='integrated', plot='correct', notebook=False)

    render_mean_grad_wrapper(
        'xray_bal', NSAMPLES,'xray-seeds', 'No SPC', size=224,
        model_name=BEST_MODEL_NAME, model_type='resnet-bal', absoloute=True,
        grad_type='integrated', plot='incorrect', notebook=False)

    render_mean_grad_wrapper(
        'rsna', NSAMPLES, 'rsna-seeds', 'With VPC', size=224,
        model_name=BEST_MODEL_NAME, model_type='resnet', absoloute=True,
        crop_mask=50, grad_type='integrated', plot='correct', notebook=False)

    render_mean_grad_wrapper(
        'rsna', NSAMPLES, 'rsna-seeds', 'With VPC', size=224,
        model_name=BEST_MODEL_NAME, model_type='resnet', absoloute=True,
        crop_mask=50, grad_type='integrated', plot='incorrect', notebook=False)

    render_mean_grad_wrapper(
        'rsna_bal', NSAMPLES, 'rsna-seeds', 'No VPC', size=224,
        model_name=BEST_MODEL_NAME, model_type='resnet-bal',
        absoloute=True, crop_mask=50, grad_type='integrated', plot='correct', notebook=False)

    render_mean_grad_wrapper(
        'rsna_bal', NSAMPLES, 'rsna-seeds', 'No VPC', size=224,
        model_name=BEST_MODEL_NAME, model_type='resnet-bal',
        absoloute=True, crop_mask=50, grad_type='integrated', plot='incorrect', notebook=False)


def plot_all_methods():
    NSAMPLES = 100
    for grad_type in ['normal', 'integrated', 'occlude']:

        render_mean_grad_wrapper(
            'synth', NSAMPLES, 'synth-seeds', 'Synthetic', size=28,
             model_name=BEST_MODEL_NAME, grad_type=grad_type, notebook=False)

        render_mean_grad_wrapper(
            'xray', NSAMPLES,'xray-seeds', 'With SPC', size=224,
            model_name=BEST_MODEL_NAME, model_type='resnet', absoloute=True,
            grad_type=grad_type, notebook=False)

        render_mean_grad_wrapper(
            'xray_bal', NSAMPLES,'xray-seeds', 'No SPC', size=224,
            model_name=BEST_MODEL_NAME, model_type='resnet-bal',
            absoloute=True, grad_type=grad_type, notebook=False)

        render_mean_grad_wrapper(
            'rsna', NSAMPLES, 'rsna-seeds', 'With VPC', size=224,
            model_name=BEST_MODEL_NAME, model_type='resnet', absoloute=True,
            crop_mask=50, grad_type=grad_type, notebook=False)

        render_mean_grad_wrapper(
            'rsna_bal', NSAMPLES, 'rsna-seeds', 'No VPC', size=224,
            model_name=BEST_MODEL_NAME, model_type='resnet-bal', absoloute=True,
            crop_mask=50, grad_type=grad_type, notebook=False)


def plot_individual_images():
    NSAMPLES = 100

    render_mean_grad_wrapper(
        'synth', NSAMPLES, 'synth-seeds', 'Synthetic', size=28,
        model_name=BEST_MODEL_NAME, grad_type='integrated', individual=True,
        notebook=False)

    render_mean_grad_wrapper(
        'xray', NSAMPLES,'xray-seeds', 'With SPC', size=224,
        model_name=BEST_MODEL_NAME, model_type='resnet', grad_type='integrated',
        absoloute=True, individual=True, notebook=False)

    render_mean_grad_wrapper(
        'xray_bal', NSAMPLES,'xray-seeds', 'No SPC', size=224,
        model_name=BEST_MODEL_NAME, model_type='resnet-bal', absoloute=True,
        grad_type='integrated', individual=True, notebook=False)

    render_mean_grad_wrapper(
        'rsna', NSAMPLES, 'rsna-seeds', 'With VPC', size=224,
        model_name=BEST_MODEL_NAME, model_type='resnet', absoloute=True,
        crop_mask=50, grad_type='integrated', individual=True, notebook=False)

    render_mean_grad_wrapper(
        'rsna_bal', NSAMPLES, 'rsna-seeds', 'No VPC', size=224,
        model_name=BEST_MODEL_NAME, model_type='resnet-bal',
        absoloute=True, crop_mask=50, grad_type='integrated', individual=True,
        notebook=False)


if __name__ == "__main__":
    #plot_incorrect_correct_split()
    #plot_all_methods()
    plot_individual_images()

