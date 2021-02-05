from activmask.datasets.synth import SyntheticDataset
from activmask.datasets.xray import JointDataset, JointXRayRSNADataset
from activmask.models.loss import compare_activations, get_grad_saliency
from activmask.models.resnet import ResNetModel
from captum.attr import (IntegratedGradients, DeepLift, GuidedBackprop,
                         Occlusion, ShapleyValueSampling)
from collections import OrderedDict
from copy import copy
from glob import glob
from skimage import io
from skimage.filters import gaussian
from textwrap import wrap
import argparse
import datetime
import gc
import itertools
import logging
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import pickle
import pprint
import random
import seaborn as sns
import sys
import time
import torch
import torch.nn as nn
import warnings
import yaml

warnings.filterwarnings("ignore")

# GLOBALS
SEEDS=[1111, 1234, 3232, 3221, 9856, 1290, 1987, 3200, 6400, 8888, 451] # 1111 replaces 3200
RESULTS_DIR = "/home/jdv/code/activmask/results"
LAST_MODEL_NAME = "last_model_1234.pth.tar"
BEST_MODEL_NAME = "best_model_1234.pth.tar"
ALPHA = 0.33
GRADMASK_THRESHOLD = 50
NSAMPLES = 50
CUDA = True
LO = np.nextafter(0, 1)
SIGMA = 1


def load_model(model_path):
    """ Loads a checkpointed model to the CPU.
    Note: this is very sensitive to the path. Check the current directory
    is set correctly in the import statements if you get 'module not found'
    errors.
    """
    return torch.load(model_path, map_location='cuda' if CUDA else 'cpu')


def get_metrics(path, best=False, last=False):
    """
    Loads the outputs of training, if best, only
    keeps best epoch for each dataframe.
    """

    def _convert_dtype(dictionary):
        """
        Converts all entries in all subdictionaries to be of datatype
        [int, float, str]. All non-matching entries are converted to str.
        """
        TYPES = [float, int, str, np.float32, np.float64, bool]

        for d in dictionary:
            if type(d) == dict:
                d = _convert_dtype(d)
            else:
                if type(dictionary[d]) not in TYPES:
                    dictionary[d] = str(dictionary[d])

    assert not all([best, last])

    all_df = []
    files = glob(os.path.join(path, "*/stats_*.pkl"))
    files.sort()

    for f in files:
        d = pickle.load(open(f,"rb"))
        _convert_dtype(d)
        d = pd.DataFrame.from_dict(d)

        if best:
            # Offset by 1 because of a bug in how "best" stats are stored.
            # Data at best_epoch are all actually the second-best epoch.
            best_epoch = d.iloc[-1]['best_epoch']
            d = d[d['this_epoch'] == best_epoch]
        elif last:
            last_epoch = d.iloc[-1]['this_epoch']
            d = d[d['this_epoch'] == last_epoch]

        d['source'] = f  # Keep track of the source file.
        all_df.append(d)

    return pd.concat(all_df)


def df_cleaner(df, keep=['auc', 'seed'], remove=[], verbose=False):
    """Selects the columns of the metrics dataframe to keep."""
    for col in df.columns:
        if not any([string in col for string in keep]):
            del df[col]
        elif any([string in col for string in remove]):
            del df[col]

    # Experiment name is determined by the configuration file used.
    experiments = df.experiment_name.unique()

    if verbose:
        print("resulting df \nshape={} tracking {} experiments, \nexperiments={}".format(
            df.shape, len(experiments), df.experiment_name.unique()))

    return(df)


def df_experiment_filter(df, name):
    """Keeps only experiments from df containing substring."""
    return df[df['experiment_name'].str.contains(name)]


def df_experiment_remover(df, name):
    """Removes all experiments from df containing substring."""
    return df[~df['experiment_name'].str.contains(name)]


def get_performance_metrics(path, best=False):
    KEEP = ["auc", "best", "seed", "epoch", "name", "source"]
    return df_cleaner(get_metrics(path, best=best), keep=KEEP)


def get_best_hyperparameters(path):
    KEEP = ['name', 'blur', 'actdiff', 'rrr', 'disc', 'gradmask', 'lr', 'type', 'acts']
    FILTER = ['loss']
    df =  df_cleaner(get_metrics(path, best=True), keep=KEEP, remove=FILTER)
    df =  df[df['experiment_name'].str.contains('search')]
    return df


def get_last_results_at_epoch(df, epoch, sig_digits=3):
    """Get the train/test/valid AUC at the final, not best, epoch."""
    #groups = ['experiment_name', 'actdiff_lambda', 'recon_lambda']
    fmt_str = "${0:." + str(sig_digits) + "f}\pm{1:." + str(sig_digits) + "f}$"

    groups = ['experiment_name']
    cols = ['train_auc', 'valid_auc', 'best_epoch']

    df = get_results_at_epoch(df, epoch, groups, cols)
    df = df.round(sig_digits)

    results = []
    for a, b in zip(df["train_auc"], df["train_auc_std"]):
        results.append(fmt_str.format(a, b))
    df['train_auc'] = results
    df = df.drop(['train_auc_std'], axis=1)

    results = []
    for a, b in zip(df["valid_auc"], df["valid_auc_std"]):
        results.append(fmt_str.format(a, b))
    df['valid_auc'] = results
    df = df.drop(['valid_auc_std'], axis=1)

    results = []
    for a, b in zip(df["best_epoch"], df["best_epoch_std"]):
        results.append(fmt_str.format(a, b))
    df['best_epoch'] = results
    df = df.drop(['best_epoch_std'], axis=1)

    return df


def make_results_table(dfs, sig_digits=3, mode='iou', count=False):
    """
    Merge the best test results for all dataframes submitted.
    Used to make a results table across datasets.
    """
    fmt_str = "${0:." + str(sig_digits) + "f}\pm{1:." + str(sig_digits) + "f}$"

    assert mode in ['iou', 'iop', 'iot']

    for i, df in enumerate(dfs):
        _df = copy(df)

        # Strip the dataset name out of the experiment name.
        name = _df['experiment_name'].iloc[0].split('_')[0]
        _df['experiment_name'] = _df['experiment_name'].str.replace('{}_'.format(name), '')

        # Reformat the table.

        cols=['best_test_score',
              '{}_normal'.format(mode),
              '{}_integrated'.format(mode),
              '{}_occlude'.format(mode)]

        _df = get_test_results(_df, cols=cols, count=count)
        _df = _df.round(sig_digits)

        # Merge mean+/-std into a single column with the experiment name.
        auc, iou_input, iou_integrated, iou_occlusion = [], [], [], []
        for a, b, c, d, e, f, g, h in zip(
            _df["best_test_score"], _df["best_test_score_std"],
            _df["{}_normal".format(mode)], _df["{}_normal_std".format(mode)],
            _df["{}_integrated".format(mode)], _df["{}_integrated_std".format(mode)],
            _df["{}_occlude".format(mode)], _df["{}_occlude_std".format(mode)]):
            auc.append(fmt_str.format(a, b))
            iou_input.append(fmt_str.format(c, d))
            iou_integrated.append(fmt_str.format(e, f))
            iou_occlusion.append(fmt_str.format(g, h))

        _df['test_auc_{}'.format(name)] = auc
        _df['test_{}_input_{}'.format(mode, name)] = iou_input
        _df['test_{}_integrated_{}'.format(mode, name)] = iou_integrated
        _df['test_{}_occlusion_{}'.format(mode, name)] = iou_occlusion

        # Drop the original columns.
        _df = _df.drop(['best_test_score', 'best_test_score_std',
                        '{}_normal'.format(mode), '{}_normal_std'.format(mode),
                        '{}_integrated'.format(mode), '{}_integrated_std'.format(mode),
                        '{}_occlude'.format(mode), '{}_occlude_std'.format(mode)], axis=1)

        # Merge the experiments.
        if i == 0:
            final_df = copy(_df)
        else:
            final_df = pd.merge(final_df, _df, on='experiment_name')

    return final_df


def df_deduplicate(df):
    return df.loc[:,~df.columns.duplicated()]


def loc_melt(df):
    return pd.melt(df.reset_index(),
                   id_vars=['experiment_name', 'test_auc'],
                   value_vars=['iop', 'iou', 'iot'],
                   var_name='Loc',
                   value_name='Score')


def get_results(df, groups, cols, count=False, mode='mean'):
    """
    Shows a reduced form of the table with mean and std values
    over experiments.
    """
    df_tmp = df.groupby(groups)[cols]
    if mode == 'mean':
        df = df_tmp.mean().join(df_tmp.std(),rsuffix='_std')
    elif mode == 'max':
        df = df_tmp.max().join(df_tmp.std(),rsuffix='_std')
    if count:
        df = df.join(df_tmp.count(), rsuffix='_count')

    return df


def get_test_results(df, cols=['best_test_score',
                               'iou_normal',
                               'iou_integrated',
                                'iou_occlude'], count=False):
    """Get the test results at the best epoch."""
    df = df_deduplicate(df)
    groups = ['experiment_name']
    return get_results(df, groups, cols, count=count, mode='mean')


def threshold(x, percentile):
    return x * (x > np.percentile(x, percentile))


def load_dataset(name, seed, nsamples=None):
    #BASE_DIR = "/srv/fast/scratch/xray"
    BASE_DIR = "/srv/data/xray"
    assert name in ['synth',
                    'msd_cardiac',
                    'msd_liver',
                    'msd_colon',
                    'msd_pancreas',
                    'xray',
                    'xray_bal',
                    'rsna',
                    'rsna_bal']
    if name == 'synth':
        return SyntheticDataset(
            dataroot="/home/jdv/code/activmask/data/synth_hard",
            mode='distractor3',
            distract_noise=1,
            nsamples=128 if not nsamples else nsamples,
            seed=seed)
    elif name == 'msd_cardiac':
        return HeartMSDDataset(
            base_path='/srv/data/msd',
            mode='test',
            nsamples=128 if not nsamples else nsamples,
            blur=0,
            seed=seed)
    elif name == 'msd_liver':
        return LiverMSDDataset(
            base_path='/srv/data/msd',
            mode='test',
            blur=0,
            nsamples=128 if not nsamples else nsamples,
            seed=seed)
    elif name == 'msd_colon':
        return ColonMSDDataset(
            base_path='/srv/data/msd',
            mode='test',
            nsamples=128 if not nsamples else nsamples,
            seed=seed)
    elif name == 'msd_pancreas':
        return PancreasMSDDataset(
            base_path='/srv/data/msd',
            mode='test',
            nsamples=128 if not nsamples else nsamples,
            seed=seed)
    elif name == 'xray':
        return JointDataset(
            BASE_DIR + "/NIH/images-224",
            BASE_DIR + "/NIH/Data_Entry_2017.csv",
            BASE_DIR + "/PC/images-224",
            BASE_DIR + "/PC/PADCHEST_chest_x_ray_images_labels_160K_01.02.19.csv",
            mode='test',
            seed=seed,
            ratio=0.9,
            nsamples=nsamples,
            new_size=224)
    elif name == 'xray_bal':
        return JointDataset(
            BASE_DIR + "/NIH/images-224",
            BASE_DIR + "/NIH/Data_Entry_2017.csv",
            BASE_DIR + "/PC/images-224",
            BASE_DIR + "/PC/PADCHEST_chest_x_ray_images_labels_160K_01.02.19.csv",
            mode='test',
            seed=seed,
            ratio=0.5,
            nsamples=nsamples,
            new_size=224)
    elif name == 'rsna':
        return JointXRayRSNADataset(
            imgpath=BASE_DIR + "/RSNA/stage_2_train_images_jpg",
            ratio=0.9,
            mode="test",
            seed=seed,
            nsamples=nsamples,
            new_size=224)
    elif name == 'rsna_bal':
        return JointXRayRSNADataset(
            imgpath=BASE_DIR + "/RSNA/stage_2_train_images_jpg",
            ratio=0.5,
            mode="test",
            seed=seed,
            nsamples=nsamples,
            new_size=224)


class ModelWrapper(nn.Module):
    """Needed for Captum."""
    def __init__(self, model):
        super(ModelWrapper, self).__init__()
        self.model = model
    def forward(self, X, seg):
        outputs = self.model(X, seg)
        return outputs['y_pred']


def get_saliency(x, y_pred, percentile, absoloute=True, blur=True):
    """
    Saliency wrapper: returns the saliency map in numpy format.
    Optionally takes the absoloute value, blurs the output and
    thresholds the mask.
    """
    # Removes channel dimension, batch dimension.
    if len(y_pred.shape) == 1:
        y_pred = y_pred.unsqueeze(0)  # Add batch dimension back.

    saliency = get_grad_saliency(x, y_pred).detach().cpu().numpy()[0][0]

    if absoloute:
        saliency = np.abs(saliency)

    if blur:
        saliency = gaussian(saliency,
                            mode='constant',
                            sigma=(SIGMA, SIGMA),
                            truncate=3.5,
                            preserve_range=True)

    if percentile > 0:
        saliency = threshold(saliency, percentile)

    return saliency


def get_captum_saliency(model, x, seg, y, mode, percentile=0, absoloute=True, blur=True, input_size=224):
    assert mode in ['integrated', 'deeplift', 'guided', 'occlude', 'shapely']

    METHODS = {'integrated': IntegratedGradients,
               'deeplift': DeepLift,
               'guided': GuidedBackprop,
               'occlude': Occlusion,
               'shapely': ShapleyValueSampling}

    if mode == 'integrated':
        kwargs = {'n_steps': 200, 'return_convergence_delta': True,
                  'internal_batch_size': 92}
    elif mode == 'deeplift':
        kwargs = {'return_convergence_delta': True}
    elif mode == 'guided':
        kwargs = {}
    elif mode == 'occlude':
        kwargs = {}
    elif mode == 'shapely':
        kwargs = {}

    captum_model = ModelWrapper(model)
    attr = METHODS[mode](captum_model)

    if mode == 'occlude':

        if input_size == 224:
            window_shape = (1, 15, 15)
        elif input_size == 28:
            window_shape = (1, 2, 2)

        results = attr.attribute(x.unsqueeze(0),
                                sliding_window_shapes=window_shape,
                                strides=window_shape,
                                target=y,
                                additional_forward_args=seg.unsqueeze(0))
    elif mode == 'shapely':
        # 224 / 16 = 14 ...
        if input_size == 224:
            feature_mask = torch.range(0, 255).reshape(16, 16).unsqueeze(0)
            feature_mask = torch.repeat_interleave(
                torch.repeat_interleave(
                    feature_mask, 14, 1), 14, 2).to(x.device)

        results = attr.attribute(x.unsqueeze(0),
                                target=y,
                                additional_forward_args=seg.unsqueeze(0),
                                feature_mask=feature_mask)
    else:
        results = attr.attribute(x.unsqueeze(0),
                                 target=y,
                                 additional_forward_args=seg.unsqueeze(0),
                                 **kwargs)


    if isinstance(results, tuple):
        (saliency), convergence_delta = results
    else:
        saliency = results

    saliency = saliency[0, 0, ...].detach().cpu().numpy()  # Remove batch and channel dimension.

    if absoloute:
        saliency = np.abs(saliency)

    if blur:
        saliency = gaussian(saliency,
                            mode='constant',
                            sigma=(SIGMA, SIGMA),
                            truncate=3.5,
                            preserve_range=True)

    if percentile > 0:
        saliency = threshold(saliency, percentile)

    return saliency


def add_loc_scores(df, dataset_name, img_size, model_type='resnet', nsamples=None):

    for grad_type in ['normal', 'integrated', 'occlude']:

        iops, ious, iots = [], [], []

        # Looping over models.
        for index, row in df.iterrows():
            gc.collect()
            source_file = row['source']
            seed = int(os.path.basename(source_file).split('_')[-1].split('.')[0])
            model_name = os.path.join(
                os.path.dirname(source_file), 'best_model_{}.pth.tar'.format(seed))
            dataset = load_dataset(dataset_name, seed, nsamples=nsamples)
            model = load_model(model_name)

            iou, iop, iot = calc_loc_scores(dataset, model, img_size, grad_type=grad_type)
            ious.append(iou)
            iops.append(iop)
            iots.append(iot)

        df['iou_{}'.format(grad_type)] = ious
        df['iop_{}'.format(grad_type)] = iops
        df['iot_{}'.format(grad_type)] = iots

    return df


def calc_loc_scores(dataset, model, img_size, absoloute=True, grad_type='normal'):
    assert grad_type in ['normal', 'integrated', 'deeplift', 'guided', 'occlude', 'shapely']

    def _get_bin_loc_scores(locs, segs):
        EPS = 10e-16
        iou = (segs & locs).sum() / ((segs | locs).sum() + EPS)
        iop = (segs & locs).sum() / (locs.sum() + EPS)
        iot = (segs & locs).sum() / (segs.sum() + EPS)

        return (iou, iop, iot)

    ious, iops, iots = [], [], []

    for sample in dataset:

        #time_start = time.time()
        x, seg, y = sample
        x = torch.tensor(x).to('cuda' if CUDA else 'cpu')
        seg = torch.tensor(seg).to('cuda' if CUDA else 'cpu')
        y = torch.tensor(y).to('cuda' if CUDA else 'cpu')

        # Only plot the positive cases.
        if y == 0:
            continue

        if seg.sum() == 0:
            continue

        # Saliency map for each image.
        percentile = (100 - ((torch.sum(seg) / img_size**2) * 100)).cpu().numpy()

        if grad_type == 'normal':
            seg = seg[0, ...]
            x_var = torch.autograd.Variable(
                torch.clone(x).unsqueeze(0), requires_grad=True)
            outputs = model(x_var, seg)
            salience_map = get_saliency(
                x_var, outputs['y_pred'], percentile, absoloute=absoloute, blur=True)

        else:
            if grad_type in ['integrated', 'deeplift', 'guided']:
                do_blur = True
            else:
                do_blur = False

            salience_map = get_captum_saliency(
                model, x, seg, y, grad_type, percentile=percentile,
                absoloute=absoloute, blur=do_blur, input_size=img_size)

        # Binarize the saliency map
        locs_bin = np.zeros((img_size, img_size)).astype(np.bool)
        idx = salience_map > 0
        locs_bin[idx] = 1
        seg = seg.cpu().numpy().astype(np.bool)

        iou, iop, iot = _get_bin_loc_scores(locs_bin, seg)
        ious.append(iou)
        iops.append(iop)
        iots.append(iot)
        #print('sample done in {} sec'.format(time.time()-time_start))

    return (np.mean(ious), np.mean(iops), np.mean(iots))


def render_mean_grad_wrapper(dataset_name, n_samples, exp_name, title, size, model_name,
                             model_type='resnet', absoloute=True, crop_mask=0,
                             grad_type='normal', plot='all', individual=False, notebook=True):

    assert grad_type in ['normal', 'integrated', 'deeplift', 'guided', 'occlude', 'shapely']
    assert plot in ['all', 'correct', 'incorrect']

    base_mdls = glob(os.path.join(RESULTS_DIR, "{}_{}/best_model_*".format(exp_name, model_type)))
    mask_mdls = glob(os.path.join(RESULTS_DIR, "{}_{}_clfshuffle/best_model_*".format(exp_name, model_type)))
    disc_mdls = glob(os.path.join(RESULTS_DIR, "{}_{}_discriminator/best_model_*".format(exp_name, model_type)))
    actd_mdls = glob(os.path.join(RESULTS_DIR, "{}_{}_actdiff/best_model_*".format(exp_name, model_type)))
    grad_mdls = glob(os.path.join(RESULTS_DIR, "{}_{}_gradmask/best_model_*".format(exp_name, model_type)))
    rrr_mdls = glob(os.path.join(RESULTS_DIR, "{}_{}_rrr/best_model_*".format(exp_name, model_type)))

    start_time = time.time()
    if individual:
        render_indv_grad(dataset_name, n_samples,
                         base_mdls, mask_mdls, disc_mdls, actd_mdls, grad_mdls, rrr_mdls,
                         title, img_size=size, absoloute=absoloute,
                         grad_type=grad_type, input_size=size, plot=plot, notebook=notebook)
    else:
        render_mean_grad(dataset_name, n_samples,
                         base_mdls, mask_mdls, disc_mdls, actd_mdls, grad_mdls, rrr_mdls,
                         title, img_size=size, absoloute=absoloute, crop_mask=crop_mask,
                         grad_type=grad_type, input_size=size, plot=plot, notebook=notebook)

    print('rendered {} {} {} in {} MIN'.format(
        dataset_name, grad_type, plot, (time.time()-start_time)/60))


def render_indv_grad(dataset_name, nsamples, base_mdls, mask_mdls, disc_mdls,
                     actd_mdls, grad_mdls, rrr_mdls, exp_name, img_size=100,
                     absoloute=True, n_rendered=5, grad_type='normal',
                     input_size=224, plot='all', individual=False, notebook=True):
    """
    Renders the mean saliency map across all inputs in the dataset, from the
    input models for visual comparison.
    """
    assert grad_type in ['normal', 'integrated', 'deeplift', 'guided', 'occlude', 'shapely']
    assert plot in ['all', 'correct', 'incorrect']

    fig, axs = plt.subplots(
        nrows=n_rendered, ncols=6, figsize=(20.5, 3.5*n_rendered), dpi=150)
    datasets = {}  # Dict to cache our datasets per seed.
    images, models = OrderedDict(), OrderedDict()
    names = ["A: Baseline", "B: Masked",
             "C: Adversarial", "D: ActDiff", "E: GradMask", "F: RRR"]

    for name in names:
        # first index: 0=image, 1=grads.
        images[name] = np.zeros((2, n_rendered, img_size, img_size))

    models['A: Baseline'] = base_mdls
    models['B: Masked'] = mask_mdls
    models['C: Adversarial'] = disc_mdls
    models['D: ActDiff'] = actd_mdls
    models['E: GradMask'] = grad_mdls
    models['F: RRR'] = rrr_mdls

    def _get_seed(filename):
        return int(os.path.splitext(
            os.path.splitext(
                os.path.basename(filename))[0])[0].split('_')[-1])

    i = 0
    for model_name, models in models.items():

        # get n_rendered unique examples, one from each model.
        idx = np.arange(nsamples)
        np.random.shuffle(idx)
        counter = 0
        i += 1
        while counter < n_rendered:

            model = models[counter]
            seed = _get_seed(model)
            model = load_model(model)

            # Fetches the dataset, either from disk or cache.
            dataset_key = "{}+{}".format(dataset_name, seed)
            if dataset_key not in datasets:
                datasets[dataset_key] = load_dataset(
                    dataset_name, seed, nsamples=nsamples)

            dataset = datasets[dataset_key]

            x, seg, y = dataset[i]
            x = torch.tensor(x).to('cuda' if CUDA else 'cpu')
            seg = torch.tensor(seg).to('cuda' if CUDA else 'cpu')
            y = torch.tensor(y).to('cuda' if CUDA else 'cpu')

            # Only plot the positive cases.
            if y == 0:
                i += 1
                continue

            # Only plot correct/incorrect images.
            if plot in ['correct', 'incorrect']:
                with torch.no_grad():
                    outputs = model(x.unsqueeze(0), seg)
                    y_hat = torch.argmax(outputs['y_pred'])
                    is_correct = y_hat == y

                    if is_correct and plot == 'incorrect':
                        i += 1
                        continue
                    elif not is_correct and plot == 'correct':
                        i += 1
                        continue

            images[model_name][0, counter, ...] = torch.clone(x).detach().cpu().numpy()[0]

            # Add the gradients for each sample.
            if grad_type == 'normal':
                x_var = torch.autograd.Variable(
                    torch.clone(x).unsqueeze(0), requires_grad=True)
                outputs = model(x_var, seg)
                images[model_name][1, counter, ...] = get_saliency(
                    x_var, outputs['y_pred'], 0, absoloute=absoloute, blur=True)

            else:
                if grad_type in ['integrated', 'deeplift', 'guided']:
                    do_blur = True
                else:
                    do_blur = False

                images[model_name][1, counter, ...] = get_captum_saliency(
                    model, x, seg, y, grad_type, percentile=0,
                    absoloute=absoloute, blur=do_blur, input_size=input_size)

                counter += 1
                i += 1

    gradient_cmap = plt.cm.jet
    gradient_cmap.set_under('k', alpha=0)

    SALIENCY_NAMES = {'normal': 'G',
                      'integrated': 'IG',
                      'occlude': 'O'}

    for i in range(len(names)):

        name = names[i]
        image = images[name]

        for j in range(n_rendered):

            if j == 0:
                axs[j][i].set_title(name, size=24)

            axs[j][i].imshow(image[0, j, ...], interpolation='none', cmap='Greys_r')

            plot_name = "{}: {}".format(exp_name, SALIENCY_NAMES[grad_type])
            if i == 0 and j == n_rendered//2:
                axs[j][i].set_ylabel(plot_name, size=24)

            # Plot gradients with image as background.
            axs[j][i].imshow(
                threshold(image[1, j, ...], GRADMASK_THRESHOLD), interpolation='none',
                cmap=gradient_cmap, clim=[LO, image[1, j, ...].max()+(2*LO)], alpha=ALPHA)

            axs[j][i].get_xaxis().set_visible(False)
            axs[j][i].get_yaxis().set_ticks([])  # So label remains.

    plt.tight_layout()
    if notebook:
        plt.show()
    else:
        plt.savefig('notebooks/img/grads_indiv_{}_{}_{}.png'.format(
            dataset_name, grad_type, plot))



def render_mean_grad(dataset_name, nsamples, base_mdls, mask_mdls, disc_mdls,
                     actd_mdls, grad_mdls, rrr_mdls, exp_name, img_size=100,
                     absoloute=True, crop_mask=0, grad_type='normal', input_size=224,
                     plot='all', individual=False, notebook=True):
    """
    Renders the mean saliency map across all inputs in the dataset, from the
    input models for visual comparison.
    """
    assert grad_type in ['normal', 'integrated', 'deeplift', 'guided', 'occlude', 'shapely']
    assert plot in ['all', 'correct', 'incorrect']

    fig, axs = plt.subplots(nrows=1, ncols=7, figsize=(24, 6), dpi=150)
    axs = axs.ravel()
    datasets = {}  # Dict to cache our datasets per seed.
    #n = len(dataset)

    images, models = OrderedDict(), OrderedDict()
    names = ["Image", "A: Baseline", "B: Masked",
             "C: Adversarial", "D: ActDiff", "E: GradMask", "F: RRR"]
    for name in names:
        images[name] = np.zeros((img_size, img_size))

    _mask = np.zeros((img_size, img_size))

    models['A: Baseline'] = base_mdls
    models['B: Masked'] = mask_mdls
    models['C: Adversarial'] = disc_mdls
    models['D: ActDiff'] = actd_mdls
    models['E: GradMask'] = grad_mdls
    models['F: RRR'] = rrr_mdls

    def _get_seed(filename):
        return int(os.path.splitext(
            os.path.splitext(
                os.path.basename(filename))[0])[0].split('_')[-1])

    for model_name, models in models.items():

        for model in models:

            seed = _get_seed(model)
            model = load_model(model)

            # Fetches the dataset, either from disk or cache.
            dataset_key = "{}+{}".format(dataset_name, seed)
            if dataset_key not in datasets:
                datasets[dataset_key] = load_dataset(
                    dataset_name, seed, nsamples=nsamples)

            dataset = datasets[dataset_key]

            for sample in dataset:

                x, seg, y = sample
                x = torch.tensor(x).to('cuda' if CUDA else 'cpu')
                seg = torch.tensor(seg).to('cuda' if CUDA else 'cpu')
                y = torch.tensor(y).to('cuda' if CUDA else 'cpu')

                # Only plot the positive cases.
                if y == 0:
                    continue

                # Only plot correct/incorrect images.
                if plot in ['correct', 'incorrect']:
                    with torch.no_grad():
                        outputs = model(x.unsqueeze(0), seg)
                        y_hat = torch.argmax(outputs['y_pred'])
                        is_correct = y_hat == y

                        if is_correct and plot == 'incorrect':
                            continue
                        elif not is_correct and plot == 'correct':
                            continue

                _mask += seg.detach().cpu().numpy()[0]
                images["Image"] += torch.clone(x).detach().cpu().numpy()[0]

                # Add the gradients for each sample.
                if grad_type == 'normal':
                    x_var = torch.autograd.Variable(
                        torch.clone(x).unsqueeze(0), requires_grad=True)
                    outputs = model(x_var, seg)
                    images[model_name] += get_saliency(
                        x_var, outputs['y_pred'], 0, absoloute=absoloute, blur=True)

                else:
                    if grad_type in ['integrated', 'deeplift', 'guided']:
                        do_blur = True
                    else:
                        do_blur = False

                    images[model_name] += get_captum_saliency(
                        model, x, seg, y, grad_type, percentile=0,
                        absoloute=absoloute, blur=do_blur, input_size=input_size)
                    #print("{} done".format(model_name))

    mask_cmap = plt.cm.Reds
    mask_cmap.set_under('k', alpha=0)
    gradient_cmap = plt.cm.jet
    gradient_cmap.set_under('k', alpha=0)

    n = len(dataset)
    _mask /= n  # Normalize by dataset size.

    if crop_mask:
        _mask *= (_mask > np.percentile(_mask, crop_mask))

    SALIENCY_NAMES = {'normal': 'G',
                      'integrated': 'IG',
                      'occlude': 'O'}

    for i, (name, image) in enumerate(images.items()):
        image /= n  # Normalize by dataset size.
        axs[i].set_title(name, size=24)
        axs[i].imshow(images["Image"], interpolation='none', cmap='Greys_r')

        # Plots the image with a transparent overlay of the mask.
        if name  == 'Image':
            axs[i].imshow(_mask, interpolation='none', cmap=mask_cmap,
                          clim=[LO, _mask.max()], alpha=ALPHA)
            plot_name = "{}: {}".format(exp_name, SALIENCY_NAMES[grad_type])
            axs[i].set_ylabel(plot_name, size=24)
        # Plot gradients with image as background.
        else:
            axs[i].imshow(
                threshold(image, GRADMASK_THRESHOLD), interpolation='none',
                cmap=gradient_cmap, clim=[LO, image.max()], alpha=ALPHA)

        axs[i].get_xaxis().set_visible(False)
        axs[i].get_yaxis().set_ticks([])  # So label remains.
        #axs[i].axis('off')

    #plt.suptitle()
    plt.tight_layout()
    if notebook:
        plt.show()
    else:
        plt.savefig('notebooks/img/grads_{}_{}_{}.png'.format(
            dataset_name, grad_type, plot))


def plot_search_space(pattern, title, seed, bal=False):
    """
    Generates skopt search results for a visualization of the
    difficulty of tuning different models.
    """

    def get_opt(state):
        """Loads a checkpoint, extracting the skopt state."""
        with open(state, 'rb') as f:
            d = torch.load(f)
        return d['hp_opt']


    def get_search_results(searches, seed, model='resnet'):
        """
        Given a list of folders, retrieve the skopt
        checkpoint and load search results.
        """
        search_results = OrderedDict()
        NAME_MAP = {'{}'.format(model): 'Baseline',
                   '{}_actdiff'.format(model): 'Actdiff',
                   '{}_clfmasked'.format(model): 'Masked',
                   '{}_discriminator'.format(model): 'Discriminator',
                   '{}_gradmask'.format(model): 'Gradmask',
                   '{}_rrr'.format(model): 'RRR'}

        searches.sort()
        for search in searches:
            state_file = os.path.join(search, 'skopt_checkpoint_{}.pth.tar'.format(seed))
            hp_opt = get_opt(state_file)

            name = '_'.join(os.path.basename(os.path.dirname(state_file)).split('_')[1:])
            if name in NAME_MAP:
                search_results[NAME_MAP[name]] = {
                    'x': hp_opt.Xi, 'y': np.abs(hp_opt.yi)}

        return search_results


    def label(x, color, label):
        """Labels an axes"""
        ax = plt.gca()
        ax.text(0, .2, label, fontweight="bold", color=color,
                ha="left", va="center", transform=ax.transAxes)

    def fix_scores(scores):
        """
        Adds a small amount of non-negative noise to all scores so that
        constant results (like all-0) are rendered.
        """
        EPS = 0.0000001
        add_noise = lambda x: x + (np.random.random(1) * EPS)
        idx = np.where(scores <= 0.99)[0]
        for i in idx:
            scores[i] = add_noise(scores[i])

        assert np.max(scores) <= 1
        assert np.min(scores) >= 0

        return scores

    SCORE_NAME = "AUC"

    # Ugly logic to handle the naming of the xray balanced task.
    searches = glob('{}_{}*'.format(pattern, 'resnet-bal' if bal else 'resnet'))
    searches.sort()

    if not bal:
        searches = list(filter(lambda x: 'bal' not in x, searches))

    search_results = get_search_results(searches, seed, model='resnet-bal' if bal else 'resnet')

    # Make a long-form dataframe.
    scores, experiment_names = [], []

    for experiment, result in search_results.items():
        aucs = np.array(result['y'])
        names = np.array([experiment] * len(aucs))

        scores.append(aucs)
        experiment_names.append(names)

    scores = fix_scores(np.concatenate(scores))
    experiment_names = np.concatenate(experiment_names)
    df = pd.DataFrame({SCORE_NAME: scores, 'name': experiment_names})

    # Plotting
    sns.set(style="white", rc={"axes.facecolor": (0, 0, 0, 0)})

    pal = sns.cubehelix_palette(len(np.unique(experiment_names)), rot=-.25, light=.5)
    g = sns.FacetGrid(df, row="name", hue='name', aspect=10, height=0.75, palette=pal)

    g.map(sns.kdeplot, SCORE_NAME, clip_on=False, shade=True, alpha=1, lw=1.5, bw=.002)
    #g.map(sns.kdeplot, "auc", clip_on=False, color="w", lw=2, bw=.002)
    g.map(plt.axhline, y=0, lw=0.1, clip_on=False)
    g.map(label, SCORE_NAME)

    g.fig.subplots_adjust(hspace=-.25)
    g.set_titles("")
    g.fig.suptitle(title)
    g.set(yticks=[])
    g.despine(bottom=True, left=True)

    plt.show()


def ax_lineplot(ax, df_filter, y_min, title, legend=False, remove=''):
    sns.set(style="white", rc={"axes.facecolor": (0, 0, 0, 0), 'legend.frameon':True})

    palette = {
        '{}'.format(df_filter): "black",
        '{}_actdiff'.format(df_filter): "red",
        '{}_clfshuffle'.format(df_filter): "blue",
        '{}_discriminator'.format(df_filter): "orange",
        '{}_gradmask'.format(df_filter): "darkviolet",
        '{}_rrr'.format(df_filter): "green"
    }

    names = ["Experiment", "Baseline", "ActDiff",
             "Masked", "Adversarial", "GradMask", "RRR"]

    _df = df_experiment_filter(get_performance_metrics(RESULTS_DIR), df_filter)
    if len(remove) > 0:
        _df = df_experiment_remover(_df, remove)

    g = sns.lineplot(
        x="this_epoch", y='valid_auc', hue='experiment_name',
        ax=ax, data=_df, palette=palette, hue_order=sorted(list(palette.keys())))
    g.set_title(title)

    if legend:
        legend = g.legend(loc='center left', bbox_to_anchor=(1, 0.5), frameon=False)
        for t, l in zip(legend.texts, names):
            t.set_text(l)
    else:
        g.get_legend().remove()

    g.set_ylim(y_min, 1.05)
    g.set_xlabel('Epoch')
    g.set_ylabel('Valid AUC')


def plot_curves(names):

    x_size = 12 if len(names) > 1 else 5.5
    y_size = 3*(len(names)//2) if len(names) > 1 else 2.5
    n_rows = len(names) // 2 if len(names) > 1 else 1
    n_cols = 2 if len(names) > 1 else 1
    legend = 1 if len(names) > 1 else 0

    fig, axs = plt.subplots(
        figsize=(x_size, y_size),
        nrows=n_rows,
        ncols=n_cols,
        sharex=True,
        sharey=False)

    # Handles length 1 lists of names.
    if not isinstance(axs, type(np.array)):
        axs = np.array(axs)

    for i, ax in enumerate(axs.ravel()):
        if len(names)-1 >= i:
            # Legend only for the first plot.
            (df_filter, title, y_min, remove) = names[i]
            ax_lineplot(ax, df_filter, y_min, title,
                        legend=i == legend, remove=remove)
        else:
            ax.set_axis_off()

