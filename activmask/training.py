from collections import OrderedDict
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import (average_precision_score, accuracy_score,
                             f1_score, roc_auc_score)
from skopt.space import Real, Integer, Categorical
from tqdm import tqdm
import activmask.manager.mlflow.logger as mlflow_logger
import activmask.utils.configuration as configuration
import activmask.utils.monitoring as monitoring
import copy
import logging
import numpy as np
import os
import pickle
import random
import skopt
import sys
import torch
from torch.optim.lr_scheduler import StepLR, ReduceLROnPlateau
import gc


_LOG = logging.getLogger(__name__)


def parse_dict(d_, prefix='', l=[]):
    """
    Find the keys in the config dict that are to be optimized.
    """
    # Recusively searches embedded dictionaries.
    if isinstance(d_, dict):
        for key in d_.keys():
            temp = parse_dict(d_[key], prefix + '.' + key, [])
            if temp:
                l += temp
        return l

    # Handles when a key references a list of values in the config.
    elif isinstance(d_, list):
        output_list = []
        for element in d_:
            temp = parse_dict(element, prefix + '.', [])
            if temp:
                # We want to return (key, [elem1, ..., elemn]).
                output_key, value = temp[0]
                output_list.append(value)

        if len(output_list) > 0:
            l.append((output_key, output_list))

        return l

    # Gets individual skopt settings.
    else:
        try:
            x = eval(d_)
            if isinstance(x, (Real, Integer, Categorical)):
                l.append((prefix, x))
        except:
            pass
        return l

def set_key(dic, key, value):
    """
    Aux function to set the value of a key in a dict.
    """
    k1 = key.split(".")
    k1 = list(filter(lambda l: len(l) > 0, k1))
    if len(k1) == 1:
        dic[k1[0]] = value
    else:
        set_key(dic[k1[0]], ".".join(k1[1:]), value)

def generate_config(config, args, new_values):
    """
    Given a configuration dictionary and a args, an OrderedDict containing
    skopt-chosen parameters. Handles the case where the parameters are
    passed as lists.
    """
    new_config = copy.deepcopy(config)
    idx = 0

    for key in list(args.keys()):
        key_values = []

        # Handles case when the value of this key is a list of n items.
        if isinstance(args[key], list):
            n = len(args[key])
        else:
            n = 1

        for i in range(n):
            try:
                key_values.append(new_values[idx+i].item())
            except AttributeError:
                # bool or str objects don't have item attribute
                key_values.append(new_values[idx+i])
        idx += n  # Iterate the counter.

        # Single items NOT stored as lists in the dictionary.
        if len(key_values) == 1:
            key_values = key_values[0]

        set_key(new_config, key, key_values)

    return new_config


def set_seed(seed, cuda=False):
    """
    Fix the seed for numpy, python random, and pytorch.
    Parameters
    ----------
    seed: int
        The seed to use.
    """
    print('pytorch/random seed: {}'.format(seed))

    # Numpy, python, pytorch (cpu), pytorch (gpu).
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    if cuda:
        torch.cuda.manual_seed_all(seed)


def set_seed_state(state):
    """
    Loads the saved state for python, numpy, and torch RNG.
    """
    torch.random.set_rng_state(state['torch_seed'])
    np.random.set_state(state['numpy_seed'])
    random.setstate(state['python_seed'])


def report(epoch, losses):
    """
    Contains a formatted string of all individual losses and the total
    loss for this minibatch.
    """
    message = epoch
    for key in losses.keys():
        message += " {:s}={:4.4f}".format(key, losses[key])

    message += " total={:4.4f}".format(sum(losses.values()))

    return message


def merge_dicts(dict_list):
    """Merges a list of dicts into a single dict containing the mean value."""
    out_dict = {};
    out_dict = out_dict.fromkeys(dict_list[0].keys(), [])

    for key in out_dict.keys():
        out_dict[key] = np.mean([d[key].item() for d in dict_list])

    return out_dict


def append_to_keys(dictionary, name):
    """Append name to keys."""
    keys = list(dictionary.keys())
    for key in keys:
        dictionary['{}_{}'.format(name, key)] = dictionary.pop(key)

    return dictionary


def save_results(results, output_dir):
    """
    Saves a .pkl of all training statistics of interest, the best model found,
    and the last model found.
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    output_name = "best_model_{}.pth.tar".format(results["seed"])
    torch.save(results["best_model"], os.path.join(output_dir, output_name))

    output_name = "last_model_{}.pth.tar".format(results["seed"])
    torch.save(results["last_model"], os.path.join(output_dir, output_name))

    output_name = "stats_{}.pkl".format(results["seed"])
    with open(os.path.join(output_dir, output_name), 'wb') as hdl:
        pickle.dump(results["metrics"], hdl, protocol=pickle.HIGHEST_PROTOCOL)


@mlflow_logger.log_metric('valid_loss', 'valid_score')
def evaluate_valid(model, device, data_loader, epoch, exp_name):
    """Wrapper for valid epoch, for mflow compatibility."""
    loss, score, metrics = evaluate_epoch(
        model, device, data_loader,epoch, exp_name, name='valid')
    metrics = append_to_keys(metrics, "valid")

    return(loss, score, metrics)


@mlflow_logger.log_metric('test_loss', 'test_score')
def evaluate_test(model, device, data_loader, epoch, exp_name):
    """Wrapper for test epoch, for mflow compatibility."""
    loss, score, metrics = evaluate_epoch(
        model, device, data_loader, epoch, exp_name, name='test')
    metrics = append_to_keys(metrics, "test")

    return(loss, score, metrics)


@mlflow_logger.log_metric('train_loss')
def train_epoch(model, device, train_loader, optimizer, epoch):

    model.train()

    all_loss = []
    all_losses = []

    t = tqdm(train_loader)
    for batch_idx, (X, seg, y) in enumerate(t):

        optimizer.zero_grad()

        X, y = X.to(device), y.to(device)
        # Get the inverse of the mask (1 *outside* ROI).
        # NB: If no mask for an image, the entire image should start out masked.
        #     The seg will be inverted to be all 0, and therefore the actdiff
        #     and gradmask loss will not be applied.
        seg = torch.abs(seg-1).type(torch.BoolTensor).to(device)
        X.requires_grad = True

        # Expected to return a dictionary of outputs.
        outputs = model.forward(X, seg)

        # Expected to return a dictionary of loss terms.
        losses = model.loss(y, outputs)

        # Optimization.
        loss = sum(losses.values())
        loss.backward()
        optimizer.step()
        gc.collect()

        # Reporting.
        all_loss.append(loss.cpu().data.numpy())
        all_losses.append(losses)

        t.set_description(report('train epoch {} --'.format(epoch), losses))

    all_losses = merge_dicts(all_losses)
    all_losses = append_to_keys(all_losses, "train")

    return (np.array(all_loss).mean(), all_losses)


def evaluate_epoch(model, device, data_loader, epoch, exp_name, name='epoch'):
    """Evaluates a given model on given data."""
    model.eval()

    ohe = OneHotEncoder(sparse=False, categories='auto')

    targets, predictions = [], []
    all_loss = []
    all_losses = []

    with torch.no_grad():
        for batch_idx, (X, seg, y) in enumerate(data_loader):

            X, y = X.to(device), y.to(device)
            # Get the inverse of the mask (1 *outside* ROI).
            # NB: If no mask for an image, the entire image should start out masked.
            #     The seg will be inverted to be all 0, and therefore the actdiff
            #     and gradmask loss will not be applied.
            seg = torch.abs(seg-1).type(torch.BoolTensor).to(device)
            X.requires_grad = True

            # Expected to return a dictionary of outputs.
            outputs = model.forward(X, seg)

            # Sum up batch loss.
            losses = model.loss(y, outputs)
            loss = sum(losses.values())
            all_loss.append(loss.cpu().data.numpy())
            all_losses.append(losses)

            # Save predictions: First output should be the predicted class!
            targets.append(y.cpu().data.numpy())
            predictions.append(outputs['y_pred'].cpu().data.numpy())

    # Classification metrics.
    targets_ohe = ohe.fit_transform(np.concatenate(targets).reshape(-1, 1))
    acc = accuracy_score(np.concatenate(targets),
                         np.concatenate(predictions).argmax(axis=1))
    ap = average_precision_score(targets_ohe, np.concatenate(predictions))
    f1 = f1_score(np.concatenate(targets),
                  np.concatenate(predictions).argmax(axis=1), average='micro')
    auc = roc_auc_score(targets_ohe, np.concatenate(predictions))
    metrics = {'acc': acc, 'ap': ap, 'f1': f1, 'auc': auc}

    # Official score used for monitoring performance.
    score_name, score = 'auc', auc
    all_loss = np.array(all_loss).mean()
    print('epoch {} Mean {} loss: {:.4f}, {}: {:.4f}'.format(
        epoch, name, all_loss, score_name, score))

    all_losses = merge_dicts(all_losses)
    metrics.update(all_losses)

    return (all_loss, score, metrics)


@mlflow_logger.log_experiment(nested=True)
@mlflow_logger.log_metric('best_valid_score', 'test_score', 'best_epoch')
def train(cfg, random_state=None, state=None, save_checkpoints=False,
          save_performance=True):
    """
    Trains a model on a dataset given the supplied configuration.
    save is by default True and will result in the model's performance being
    saved to a handy pickle file, as well as the best-performing model being
    saved. Set this to False when doing an outer loop of hyperparameter
    optimization.
    """
    exp_name = cfg['experiment_name']
    n_epochs = cfg['n_epochs']

    if isinstance(random_state, type(None)):
        seed = cfg['seed']
    else:
        seed = random_state

    CHECKPOINT_FREQ = 20

    if save_checkpoints:
        output_dir = os.path.join('results', cfg['experiment_name'])
        checkpoint = os.path.join(output_dir,
                                  'skopt_checkpoint_{}.pth.tar'.format(seed))
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

    # Hard-coded to cuda.
    device = 'cuda'
    #if cfg['device'] >= 0:
    #torch.cuda.set_device(cfg['device'])

    # Transforms
    tr_train = configuration.setup_transform(cfg, 'train')
    tr_valid = configuration.setup_transform(cfg, 'valid')
    tr_test = configuration.setup_transform(cfg, 'test')

    # dataset
    dataset_train = configuration.setup_dataset(cfg, 'train')(tr_train)
    dataset_valid = configuration.setup_dataset(cfg, 'valid')(tr_valid)
    dataset_test = configuration.setup_dataset(cfg, 'test')(tr_test)

    train_loader = torch.utils.data.DataLoader(dataset_train,
                                               batch_size=cfg['batch_size'],
                                               shuffle=cfg['shuffle'],
                                               num_workers=cfg['num_workers'],
                                               pin_memory=cfg['pin_memory'])
    valid_loader = torch.utils.data.DataLoader(dataset_valid,
                                               batch_size=cfg['batch_size'],
                                               shuffle=cfg['shuffle'],
                                               num_workers=cfg['num_workers'],
                                               pin_memory=cfg['pin_memory'])
    test_loader = torch.utils.data.DataLoader(dataset_test,
                                              batch_size=cfg['batch_size'],
                                              shuffle=cfg['shuffle'],
                                              num_workers=cfg['num_workers'],
                                              pin_memory=cfg['pin_memory'])

    model = configuration.setup_model(cfg).to(device)
    optim = configuration.setup_optimizer(cfg)(model.parameters())

    print('model: \n{}'.format(model))
    print('optimizer: {}'.format(optim))

    # Best_valid_score for early stopping, best_test_score reported then.
    # Case when we are not using skopt.
    best_train_loss, best_valid_loss, best_test_loss = np.inf, np.inf, np.inf
    best_train_score, best_valid_score, best_test_score = 0, 0, 0
    base_epoch, best_epoch = 0, 0
    metrics = []

    # If checkpoints exist, load them.
    if isinstance(state, dict):
        if not isinstance(state['model'], type(None)):
            model.load_state_dict(state['model'])
        if not isinstance(state['optimizer'], type(None)):
            optim.load_state_dict(state['optimizer'])
        if not isinstance(state['best_model'], type(None)):
            best_model = state['best_model']
        if not isinstance(state['stats'], type(None)):
            stats = state['stats']
            base_epoch = stats['this_epoch']
            best_train_loss = stats['best_train_loss']
            best_valid_loss = stats['best_valid_loss']
            best_test_loss = stats['best_test_loss']
            best_valid_score = stats['best_valid_score']
            best_test_score = stats['best_test_score']
            best_epoch = stats['best_epoch']
        metrics = state['metrics']

    # We're not using skop if state == None.
    else:
        state = {}

    # If this is the first epoch, reset the seed.
    if base_epoch == 0:
        set_seed(seed)
    else:
        set_seed_state(state)

    for epoch in range(base_epoch, n_epochs):

        train_loss, train_losses = train_epoch(
            model=model, device=device, optimizer=optim,
            train_loader=train_loader, epoch=epoch)

        valid_loss, valid_score, valid_metrics = evaluate_valid(
            model=model, device=device, data_loader=valid_loader, epoch=epoch,
            exp_name=exp_name)

        # Get test score if this is the best valid loss!
        if valid_loss < best_valid_loss:

            test_loss, test_score, test_metrics = evaluate_test(
                model=model, device=device, data_loader=test_loader,
                epoch=epoch, exp_name=exp_name)

            best_train_loss = train_loss        # Loss
            best_valid_loss = valid_loss        #
            best_test_loss = test_loss          #
            best_valid_score = valid_score      # Score
            best_test_score = test_score        #
            best_epoch = epoch                  # Epoch
            best_model = copy.deepcopy(model)   # Model

        # Updated every epoch.
        stats = {"this_epoch": epoch+1,
                 "train_loss": train_loss,
                 "valid_loss": valid_loss,
                 "best_train_loss": best_train_loss,
                 "best_valid_loss": best_valid_loss,
                 "best_test_loss": best_test_loss,
                 "best_valid_score": best_valid_score,
                 "best_test_score": best_test_score,
                 "best_epoch": best_epoch}

        stats.update(configuration.flatten(cfg))  # cfg settings added.
        stats.update(train_losses)   # Add in the losses from train_epoch.
        stats.update(valid_metrics)  # Add in all scores from evaluate.
        metrics.append(stats)

        if valid_loss < best_valid_loss:
            stats.update(test_metrics)   # Breaks with resume. TODO: better way?

        if save_checkpoints and epoch % CHECKPOINT_FREQ == 0:
            print('checkpointing at epoch {}'.format(epoch))
            state['base_epoch'] = epoch+1
            state['best_valid_score'] = best_valid_score
            state['best_test_score'] = best_test_score
            state['best_epoch'] = best_epoch
            state['torch_seed'] = torch.random.get_rng_state()
            state['numpy_seed'] = np.random.get_state()
            state['python_seed'] = random.getstate()
            state['optimizer'] = optim.state_dict()
            state['scheduler'] = None  # TODO
            state['model'] = model.state_dict()
            state['stats'] =  stats
            state['metrics'] = metrics
            state['best_model'] = best_model
            torch.save(state, checkpoint)

    results = {"exp_name": exp_name,
               "seed": seed,
               "metrics": metrics,
               "best_model": best_model,
               "last_model": model}

    if save_performance:
        output_dir = os.path.join('results', exp_name)
        save_results(results, output_dir)

    monitoring.log_experiment_csv(
        cfg, [best_valid_score, best_test_score, best_epoch])
    return (best_valid_score, best_test_score, best_epoch, results, state)


@mlflow_logger.log_metric('best_valid_score', 'best_test_score', 'best_epoch')
@mlflow_logger.log_experiment(nested=False)
def train_skopt(cfg, n_iter, base_estimator, n_initial_points,
                random_state=None):
    """
    Do a Bayesian hyperparameter optimization.

    :param cfg: Configuration file.
    :param n_iter: Number of Bayesien optimization steps.
    :param base_estimator: skopt Optimization procedure.
    :param n_initial_points: Number of random search before starting the optimization.
    :param random_state: seed.
    :return:
    """
    if isinstance(random_state, type(None)):
        seed = cfg['seed']
    else:
        seed = random_state

    output_dir = os.path.join('results', cfg['experiment_name'])
    checkpoint = os.path.join(output_dir,
                              'skopt_checkpoint_{}.pth.tar'.format(seed))

    if os.path.isfile(checkpoint):

        resume = True
        state = torch.load(checkpoint)
        base_iteration = state['base_iteration']
        base_epoch = state['base_epoch']
        hp_opt = state['hp_opt']
        hp_args = state['hp_args']
        this_cfg = state['this_cfg']
        set_seed_state(state)

        # Check whether we are done (edge case where code crashes at the end).
        # TODO: Assumes that we do early stopping without any patience. If we
        #       add patience this logic does not work.
        done_epochs = base_epoch == cfg['n_epochs']
        done_iterations = base_iteration == n_iter
        if done_epochs and done_iterations:
            sys.exit('Resuming a training session that is already complete!')
    else:
        resume = False
        base_iteration = 0
        best_final_acc = 0

        # Parse the parameters that we want to optimize, then flatten list.
        hp_args = OrderedDict(parse_dict(cfg))
        all_vals = []
        for val in hp_args.values():
            if isinstance(val, list):
                all_vals.extend(val)
            else:
                all_vals.append(val)

        hp_opt = skopt.Optimizer(dimensions=all_vals,
                         base_estimator=base_estimator,
                         n_initial_points=n_initial_points,
                         random_state=random_state)

        set_seed(seed)

        # best_valid and best_test score are used inside of train(), best_model
        # score is only used in train_skopt() for final model selection.
        state = {'base_iteration': 0,
                 'base_epoch': 0,
                 'best_valid_score': -np.inf,
                 'best_test_score': -np.inf,
                 'best_model_score': -np.inf,
                 'best_epoch': 0,
                 'hp_opt': hp_opt,
                 'hp_args': hp_args,
                 'base_cfg': cfg,
                 'this_cfg': None,
                 'numpy_seed': None,
                 'optimizer': None,
                 'python_seed': None,
                 'scheduler': None,
                 'model': None,
                 'stats': None,
                 'metrics': [],
                 'torch_seed': None,
                 'suggestion': None,
                 'best_model': None}

    # Do a bunch of loops.
    for iteration in range(state['base_iteration'], n_iter+1):

        # Will not be true if training crashed for an iteration.
        if isinstance(state['this_cfg'], type(None)):
            suggestion = hp_opt.ask()
            state['this_cfg'] = generate_config(cfg, hp_args, suggestion)
            state['suggestion'] = suggestion
        try:
            this_valid_score, this_test_score, this_best_epoch, results, state = train(state['this_cfg'],
                                                                                       state=state,
                                                                                       random_state=seed,
                                                                                       save_checkpoints=True,
                                                                                       save_performance=False)

            # Skopt tries to minimize the valid score, so it's inverted.
            this_metric = this_valid_score * -1
            hp_opt.tell(state['suggestion'], this_metric)
        except RuntimeError as e:
            # Something went wrong, (probably a CUDA error).
            results = {'seed': seed}
            this_metric = 0.0
            this_valid_score = 0.0
            print("Experiment failed:\n{}\nAttempting next config.".format(e))
            hp_opt.tell(suggestion, this_metric)

        if this_valid_score > state['best_model_score']:
            print("*** new best model found: score={}".format(this_valid_score))
            state['best_model_score'] = this_valid_score
            save_results(results, output_dir)

        if resume:
            resume = False

        # Checkpoint the results of this successful run.
        state['torch_seed'] = torch.random.get_rng_state()
        state['numpy_seed'] = np.random.get_state()
        state['python_seed'] = random.getstate()
        state['optimizer'] = None     # Reset for next iteration.
        state['best_valid_score'] = 0 #
        state['best_test_score'] = 0  #
        state['best_epoch'] = 0       #
        state['model'] = None         #
        state['scheduler'] = None     #
        state['stats'] = None         # Save best-performance information.
        state['metrics'] = []         # Stores per-epoch information.
        state['hp_opt'] = hp_opt      # Updated with our most recent results.
        state['base_iteration'] = iteration+1  # How many skopt runs done.

        torch.save(state, checkpoint)

    # Finally, return the inverse of the best metric.
    return(state['best_valid_score'], state['best_test_score'], state['best_epoch'])
