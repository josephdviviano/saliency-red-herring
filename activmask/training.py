from collections import OrderedDict
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import (average_precision_score, accuracy_score,
                             f1_score, roc_auc_score)
from skopt.space import Real, Integer, Categorical
from torch.optim.lr_scheduler import StepLR, ReduceLROnPlateau
from tqdm import tqdm
import activmask.manager.mlflow.logger as mlflow_logger
import activmask.utils.configuration as configuration
import activmask.utils.monitoring as monitoring
import copy
import gc
import logging
import numpy as np
import os
import pickle
import random
import skopt
import sys
import torch

# Disables all warnings ('UserWarning' during save, mostly.)
if not sys.warnoptions:
    import warnings
    warnings.simplefilter("ignore")


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


def report(base, losses, noop_losses):
    """
    Contains a formatted string of all individual op losses, noop losses, and
    the total op loss for this minibatch. "noop" losses are those that are not
    used by the main optimizer (they could be used by an internal optimizer,
    inside of the model).
    """
    assert isinstance(base, str)
    LOSS_FMT = " {:s}={:4.4f}"

    for key in losses.keys():
        base += LOSS_FMT.format(key, losses[key])

    for key in noop_losses.keys():
        base += LOSS_FMT.format(key, noop_losses[key])

    base += LOSS_FMT.format('total_op', sum(losses.values()))

    return base


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
        losses, noop_losses = model.loss(y, outputs)

        # Optimization.
        loss = sum(losses.values())
        if model.op_counter == 0:
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
        gc.collect()

        # Reporting.
        all_loss.append(loss.detach().cpu().data.numpy())
        all_losses.append({k: v.detach().cpu().data.numpy() for (k, v) in losses.items()})

        t.set_description(
            report('train epoch {} --'.format(epoch), losses, noop_losses))

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
            losses, _ = model.loss(y, outputs)
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
          save_performance=True, burn_in=5):
    """
    Trains a model on a dataset given the supplied configuration.
    save is by default True and will result in the model's performance being
    saved to a handy pickle file, as well as the best-performing model being
    saved. Set this to False when doing an outer loop of hyperparameter
    optimization.
    """
    exp_name = cfg['experiment_name']
    n_epochs = cfg['n_epochs']
    patience = cfg['patience']
    checkpoint_freq = cfg['checkpoint_freq']
    device = 'cuda'
    #if cfg['device'] >= 0:
    #torch.cuda.set_device(cfg['device'])

    if burn_in >= 1:  # Corrects for zero indexing.
        burn_in -= 1

    if isinstance(random_state, type(None)):
        seed = cfg['seed']
    else:
        seed = random_state

    if save_checkpoints:
        output_dir = os.path.join('results', cfg['experiment_name'])
        checkpoint = os.path.join(output_dir,
                                  'skopt_checkpoint_{}.pth.tar'.format(seed))
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

    # Transforms
    tr_train = configuration.setup_transform(cfg, 'train')
    tr_valid = configuration.setup_transform(cfg, 'valid')
    tr_test = configuration.setup_transform(cfg, 'test')

    # dataset
    dataset_train = configuration.setup_dataset(cfg, 'train')(tr_train)
    dataset_valid = configuration.setup_dataset(cfg, 'valid')(tr_valid)
    dataset_test = configuration.setup_dataset(cfg, 'test')(tr_test)

    loader_kwargs = {
        'batch_size': cfg['batch_size'], 'shuffle': cfg['shuffle'],
        'num_workers': cfg['num_workers'], 'pin_memory': cfg['pin_memory']}

    train_loader = torch.utils.data.DataLoader(dataset_train, **loader_kwargs)
    valid_loader = torch.utils.data.DataLoader(dataset_valid, **loader_kwargs)
    test_loader = torch.utils.data.DataLoader(dataset_test, **loader_kwargs)

    model = configuration.setup_model(cfg).to(device)
    optim = configuration.setup_optimizer(cfg)(model.encoder.parameters())

    print('model: \n{}'.format(model))
    print('optimizer: {}'.format(optim))

    # Best_valid_score for early stopping, best_test_score reported then.
    # Case when we are not using skopt.
    best_train_loss, best_valid_loss, best_test_loss = np.inf, np.inf, np.inf
    best_train_score, best_valid_score, best_test_score = -np.inf, -np.inf, -np.inf
    base_epoch, best_epoch = 0, 0
    patience_counter = 0
    best_model = None
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
        patience_counter = state['patience_counter']
        metrics = state['metrics']

    # We're not using skopt if state == None.
    else:
        state = {}

    # If this is the first epoch, reset the seed.
    if base_epoch == 0:
        set_seed(seed)
    else:
        set_seed_state(state)

    for epoch in range(base_epoch, n_epochs):

        is_burned_in = epoch >= burn_in

        train_loss, train_losses = train_epoch(
            model=model, device=device, optimizer=optim,
            train_loader=train_loader, epoch=epoch)

        valid_loss, valid_score, valid_metrics = evaluate_valid(
            model=model, device=device, data_loader=valid_loader, epoch=epoch,
            exp_name=exp_name)

        if is_burned_in:
            patience_counter += 1

        update_condition = (
            valid_loss < best_valid_loss and is_burned_in) or (epoch == burn_in)

        # Get test score if this is the best valid loss or we complete burn_in.
        if update_condition:

            test_loss, test_score, test_metrics = evaluate_test(
                model=model, device=device, data_loader=test_loader,
                epoch=epoch, exp_name=exp_name)

            best_train_loss = train_loss        # Loss
            best_valid_loss = valid_loss        #
            best_test_loss = test_loss          #
            best_valid_score = valid_score      # Score
            best_test_score = test_score        #
            best_epoch = epoch+1                # Epoch
            best_model = copy.deepcopy(model)   # Model
            patience_counter = 0                # Reset

        # Updated every epoch.
        stats = {"this_epoch": epoch+1,
                 "train_loss": train_loss,
                 "valid_loss": valid_loss,
                 "best_train_loss": best_train_loss,
                 "best_valid_loss": best_valid_loss,
                 "best_test_loss": best_test_loss,
                 "best_valid_score": best_valid_score,
                 "best_test_score": best_test_score,
                 # TODO: best_epoch and this_epoch are not aligned!
                 "best_epoch": best_epoch}

        stats.update(configuration.flatten(cfg))  # NB: always best (skopt).
        stats.update(train_losses)   # Add in the losses from train_epoch.
        stats.update(valid_metrics)  # Add in all scores from evaluate.
        metrics.append(stats)

        if update_condition:
            stats.update(test_metrics)   # Breaks with resume. TODO: better way?

        if save_checkpoints and epoch % checkpoint_freq == 0:
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
            state['patience_counter'] = patience_counter
            torch.save(state, checkpoint)

        if patience_counter >= patience:
            break

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
def train_skopt(cfg, base_estimator, random_state=None):
    """
    Do a Bayesian hyperparameter optimization.

    :param cfg: Configuration file.
    :param base_estimator: skopt Optimization procedure.
    :param random_state: seed.
    :return:
    """
    if isinstance(random_state, type(None)):
        seed = cfg['seed']
    else:
        seed = random_state
    n_iter = cfg['n_iter']
    n_initial_points = cfg['n_initial_points']

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
                 'patience_counter': 0,
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
            _train = train(state['this_cfg'], state=state, random_state=seed,
                           save_checkpoints=True, save_performance=False)
            this_v_score, this_t_score, this_best_epoch, results, state = _train

            # Skopt tries to minimize the valid score, so it's inverted.
            this_metric = this_v_score * -1
            hp_opt.tell(state['suggestion'], this_metric)
        except RuntimeError as e:
            # Something went wrong, (probably a CUDA error).
            results = {'seed': seed}
            this_metric = 0.0
            this_v_score = 0.0
            print("Experiment failed:\n{}\nAttempting next config.".format(e))
            hp_opt.tell(suggestion, this_metric)

        if this_v_score > state['best_model_score']:
            print("*** new best model found: score={}".format(this_v_score))
            state['best_model_score'] = this_v_score
            save_results(results, output_dir)

        if resume:
            resume = False

        # Checkpoint the results of this successful run and reset the state!
        state['torch_seed'] = torch.random.get_rng_state()
        state['numpy_seed'] = np.random.get_state()
        state['python_seed'] = random.getstate()
        state['base_iteration'] = iteration+1  # How many skopt runs done.
        state['hp_opt'] = hp_opt      # Updated with our most recent results.

        # Reset for the next iteration!  # TODO TEST THIS!!!
        state['optimizer'] = None
        state['best_valid_score'] = 0
        state['best_test_score'] = 0
        state['best_epoch'] = 0
        state['patience_counter'] = 0
        state['model'] = None
        state['scheduler'] = None
        state['stats'] = None         # Save best-performance information.
        state['metrics'] = []         # Stores per-epoch information.
        state['this_cfg'] = None      # TODO: write a test for this!

        torch.save(state, checkpoint)

    # Finally, return the inverse of the best metric.
    return(state['best_valid_score'], state['best_test_score'], state['best_epoch'])
