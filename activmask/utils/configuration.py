import activmask.datasets as datasets
import activmask.models as models
import collections
import functools
import importlib
import inspect
import logging
import os
import torchvision
import yaml
_LOG = logging.getLogger(__name__)


def flatten(d, parent_key='', sep='_'):
    """
    Will flatten the dictionary d (i.e., config) with all keys merged by `sep`.
    """
    items = []
    for k, v in d.items():
        new_key = parent_key + sep + k if parent_key else k
        if isinstance(v, collections.MutableMapping):
            items.extend(flatten(v, new_key, sep=sep).items())
        else:
            items.append((new_key, v))
    return dict(items)


def merge(source, destination):
    """
    run me with nosetests --with-doctest file.py

    >>> a = { 'first' : { 'all_rows' : { 'pass' : 'dog', 'number' : '1' } } }
    >>> b = { 'first' : { 'all_rows' : { 'fail' : 'cat', 'number' : '5' } } }
    >>> merge(b, a) == { 'first' : { 'all_rows' : { 'pass' : 'dog', 'fail' : 'cat', 'number' : '5' } } }
    True
    """
    for key, value in source.items():
        if isinstance(value, dict):
            # get node or create one
            node = destination.setdefault(key, {})
            merge(value, node)
        else:
            destination[key] = value

    return destination


def load_config(config_file):
    """
    The configuration is managed in a 3-level hierarchy:
        default < base < experiment.

    The default configuration is defined below and contains some variables
    required (at a minimum) for training.py to function.

    The experiment configuration is what is passed at the command line. It
    contains experiment settings.

    The base configuration can optionally be defined in the experiment
    configuration using the key-value pair `base: filename.yml`. `filename.yml`
    is expected to be in the same folder as the experiment configuration. For
    any settings shared by the base config and the experiment config,
    training.py will obey the experiment config.
    """
    # Required by training.py to run.
    default_cfg = {'cuda': True,
                   'seed': 0,
                   'optimizer': {'Adam': {}},
                   'batch_size': 32,
                   'n_epochs': 20,
                   'patience': 20,
                   'checkpoint_freq': 20}

    # Load the experiment-level config.
    with open(config_file, 'r') as f:
        experiment_cfg = yaml.safe_load(f)

    # If it is defined, import the base-config for the experiment.
    if 'base' in experiment_cfg.keys() and experiment_cfg['base'] != None:
        basename = os.path.dirname(config_file)
        base_file = os.path.join(basename, experiment_cfg['base'])
        with open(base_file, 'r') as f:
            base_cfg = yaml.safe_load(f)
    else:
        base_cfg = {}

    full_cfg = merge(experiment_cfg, merge(base_cfg, default_cfg))
    full_cfg['experiment_name'] = os.path.basename(config_file).split('.')[0]

    return full_cfg


def get_available_classes(mod, mod_path, control_variable):
    """
    Get all classes objects available in a custom module

    :param mod: the module
    :type mod: object
    :param mod_path: path to the module
    :type mod_path: str
    :param control_variable: module specific attribute name (please refer to
                             the documentation sec XX)
    :type control_variable: str
    :return: a dictionary with the associated class objects
    :rtype: dict{str: object}
    """
    available_objects = {}
    for c in mod.__all__:
        m = importlib.import_module(mod_path + c)
        for name, obj in inspect.getmembers(m, lambda x: inspect.isclass(x) or inspect.isfunction(x)):

            if control_variable not in obj.__dict__:
                continue

            available_objects[obj.__dict__[control_variable]] = obj
    return available_objects


def setup_model(config, yaml_section='model'):
    """
    Prepare model according to config file.
    """
    available_models = get_available_classes(
        models, 'models.', '_MODEL_NAME')
    models_from_module = importlib.import_module('torchvision.models')

    if type(yaml_section) == str and yaml_section != '':
        yaml_section = [yaml_section]
    sub_section = functools.reduce(
        lambda sub_dict, key: sub_dict.get(key), yaml_section, config)

    # Allows us to optionally define models of different types.
    if not sub_section:
        return None

    model_name = list(sub_section.keys())[0]
    model_args = list(sub_section.values())[0]

    _LOG.info('Model {} with arguments {}'.format(model_name, model_args))

    if hasattr(models_from_module , model_name):
        obj = getattr(models_from_module, model_name)
    else:
        obj = available_models[model_name]

    # Create the model
    model = obj(**model_args)
    return model


def setup_dataset(config, split='train'):
    """
    Prepare data generators for training set and optionally for validation set.
    """

    available_datasets = get_available_classes(
        datasets, 'datasets.', '_DG_NAME')
    datasets_from_module = importlib.import_module('torchvision.datasets')

    dataset_name = list(config['dataset'][split].keys())[0]
    dataset_args = list(config['dataset'][split].values())[0]

    _LOG.info('Dataset {} with arguments {}'.format(dataset_name, dataset_args))
    if hasattr(datasets_from_module , dataset_name):
        obj = getattr(datasets_from_module, dataset_name)
    else:
        obj = available_datasets[dataset_name]

    dataset = lambda transform: obj(transform=transform, **dataset_args)
    return dataset


def setup_optimizer(config, yaml_section='optimizer'):
    """
    Prepare optimizer according to configuration file
    """

    optimizer_module = importlib.import_module('torch.optim')

    if type(yaml_section) == str and yaml_section != '':
        yaml_section = [yaml_section]
    sub_section = functools.reduce(
        lambda sub_dict, key: sub_dict.get(key), yaml_section, config)
    optimizer_name = list(sub_section.keys())[0]
    optimizer_args = list(sub_section.values())[0]

    _LOG.info('Optimizer {} with arguments {}'.format(optimizer_name,
                                                      optimizer_args))

    optimizer_obj = getattr(optimizer_module, optimizer_name)
    optimizer_lambda = lambda param: optimizer_obj(param, **optimizer_args)

    return optimizer_lambda


def setup_transform(config, split='train'):
    """
    Prepare transform according to configuration file
    """

    transform_module = importlib.import_module('torchvision.transforms')

    transforms = []
    for tr in config['transform'][split]:
        tr_name, tr_args = list(tr.keys())[0], list(tr.values())[0]

        _LOG.info('transform {} with arguments {}'.format(tr_name, tr_args))

        tr_obj = getattr(transform_module, tr_name)
        transforms.append(tr_obj(**tr_args))

    compose_transform = torchvision.transforms.Compose(transforms)
    return compose_transform
