import importlib
import inspect
import logging
import yaml
import models as models
import datasets as datasets
import torchvision

_LOG = logging.getLogger(__name__)

def process_config(config):
    """
    Here to deconstruct the config file from nested dicts into a flat structure.
    """
    output_dict = {}
    for mode in ['train', 'valid','test']:
        for key, item in config.items():
            if type(config[key]) == dict:
                # if the 'value' is actually a dict, iterate through and 
                # collect train/valid/test values.
                try:
                    # config is of the form main_key: train/test/valid: 
                    # key_val: more_key_val_pairs
                    sub_dict = config[key][mode]
                    main_key_value = list(sub_dict.keys())[0]
                    output_dict["{}_{}".format(mode, key)] = main_key_value

                    # e.g. name of optimiser, name of dataset
                    sub_sub_dict = sub_dict[main_key_value] 
                    for k, i in sub_sub_dict.items():
                        if type(i) == float:
                            i = round(i, 4)
                        # so we don't have e.g. train_dataset_MSD_mode.
                        output_dict["{}_{}_{}".format(mode, key, k)] = i 
                except:
                    # config is of the form main_key: 
                    # key_val: more_key_val_pairs e.g. optimiser: Adam: lr: 0.1
                    sub_dict = config[key]
                    main_key_value = list(sub_dict.keys())[0]
                    output_dict[key] = main_key_value
                    # e.g. name of optimiser, name of dataset.
                    sub_sub_dict = sub_dict[main_key_value] 

                    for k, i in sub_sub_dict.items():
                        if type(i) == float:
                            i = round(i, 4)
                        output_dict["{}_{}".format(key, k)] = i
            else:
                # standard key: val pair
                if type(item) == float:
                    item = round(item, 4)
                output_dict[key] = item
    return output_dict

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
    default_cfg = {'cuda': True,
                   'seed': 0,
                   'optimizer': {'Adam': {}},
                   'batch_size': 32,
                   'num_epochs': 10}

    # Load the experiment-level config.
    with open(config_file, 'r') as f:
        yaml_cfg = yaml.load(f)

    # If it is defined, import the base-config for the experiment.
    if 'base' in experiment_cfg.keys() and experiment_cfg['base'] != None:
        basename = os.path.dirname(config_file)
        base_file = os.path.join(basename, experiment_cfg['base'])
        with open(base_file, 'r') as f:
            base_cfg = yaml.load(f)
    else:
        base_cfg = {}

    return {**default_cfg, **base_cfg, **experiment_cfg}


def get_available_classes(mod, mod_path, control_variable):
    """
    Get all classes objects available in a custom module

    :param mod: the module
    :type mod: object
    :param mod_path: path to the module
    :type mod_path: str
    :param control_variable: module specific attribute name (please refer to the documentation sec XX)
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

def setup_dataset(config, split='train'):
    """
    Prepare data generators for training set and optionally for validation set
    """

    available_datasets = get_available_classes(datasets, 'datasets.', '_DG_NAME')
    datasets_from_module = importlib.import_module('torchvision.datasets')

    dataset_name = list(config['dataset'][split].keys())[0]
    dataset_args = list(config['dataset'][split].values())[0]

    _LOG.info('Dataset {} with arguments {}'.format(dataset_name, dataset_args))
    if hasattr(datasets_from_module , dataset_name):
        obj = getattr(datasets_from_module, dataset_name)
    else:
        obj = available_datasets[dataset_name]

    #dataset_args['nsamples'] = config['nsamples']
    #dataset_args['seed'] = config['seed']
    #dataset_args['blur'] = config['blur']
    #dataset_args['maxmasks'] = config['maxmasks']
    print("dataset_args:", dataset_args)
    dataset = lambda transform: obj(transform=transform, **dataset_args)
    return dataset


def setup_model(config, yaml_section='model'):
    """
    Prepare model according to config file
    """

    available_models = get_available_classes(models, 'models.', '_MODEL_NAME')
    models_from_module = importlib.import_module('torchvision.models')

    model_name = list(config[yaml_section].keys())[0]
    model_args = list(config[yaml_section].values())[0]

    _LOG.info('Model {} with arguments {}'.format(model_name, model_args))

    if hasattr(models_from_module , model_name):
        obj = getattr(models_from_module, model_name)
    else:
        obj = available_models[model_name]

    model = obj(**model_args)
    return model


def setup_optimizer(config, yaml_section='optimizer'):
    """
    Prepare optimizer according to configuration file
    """

    optimizer_module = importlib.import_module('torch.optim')

    optimizer_name = list(config[yaml_section].keys())[0]
    optimizer_args = list(config[yaml_section].values())[0]

    _LOG.info('Optimizer {} with arguments {}'.format(optimizer_name, optimizer_args))

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

