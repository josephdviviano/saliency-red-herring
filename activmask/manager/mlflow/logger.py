import git
import inspect
from copy import copy, deepcopy

KEEP = ['seed', 'optimizer', 'batch_size', 'num_epochs', 'epoch', 'transform',
        'model', 'experiment_name']

def log_experiment(nested=False):

    """
    Decorator of train functions. Will log everything in `mlflow` experiment
    manager if the correct configuration is setup. To use, simply put
    `@ai.semrep.manager.mlflow.log_experiment()` as a decorator to trains
    function. The only requirement is that the config (`cfg`) is either the
    first parameter passed, or is namned `cfg`.

    :param nested: If this is a nested experiment or not (i.e. inside an
    hyperparameter search of not).
    """

    # Flatten all the keys to have something more readable
    def flatten(d, parent_key='', sep='.'):
        items = []
        for k, v in d.items():
            new_key = parent_key + sep + k if parent_key else k
            try:
                items.extend(flatten(v, new_key, sep=sep).items())
            except:
                items.append((new_key, v))
        return dict(items)


    def inner_log(func):
        def foo(cfg, **kwargs):

            # Check if an experiment manager needs to be setup. otherwise do
            # nothing.
            if 'manager' not in cfg:
                return func(cfg, **kwargs)

            # We want to use mlflow, but it's not installed. I'm a merciful
            # god, so I'm letting the experiment run anyway. You're welcome.
            try:
                import mlflow
            except ImportError:
                print("Trying to use mlflow, but module not found. No experiment manager will be used for this experiment.")
                return func(cfg, **kwargs)

            # Set the tracking uri (i.e. were we save everything.
            # Default is '' (i.e. ./mlruns)).
            tracking_uri = cfg['manager'].get('set_tracking_uri', {}).get('uri', '')
            mlflow.set_tracking_uri(tracking_uri)

            # Create an experiment, if not already created.
            if 'set_experiment' in cfg['manager']:
                name = cfg['manager']['set_experiment']['name']
                mlflow.set_experiment(name)

            # Start a run
            # Git commit
            source_version = None
            try:
                repo = git.Repo(search_parent_directories=True)
                source_version = repo.head.object.hexsha
            except git.exc.InvalidGitRepositoryError:
                pass

            # This file.
            source_name = inspect.getabsfile(func)

            # Start the run.
            with mlflow.start_run(nested=nested):

                log_cfg = copy(cfg)

                # Clear config of unwanted categories.
                for key in cfg.keys():
                    if key not in KEEP:
                        log_cfg.pop(key)

                # Save the config.
                for key, value in flatten(log_cfg).items():

                    # A bug in mlflow, can't save empty string. I know, I know.
                    if value == '':
                        value = None

                    mlflow.log_param(key, value)

                # Saving the rest except for the checkpoint state.
                kwargs_to_log = deepcopy(kwargs)
                try:
                    kwargs_to_log.pop('state')
                except:
                    pass

                for key, value in flatten(kwargs_to_log).items():
                    if value == '':
                        value = None

                    mlflow.log_param(key, value)

                # Run the real function.
                result = func(cfg, **kwargs)

            return result
        return foo
    return inner_log


def log_metric(*keys):
    """
    Log metrics in the current `mlflow` experiment.
    :param keys: The keys of the metric to log, ex: `valid_auc`, `train_loss`.
    The order correspond to the return order.
    To use simply add `@ai.semrep.manager.mlflow.log_metric('metric1', 'metric2')` as a decorator.
    :return:
    """
    def inner_log_metric(func):

        # no keys defined, we do nothing.
        if len(keys) == 0:
            return func

        # We want to use mlflow, but it's not installed. I'm a merciful god, so I'm letting the experiment run anyway. You're welcome.
        try:
            import mlflow
        except ImportError:
            return func
        import mlflow

        def foo(*args, **kwargs):

            result = func(*args, **kwargs)
            iter_result = result # Don't want to overwrite the return

            # It's easier to only have iterables
            try:
                _ = iter(iter_result)
            except TypeError:
                iter_result = [iter_result]

            for k, v in zip(keys, iter_result):
                mlflow.log_metric(k, v)

            return result
        return foo

    return inner_log_metric