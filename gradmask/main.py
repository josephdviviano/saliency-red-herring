import click
import gradmask.utils.training as training
import gradmask.utils.configuration as configuration

@click.group()
def run():
    pass

@run.command()
@click.option('--config', '-cgf', type=click.Path(exists=True, resolve_path=True), help='Configuration file.')
def train(config):
    cfg = configuration.load_config(config)
    training.train(cfg)

@run.command()
@click.option('--config', '-cgf', type=click.Path(exists=True, resolve_path=True), help='Configuration file.')
@click.option('--n-iter', type=int, default=10, help='Configuration file.')
@click.option('--base-estimator', type=click.Choice(["GP", "RF", "ET", "GBRT"]), default="GP", help='Estimator.')
@click.option('--n-initial-points', type=int, default=10, help='Number of evaluations of func with initialization points before approximating it with base_estimator.')
@click.option('--random-state', type=int, default=1234, help='Set random state to something other than None for reproducible results.')
@click.option('--train-function', type=str, default="train", help='Training function to optimize over.')
def train_skopt(config, n_iter, base_estimator, n_initial_points, random_state, train_function):
    cfg = configuration.load_config(config)
    training.train_skopt(cfg, n_iter=n_iter,
                    base_estimator=base_estimator,
                    n_initial_points=n_initial_points,
                    random_state=random_state,
                    train_function=getattr(training, train_function))

def main():
    run()

if __name__ == '__main__':
    main()

