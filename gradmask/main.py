import click
import gradmask.training as training
import gradmask.utils.configuration as configuration

penalise_grad_choices = ["contrast", "diff_from_ref", "nonhealthy"]

@click.group()
def run():
    pass

@run.command()
@click.option('--config', '-cgf', type=click.Path(exists=True, resolve_path=True), help='Configuration file.')
@click.option('-seed', type=int, help='Seed for split and model')
@click.option('-penalise_grad', type=str, help='penalise_grad')
@click.option('-penalise_grad_usemask', type=bool, help='penalise_grad_usemask')
@click.option('-conditional_reg', type=bool, help='conditional_reg')
@click.option('-nsamples_train', type=int, help='nsamples_train')
@click.option('-nsamples_train', type=int, help='new_size')
def train(config, seed, penalise_grad, nsamples_train, penalise_grad_usemask, conditional_reg):
    cfg = configuration.load_config(config)
    if not seed is None:
        cfg["seed"] = seed

    if not penalise_grad is None:
        cfg["penalise_grad"] = penalise_grad
        
    if not penalise_grad is None:
        cfg["penalise_grad_usemask"] = penalise_grad_usemask
    if not nsamples_train is None:
        dataset = cfg["dataset"]["train"]
        dataset[list(dataset.keys())[0]]["nsamples"] = nsamples_train
    if not new_size is None:
        dataset = cfg["dataset"]["train"]
        dataset[list(dataset.keys())[0]]["new_size"] = new_size
    if not conditional_reg is None:
        cfg["conditional_reg"] = conditional_reg

    training.train(cfg)

@run.command()
@click.option('--config', '-cgf', type=click.Path(exists=True, resolve_path=True), help='Configuration file.')
@click.option('-seed', type=int, help='Seed for split and model')
@click.option('-penalise_grad', type=str, help='penalise_grad')
@click.option('-nsamples_train', type=int, help='nsamples_train')
@click.option('--n_iter', type=int, default=10, help='Configuration file.')
@click.option('--base_estimator', type=click.Choice(["GP", "RF", "ET", "GBRT"]), default="GP", help='Estimator.')
@click.option('--n_initial_points', type=int, default=10, help='Number of evaluations of func with initialization points before approximating it with base_estimator.')
@click.option('--train_function', type=str, default="train", help='Training function to optimize over.')
@click.option('-penalise_grad_usemask', type=bool, help='penalise_grad_usemask')
@click.option('-conditional_reg', type=bool, help='conditional_reg')
@click.option('-new_size', type=int, help='new_size')
def train_skopt(config, seed, penalise_grad, nsamples_train, n_iter, base_estimator, n_initial_points, train_function,
               penalise_grad_usemask, conditional_reg, new_size):
    cfg = configuration.load_config(config)
    cfg["skopt"] = True
    if not seed is None:
        cfg["seed"] = seed

    if not penalise_grad is None:
        cfg["penalise_grad"] = penalise_grad

    if not penalise_grad is None:
        cfg["penalise_grad_usemask"] = penalise_grad_usemask
    if not nsamples_train is None:
        dataset = cfg["dataset"]["train"]
        dataset[list(dataset.keys())[0]]["nsamples"] = nsamples_train
    if not new_size is None:
        dataset = cfg["dataset"]["train"]
        dataset[list(dataset.keys())[0]]["new_size"] = new_size
    if not conditional_reg is None:
        cfg["conditional_reg"] = conditional_reg
    training.train_skopt(cfg, n_iter=n_iter,
                    base_estimator=base_estimator,
                    n_initial_points=n_initial_points,
                    random_state=seed,
                    train_function=getattr(training, train_function))

def main():
    run()

if __name__ == '__main__':
    main()

