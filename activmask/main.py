#!/usr/bin/env python

import os
import click
import training as training
import activmask.utils.configuration as configuration


# Enable click
os.environ['LC_ALL'] = 'C.UTF-8'
os.environ['LANG'] = 'C.UTF-8'


@click.group()
def run():
    pass


@run.command()
@click.option('--config', '-cgf',
              type=click.Path(exists=True, resolve_path=True),
              help='Configuration file.')
@click.option('--seed',
              type=int, default=None,
              help='Set random state to something other than None for reproducible results.')
def train(config, seed):
    cfg = configuration.load_config(config)
    training.train(cfg, random_state=seed)


@run.command()
@click.option('--config', '-cgf',
              type=click.Path(exists=True, resolve_path=True),
              help='Configuration file.')
@click.option('--n-iter',
              type=int, default=10,
              help='Configuration file.')
@click.option('--base-estimator',
              type=click.Choice(["GP", "RF", "ET", "GBRT"]), default="GP",
              help='Estimator.')
@click.option('--seed',
              type=int, default=None,
              help='Set random state to something other than None for reproducible results.')
def train_skopt(config, base_estimator, n_initial_points, seed):
    cfg = configuration.load_config(config)
    training.train_skopt(cfg,
                         base_estimator=base_estimator,
                         random_state=seed)


def main():
    run()

if __name__ == '__main__':
    main()
