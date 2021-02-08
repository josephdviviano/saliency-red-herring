saliency-red-herring
--------------------

**NB: I NEED TO UPDATE THIS README**

**installation**:

`source install.sh`.


**training**:

A config file needs to be defined to run the experiments, e.g.:

```
gradmask train --config gradmask/config/mnist.yml
```

**monitoring**:

The code right now will log all experiments in the `logs/experiments.csv` file.
The time of saving, git hash, config, and best accuracy on the valid set is
saved.

**skopt**

Steps to launch bayesian hyperparameters search:
1. In your `.yml` config file, choose the parameters you want to optimize
   (i.e. learning rate).
2. Replace the value with  the search parameters. For example:
    ```
    # Optimizer
    optimizer:
      Adam:
        lr: "Real(10**-4, 10**-2, 'log-uniform')"
    ```
    search the learning rate in the range (0.01, 0.0001), on a log scale.
    [Examples.](https://scikit-optimize.github.io/#skopt.BayesSearchCV).
3. Launch your config file with `activmask train-skopt --config config/path.yml`

An config example can be found in `config/mnist_skopt.yml`
