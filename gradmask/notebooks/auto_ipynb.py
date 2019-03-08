def to_ipynb(func, execute=True, **cookies):
    """
    Convert a pyfunc in a `.ipynb`. Save the resulting file and it's html convertion in the running `mlflow` run.
    :param func: The function to convert. The markdown cells should be in  \"\"\"...\"\"\", and the code should inside # <code> ... # </code>
    :param execute: If we want to execute the notebook or not.
    :param cookies: The magic variable to pass to the notebook. ex: run_uuid, dataset.
    """

    try:
        import inspect
        import re
        from functools import reduce
        import nbformat
        import nbconvert
        import tempfile
        import os
        import mlflow
    except ImportError as e:
        print(e)
        print("Trying to generate a .ipynb, but some dependencies not found. Doing nothing.")
        return

    def extract_cell(block):

        # Markdown cell.
        # do a cookiecutter-type-like search and replace. Not as fancy, but it's a start.
        # To any time traveller wondering these lands, have a look at jinja: http://jinja.pocoo.org/
        if block.startswith('"'):
            cell = block[3:-3]  # We remove the '"""'
            cell = reduce(lambda my_str, kv: my_str.replace('"{{{{{}}}}}"'.format(kv[0]), str(kv[1])),
                          cookies.items(), cell)
            cell = reduce(lambda my_str, kv: my_str.replace("'{{{{{}}}}}'".format(kv[0]), str(kv[1])),
                          cookies.items(), cell)
            cell = nbformat.v4.new_markdown_cell(cell)
            return cell

        # Code cell. We return the code as is (after some clean up).
        if block.startswith("<code>"):
            cell = re.sub("^<code>", "", block)  # remove <code>
            cell = re.sub("</code>", "", cell)  # remove </code>
            cell = re.sub("^[ \t\r\f\v]{4}", "", cell, flags=re.MULTILINE)  # Remove space in front of lines (except \n)

            # place the cookies (run_uuid, etc.).
            cell = reduce(lambda my_str, kv: my_str.replace('"{{{{{}}}}}"'.format(kv[0]), str(kv[1])),
                          # if it's defined as ""
                          cookies.items(), cell)
            cell = reduce(lambda my_str, kv: my_str.replace("'{{{{{}}}}}'".format(kv[0]), str(kv[1])),
                          # if it's defined as ''
                          cookies.items(), cell)

            cell = nbformat.v4.new_code_cell(cell)

            return cell

    # All the .ipynb blocks.
    all_blocks = nbformat.v4.new_notebook()

    # Get the code. We remove the first line since it's the function header.
    code = "".join(inspect.getsourcelines(func)[0][1:])
    markdow_and_code_regex = re.compile('""".*?"""|<code>.*?</code>|<code>.*',
                                        flags=re.DOTALL)  # spot markdown and code cells.
    all_blocks['cells'] += [extract_cell(block.group()) for block in
                            markdow_and_code_regex.finditer(code)]  # create each cells.

    # Execute it.
    if execute:
        ep = nbconvert.preprocessors.ExecutePreprocessor(timeout=1000, kernel='python3')
        ep.preprocess(all_blocks, {'metadata': {'path': './'}})

    # Convert it to html.
    html_exporter = nbconvert.HTMLExporter()
    html_exporter.template_file = 'basic'
    (body, resources) = html_exporter.from_notebook_node(all_blocks)

    # Save and log the ipynb.
    tempdir = tempfile.gettempdir()
    nbformat.write(all_blocks, os.path.join(tempdir, "{}.ipynb".format(func.__name__)))
    mlflow.log_artifact(os.path.join(tempdir, "{}.ipynb".format(func.__name__)), artifact_path="ipynb")

    # Save and log the html
    with open(os.path.join(tempdir, "{}.html".format(func.__name__)), 'w') as ff:
        ff.write(body)
    mlflow.log_artifact(os.path.join(tempdir, "{}.html".format(func.__name__)), artifact_path="ipynb")


def plot_optimizer():
    """
    This jupyter notebook is taken from the scikit-optimize documentation, taken from: https://scikit-optimize.github.io/notebooks/visualizing-results.html

    Bayesian optimization or sequential model-based optimization uses a surrogate model to model the expensive to evaluate objective function func. It is this model that is used to determine at which points to evaluate the expensive objective next.

    To help understand why the optimization process is proceeding the way it is, it is useful to plot the location and order of the points at which the objective is evaluated. If everything is working as expected, early samples will be spread over the whole parameter space and later samples should cluster around the minimum.

    The plot_evaluations() function helps with visualizing the location and order in which samples are evaluated for objectives with an arbitrary number of dimensions.

    The plot_objective() function plots the partial dependence of the objective, as represented by the surrogate model, for each dimension and as pairs of the input dimensions.

    All of the minimizers implemented in skopt return an OptimizeResult instance that can be inspected. Both plot_evaluations and plot_objective are helpers that do just that.
    """

    # <code>
    get_ipython().run_line_magic('matplotlib', 'inline')
    import os
    os.environ["CUDA_VISIBLE_DEVICES"] = ""

    import mlflow.sklearn
    import skopt
    from skopt.plots import plot_evaluations, plot_objective

    # </code> <code>
    opt_results = mlflow.sklearn.load_model("'{{path}}'", run_id="'{{run_uuid}}'")
    dimensions = '{{dimensions}}'

    # </code>

    """
    # Evaluating the objective function
    Next we use an extra trees based minimizer to find one of the minima of the branin function. Then we visualize at which points the objective is being evaluated using plot_evaluations().

    plot_evaluations() creates a grid of size n_dims by n_dims. The diagonal shows histograms for each of the dimensions. In the lower triangle, a two dimensional scatter plot of all points is shown. The order in which points were evaluated is encoded in the color of each point. 
    Darker/purple colors correspond to earlier samples and lighter/yellow colors correspond to later samples. A red point shows the location of the minimum found by the optimization process.

    """

    # <code>

    _ = plot_evaluations(opt_results, dimensions=dimensions)

    # </code>

    """

    Using plot_objective() we can visualise the one dimensional partial dependence of the surrogate model for each dimension. 
    The contour plot in the bottom left corner shows the two dimensional partial dependence.


    """

    # <code>

    _ = plot_objective(opt_results, dimensions=dimensions)

    # </code>

