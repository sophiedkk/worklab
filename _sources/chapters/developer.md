# Developers

The current section contains information that people that wish to contribute to the package. I you just with to use the
package you can skip this section.

## Development environment

First make sure you have the latest version of the code by cloning the repository on GitHub:

```shell
git clone https://github.com/rickdkk/worklab.git
cd worklab
```

It is recommended to start with a clean environment to start development, i.e. a virtual environment or a conda 
environment. In this example we'll use conda to make ourselves an environment:

```shell
conda create --name worklab python=3.10
conda activate worklab
```

Afterwards you can install the worklab package with all the development tools (by specifying `[dev]` in pip) that are 
required. You can install the package in edit mode with the `-e` option:

```shell
pip install -e ".[dev]"
```

Any changes that you apply to the package in edit mode will be reflected in your environment. Alternatively, you can 
develop in-tree, which admittedly reduces complexity a little.

## Changing code

Please remind me to make a pre-commit hook for these steps one day. Worklab uses `black` to format its code. The CI/CD 
will fail on inconsistencies with the black codestyle. To format the code using black you can run (from the root folder):

:::{margin} Good Practice 👍
You can add the black formatter to your IDE and format on save. That way you don't need to worry about formatting!
:::

```shell
black worklab
```

You don't need to add any optional parameters as those are stored in the `pyproject.toml`. Additionally, `flake8` is 
used to check for code-style consistency. Again, this is done with a simple command:

```shell
flake8 worklab
```

Finally, the code is tested for errors with Pytest. Due to the history of the project the coverage is very low. 
Nevertheless, any new code that will be added should also include test cases to ensure proper execution. You can run
all tests by simply calling the `pytest` command from the terminal.

## Changing documentation

The project uses [Jupyter Book](https://jupyterbook.org) to build the documentation. You can choose to write documentation 
in a markdown file (.md) or in Jupyter Notebook (.ipynb). The documentation is automatically published when a new release 
is made on GitHub. To preview your changes you can develop locally with Jupyter Book by calling it from the command line:

```shell
cd docs
jupyter-book build --all .
```

The documentation can be found in `./_build/html/`. You can open `README.html` in your browser to examine the changes 
you made.

If you add a file to the documentation it should also be added in the table of contents `_toc.yml`. Don't worry, if you
forget to add the file jupyter-book will complain that it hasn't been added. In general, you shouldn't get any warnings
when building the book.

# Submitting a change

