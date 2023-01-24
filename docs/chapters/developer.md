# Developers

The current section contains information that people that wish to contribute to the package. 

## Development environment

First make sure you have the latest version of the code by cloning the repository on GitHub:

```shell
git clone https://github.com/rickdkk/worklab.git
cd worklab
```

It is recommended to start with a clean environment to start development, i.e. a virtual 
environment or a conda environment. In this example we'll use conda to make ourselves an
environment:

```shell
conda create --name worklab python=3.10
conda activate worklab
```

Afterwards you can install the worklab package with all the development tools that are 
required. You can install the package in edit mode with the `-e` option:

```shell
pip install -e ".[dev]"
```

## Changing code

## Changing documentation
