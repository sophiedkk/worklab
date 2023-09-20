# Quick Start Guide

The worklab package makes use of the Scientific Python ecosystem. The original goal of the project was to make analysis 
of wheelchair biomechanics data more accessible and transparent. Previously, all analyses were performed with commercial 
software that is not available to everyone, especially to people not associated with a university. The scripts were then
shared by e-mail or with a USB-drive. Having the analysis in Python makes it accessible and more readable (hopefully) 
for everyone. By sharing the code I hope to be transparent and to reduce the amount of times this code has to be written 
by other people.

## Intended audience

People working in our lab that want to work with data from any of our instruments. It can, of course, also be used by
other people, provided that you have similar equipment or access to data. Also have a look at the
[examples](./examples).

## Requirements

The worklab package uses the Python programming language. You need to install Python from their [website](https://www.python.org/)
or a scientific Python distribution like [Anaconda](https://www.anaconda.com/) (recommended). 

## Installation

The package is now available on the Python Package Index (PyPI, [link](https://pypi.org/project/worklab/)), which means 
you can install it with pip from your terminal:

```shell
pip install worklab
```

:::{margin} Good Practice 👍
Don't install packages in your global (system) Python, but use a separate environment (e.g. using conda or venv)!
:::

To verify if everything works simply try to import worklab in Python:

```python
import worklab as wl
```

If that returns no errors you are (probably) good to go. That's it.

## Modules

- **ana:** Contains functions to analyse submax, isometric force, wingate tests and more.
- **com:** Provides functions for reading and writing data, use `load` to infer filetype and automatically read it. If 
you use a different naming scheme you can always call the specific load functions.
- **kinetics:** Contains all essentials for measurement wheel and ergometer data. You only need the top-level function 
`auto_process` for most use-cases.
- **move:** Contains kinematics and movement related functions for NGIMU and some functions for 3D kinematics.
- **physio:** Contains physiological calculations, which for now is basically nothing as the spirometer does everything
for you. Might include EMG and the likes later though.
- **plots:** Contains some basic plotting functionalities for plots that become repetitive, needs some TLC to become 
really useful.
- **utils:** Contains all functions that are useful for more than one application (e.g. filtering and interpolation).

The return of a function is a Pandas DataFrame in 9/10 cases. This means that you can also use all Pandas goodness. The 
exception to this rule is spatial data as that can be better represented in 3D arrays.

