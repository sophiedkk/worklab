# Worklab: a wheelchair biomechanics mini-package

[![image](https://zenodo.org/badge/DOI/10.5281/zenodo.8362962.svg)](https://doi.org/10.5281/zenodo.8362962)
[![image](https://badge.fury.io/py/worklab.svg)](https://badge.fury.io/py/worklab) [![image](https://img.shields.io/badge/License-GPLv3-blue.svg)](https://github.com/sophiedkk/worklab/blob/main/LICENSE)

Essential data analysis and (pre-)processing scripts used in projects
researching the Lode
[Esseda](https://www.lode.nl/en/product/esseda-wheelchair-ergometer/637)
wheelchair ergometer in the worklab at the University Medical Centre
Groningen. Includes all basic io and calculations for the equipment in
the worklab, which means:

-   Measurement wheel (Optipush and SMARTwheel) data processing
-   Ergometer (Esseda) data processing
-   Push-by-push analysis
-   Spirometer (COSMED) data processing
-   IMU (NGIMU, XIMU3 and MoveSense) data processing
-   Kinematics (Optotrak/OptiTrack) data processing
-   More(?)

# Documentation

For more detailed documentation you can look at the
[docs](https://sophiedkk.github.io/worklab/).

# Prerequisites

You need a valid version of Python 3.6+ (because f-strings). This
project has a bunch of dependencies for *reasons*, so you will also need
the scipy ecosystem , which you probably already have.

# Installing

You can install this package with pip:

    pip install worklab

# Examples

You can find some Jupyter Notebook examples
[here](https://sophiedkk.github.io/worklab/chapters/examples.html).

# Reporting errors

If you find an error or mistake, which entirely possible, please contact
me or submit an issue through this page.

# Citing

If you want to refer to this package please use this DOI:
10.5281/zenodo.8362962, or cite:

Sophie de Klerk, Thomas Rietveld, Rowie Janssen, & Jelmer Braaksma. (2023). Worklab: a wheelchair biomechanics mini-package. Zenodo. https://doi.org/10.5281/zenodo.8362963
