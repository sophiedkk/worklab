# Worklab Documentation

[![image](https://zenodo.org/badge/DOI/10.5281/zenodo.3268671.svg)](https://doi.org/10.5281/zenodo.3268671) [![image](https://badge.fury.io/py/worklab.svg)](https://badge.fury.io/py/worklab) [![image](https://img.shields.io/badge/License-GPLv3-blue.svg)](https://www.gitlab.com/Rickdkk/worklab/tree/master/LICENCE)
```{eval-rst}
**Date**: |today|
```

The current page contains information on wheelchair propulsion research in the worklab and the correspondingly named 
Python package. As such, the page contains a general outline of wheelchair propulsion research, but also some practical 
examples on how to work with that data together with the API-reference of the package. The worklab package aims
to provide functions for the most common data (pre-)processing steps in wheelchair research (at our lab). As such, it is
mainly focussed on the hardware that is available in the worklab at the University Medical Center Groningen. It makes 
extensive use of pandas [dataframes](https://pandas.pydata.org/) as those are familiar to most researchers and uses a 
(mostly) functional approach.

::::{grid} 1 1 2 2
:class-container: text-center
:gutter: 3

:::{grid-item-card}
:link: chapters/setup.html
:class-header: bg-light

🚀 Quick Start Guide
^^^
Jump straight to the documentation on how to install the worklab package on your machine. This section also covers the
fundamentals of wheelchair research and data analysis.
:::

:::{grid-item-card}
:link: chapters/examples.html
:class-header: bg-light

👍 Minimal examples
^^^
This part provides you with a set of small examples that you can use to kickstart your analysis. The 
examples are used to showcase the functionality of the package while also showing you some good-practices.
:::

:::{grid-item-card}
:link: chapters/package.html
:class-header: bg-light

🙌 Package documentation
^^^
The API documentation provides a nicely rendered version of all the docstrings in the worklab package. This allows you
to search for functions in the documentation and read them in a way that is easy on the eyes.
:::

:::{grid-item-card}
:link: chapters/developer.html
:class-header: bg-light

🌍 Developer information
^^^
If you wish to contribute to the worklab package this is the place to be. Have a look at these pages to get your 
development environment up and running and your first pull request going.
:::
::::

## Authors

-  **Rick de Klerk** - *Initial work* - [ORCID](https://orcid.org/0000-0003-2745-1963)
-  **Thomas Rietveld** - [GitLab](https://gitlab.com/Thomas2016) -
    [UMCG](https://www.rug.nl/staff/t.rietveld)
-  **Rowie Janssen** - [UMCG](http://www.wheelpower.online) 

## Citing

If you want to refer to this package please use this DOI:
10.5281/zenodo.3268671, or cite:

R. de Klerk. (2019, July 4). Worklab: a wheelchair biomechanics mini-package (Version 1.0.0). Zenodo.
<http://doi.org/10.5281/zenodo.3268671>

## Acknowledgments

-   Thanks to [R.J.K. Vegter](https://www.rug.nl/staff/r.j.k.vegter/) for providing information on the Optipush and 
SMARTwheel systems.
-   Thanks to R.M.A. van der Slikke for providing information on skid correction.
-   Thanks to the people at Umaco for answering my questions however dumb they may be.

## References

Functions that use a specific algorithm or base have their references in the docstring. More generally, the analyses in
this package are taken from:

-   Vegter, R. J., Lamoth, C. J., De Groot, S., Veeger, D. H., & Van der, Woude, L. H. (2013). Variability in bimanual 
wheelchair propulsion: consistency of two instrumented wheels during handrim wheelchair propulsion on a motor driven 
treadmill. Journal of neuroengineering and rehabilitation, 10(1), 9.
-   Van der Slikke, R. M. A., Berger, M. A. M., Bregman, D. J. J., & Veeger, H. E. J. (2015). Wheel skid correction is 
a prerequisite to reliably measure wheelchair sports kinematics based on inertial sensors. Procedia Engineering, 112, 
207-212.
-   van der Slikke, R., Berger, M., Bregman, D., & Veeger, D. (2016). Push characteristics in wheelchair court sport 
sprinting. Procedia engineering, 147, 730-734.
