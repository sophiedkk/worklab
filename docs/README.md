# Worklab Documentation

[![image](https://zenodo.org/badge/DOI/10.5281/zenodo.8362963.svg)](https://doi.org/10.5281/zenodo.8362963) [![image](https://badge.fury.io/py/worklab.svg)](https://badge.fury.io/py/worklab) [![image](https://img.shields.io/badge/License-GPLv3-blue.svg)](https://www.gitlab.com/sophiedkk/worklab/tree/master/LICENCE)
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

-  **Sophie de Klerk** - *Initial work* - [ORCID](https://orcid.org/0000-0003-2745-1963)
-  **Thomas Rietveld** - [GitLab](https://gitlab.com/Thomas2016) -
    [ORCID](https://orcid.org/0000-0002-7753-9958)
-  **Rowie Janssen** - [ORCID](https://orcid.org/0000-0001-7480-3779) 
-  **Jelmer Braaksma** - [ORCID](https://orcid.org/0000-0002-9103-3590)

## Citing

If you want to refer to this package please use this DOI: 10.5281/zenodo.8362962, or cite:

Sophie de Klerk, Thomas Rietveld, Rowie Janssen, & Jelmer Braaksma. (2023). Worklab: a wheelchair biomechanics mini-package. 
Zenodo. https://doi.org/10.5281/zenodo.8362963

## Acknowledgments

-   Thanks to [R.J.K. Vegter](https://www.rug.nl/staff/r.j.k.vegter/) for providing information on the Optipush and 
SMARTwheel systems.
-   Thanks to R.M.A. van der Slikke for providing information on skid correction.
-   Thanks to the people at Umaco for answering my questions however dumb they may be.

## References

Functions that use a specific algorithm or base have their references in the docstring. More generally, the analyses in
this package are taken from:

-   Janssen RJF, Vegter RJK, Houdijk H, Van der Woude LHV and De Groot S (2022). Evaluation of a standardized test protocol
to measure wheelchair-specific anaerobic and aerobic exercise capacity in healthy novices on an instrumented roller
ergometer. PLOS ONE 17(9): e0274255. https://doi.org/10.1371/journal.pone.0274255
- De Klerk R, Vegter RJK, Veeger HEJ and Van der Woude LHV (2020). Technical Note: A Novel Servo-Driven
Dual-Roller Handrim Wheelchair Ergometer. IEEE Transactions on Neural Systems and Rehabilitation Engineering,
28(4), 953-960. https://doi.org/10.1109/TNSRE.2020.2965281
- Rietveld T, Vegter RJK, van der Slikke RMA, Hoekstra AE, van der Woude LHV and de Groot S (2019). Wheelchair mobility
performance of elite wheelchair tennis players during four field tests: Inter-trial reliability and construct validity.
PLoS One 14(6): e0217514. https://doi.org/10.1371/journal.pone.0217514
- de Klerk R, Vegter RJK, Leving MT, de Groot S, Veeger HEJ, van der Woude LHV (2020). Determining and Controlling External
Power Output During Regular Handrim Wheelchair Propulsion. J Vis Exp. 156. https://doi.org/10.3791/60492
-   Vegter RJK, Lamoth CJ, De Groot S, Veeger HEJ and Van der Woude LHV (2013). Variability in bimanual 
wheelchair propulsion: consistency of two instrumented wheels during handrim wheelchair propulsion on a motor driven 
treadmill. Journal of neuroengineering and rehabilitation, 10(1), 9. https://doi.org/10.1186/1743-0003-10-9
-   Van der Slikke RMA, Berger MAM, Bregman DJJ, & Veeger HEJ (2015). Wheel skid correction is 
a prerequisite to reliably measure wheelchair sports kinematics based on inertial sensors. Procedia Engineering, 112, 
207-212. https://doi.org/10.1016/j.proeng.2015.07.201
-   Van der Slikke RMA, Berger MAM, Bregman DJJ and Veeger HEJ (2016). Push characteristics in wheelchair court sport 
sprinting. Procedia engineering, 147, 730-734. https://doi.org/10.1016/j.proeng.2016.06.265
