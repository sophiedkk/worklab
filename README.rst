Worklab: a wheelchair biomechanics mini-package
===============================================

.. image:: https://zenodo.org/badge/DOI/10.5281/zenodo.3268671.svg
   :target: https://doi.org/10.5281/zenodo.3268671

.. image:: https://badge.fury.io/py/worklab.svg
    :target: https://badge.fury.io/py/worklab

.. image:: https://img.shields.io/badge/License-GPLv3-blue.svg
    :target: https://www.gitlab.com/Rickdkk/worklab/tree/master/LICENCE

Essential data analysis and (pre-)processing scripts used in my project researching the Lode `Esseda`_
wheelchair ergometer in the worklab at the University Medical Centre Groningen. Includes all basic io and calculations for the equipment in the worklab, which means:

.. _Esseda: https://www.lode.nl/en/product/esseda-wheelchair-ergometer/637

* Measurement wheel (Optipush and SMARTwheel) and ergometer (Esseda) data processing
* Push-by-push analysis
* Spirometer (COSMED) data processing
* IMU (NGIMU) data processing
* Kinematics (Optotrak/OptiTrack) data processing
* more(?)

Documentation
-------------
For more detailed documentation you can look `here <https://worklab.readthedocs.io/en/latest>`_.

Prerequisites
-------------
You need a valid version of Python 3.6+ (because f-strings). This project has a bunch of dependencies for *reasons* so you will also need the scipy ecosystem
, which you probably already have.

Installing
----------
You can install this package with pip::

    pip install worklab

Examples
--------
You can find some Jupyter Notebook examples `here`_.

.. _here: https://gitlab.com/Rickdkk/worklab/tree/master/examples

Projects using same code
------------------------
* Viewer (built with PyQt) - source unfortunately was lost when my laptop was stolen	
* `Coast-down`_ analyzer (built with PySide 2)

.. _Coast-down: https://gitlab.com/Rickdkk/coast_down_test

Reporting errors
----------------
If you find an error or mistake, which entirely possible, please contact me or submit an issue through this page.

Authors
-------
* **Rick de Klerk** - *Initial work* - `gitlab`_ - `UMCG`_

.. _gitlab: https://gitlab.com/rickdkk
.. _UMCG: https://www.rug.nl/staff/r.de.klerk/

Citing
------
If you want to refer to this package please use this DOI: 10.5281/zenodo.3268671, or cite: R.de Klerk. (2019, July 4). Worklab: a wheelchair biomechanics mini-package (Version 1.0.0). Zenodo. http://doi.org/10.5281/zenodo.3268671

Acknowledgments
---------------
* Thanks to R.J.K. `Vegter`_ for providing information on the Optipush and SMARTwheel systems.
* Thanks to R.M.A. van der Slikke for providing information on skid correction.
* Thanks to the people at Umaco for answering my questions however dumb they may be.

.. _Vegter: https://www.rug.nl/staff/r.j.k.vegter/

References
----------
* Vegter, R. J., Lamoth, C. J., De Groot, S., Veeger, D. H., & Van der Woude, L. H. (2013). Variability in bimanual wheelchair propulsion: consistency of two instrumented wheels during handrim wheelchair propulsion on a motor driven treadmill. Journal of neuroengineering and rehabilitation, 10(1), 9.
* Van der Slikke, R. M. A., Berger, M. A. M., Bregman, D. J. J., & Veeger, H. E. J. (2015). Wheel skid correction is a prerequisite to reliably measure wheelchair sports kinematics based on inertial sensors. Procedia Engineering, 112, 207-212.
* van der Slikke, R., Berger, M., Bregman, D., & Veeger, D. (2016). Push characteristics in wheelchair court sport sprinting. Procedia engineering, 147, 730-734.