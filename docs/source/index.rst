Worklab documentation
=====================
.. image:: https://zenodo.org/badge/DOI/10.5281/zenodo.3268671.svg
   :target: https://doi.org/10.5281/zenodo.3268671

.. image:: https://badge.fury.io/py/worklab.svg
    :target: https://badge.fury.io/py/worklab

.. image:: https://img.shields.io/badge/License-GPLv3-blue.svg
    :target: https://www.gitlab.com/Rickdkk/worklab/tree/master/LICENCE

.. image:: https://readthedocs.org/projects/worklab/badge/?version=latest
    :target: https://worklab.readthedocs.io/en/latest/?badge=latest
    :alt: Documentation Status

Introduction
------------
The current page contains information on wheelchair propulsion research in the worklab and the correspondingly named Python package.
As such, the page contains a general outline of wheelchair propulsion research, but also some practical examples on how to work with
that data together with the API-reference of the worklab package. The worklab package aims to provide functions for the most common
data (pre-)processing steps in wheelchair research (at our lab). It makes extensive use of pandas dataframes as those are familiar to
most researchers.

Contents
--------

.. toctree::
    :maxdepth: 1

    theory
    overground
    treadmill
    ergometer
    general
    examples
    API

Source
------
The source is on `this GitLab page <https://gitlab.com/Rickdkk/worklab>`_.

About
-----
Authors
^^^^^^^
* **Rick de Klerk** - *Initial work* - `GitLab <https://gitlab.com/rickdkk>`_ - `UMCG <https://www.rug.nl/staff/r.de.klerk/>`_
* **Thomas Rietveld** - `GitLab <https://gitlab.com/Thomas2016>`_ - `UMCG <https://www.rug.nl/staff/t.rietveld>`_

Citing
^^^^^^
If you want to refer to this package please use this DOI: 10.5281/zenodo.3268671, or cite:

R.de Klerk. (2019, July 4). Worklab: a wheelchair biomechanics mini-package (Version 1.0.0). Zenodo. http://doi.org/10.5281/zenodo.3268671

Acknowledgments
^^^^^^^^^^^^^^^
* Thanks to `R.J.K. Vegter <https://www.rug.nl/staff/r.j.k.vegter/>`_ for providing information on the Optipush and SMARTwheel systems.
* Thanks to R.M.A. van der Slikke for providing information on skid correction.
* Thanks to the people at Umaco for answering my questions however dumb they may be.

References
^^^^^^^^^^
Specifically for the Python module:

* Vegter, R. J., Lamoth, C. J., De Groot, S., Veeger, D. H., & Van der Woude, L. H. (2013). Variability in bimanual wheelchair propulsion: consistency of two instrumented wheels during handrim wheelchair propulsion on a motor driven treadmill. Journal of neuroengineering and rehabilitation, 10(1), 9.
* Van der Slikke, R. M. A., Berger, M. A. M., Bregman, D. J. J., & Veeger, H. E. J. (2015). Wheel skid correction is a prerequisite to reliably measure wheelchair sports kinematics based on inertial sensors. Procedia Engineering, 112, 207-212.
* van der Slikke, R., Berger, M., Bregman, D., & Veeger, D. (2016). Push characteristics in wheelchair court sport sprinting. Procedia engineering, 147, 730-734.

Disclaimer
^^^^^^^^^^
THE GUIDE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED,
INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A
PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER
IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN
CONNECTION WITH THE GUIDE OR THE USE OR OTHER DEALINGS IN THE GUIDE.