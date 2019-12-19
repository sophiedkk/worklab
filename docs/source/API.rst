Python modules
==============


Rationale
---------
This is an attempt to make analysis of wheelchair biomechanics data more accessible and transparent. Previously all
analyses were performed with commercial software that is not available to everyone, especially to people not associated
with a university. Having the analysis in Python makes it accessible and more readable (hopefully) for everyone.
By sharing the code I hope to be transparent and to reduce the amount of times this code has to be written by other people.

Target audience
---------------
People working in our lab that want to work with data from any of our instruments. It can, of course, also be used by other
people, provided that you have similar equipment. Most of the time you will only need one or two functions which you can
just take from the source code or you can just install the package as it has very little overhead anyways and only uses
packages that you probably already have installed. Also have a look at the
`examples <https://gitlab.com/Rickdkk/worklab/tree/master/examples>`_.

Installation
------------
Option 1: the package is now on pip::

    pip install worklab

Option 2: download the package from this page, and run::

    python setup.py install

Option 3: don't install it and just include the scripts in your working directory (why though?).

To verify if everything works simply try to import worklab::

    python
    import worklab as wl

That's it.

Breakdown
---------
Core files:

* com: 		    Provides functions for reading and writing data, use ``load`` to infer filetype and automatically read it. If you use a different naming scheme you can always call the specific load functions.
* kinetics: 	Contains all essentials for measurement wheel and ergometer data. You only need the top-level function ``auto_process`` for most use-cases.
* move: 	    Contains kinematics and movement related functions for NGIMU and some functions for 3D kinematics.
* physio: 	    Contains physiological calculations, which for now is basically nothing as the spirometer does everything for you. Might include EMG and the likes later though.
* plots:        Contains some basic plotting functionalities for plots that become repetitive, needs some TLC to become really useful.
* utils:        Contains all functions that are useful for more than one application (e.g. filtering and interpolation).

The return of a function is a Pandas DataFrame in 9/10 cases. This means that you can also use all Pandas goodness.

Communication module
--------------------
Contains functions for reading data from any worklab device. If you abide by regular naming conventions
you will only need the load function which will infer the correct function for you. You can also use device-specific
load functions if needed.

load
^^^^
.. autofunction:: worklab.com.load

load_bike
^^^^^^^^^
.. autofunction:: worklab.com.load_bike

load_esseda
^^^^^^^^^^^
.. autofunction:: worklab.com.load_esseda

load_hsb
^^^^^^^^
.. autofunction:: worklab.com.load_hsb

load_n3d
^^^^^^^^
.. autofunction:: worklab.com.load_n3d

load_opti
^^^^^^^^^
.. autofunction:: worklab.com.load_opti

load_optitrack
^^^^^^^^^^^^^^
.. autofunction:: worklab.com.load_optitrack

load_session
^^^^^^^^^^^^
.. autofunction:: worklab.com.load_session

load_spiro
^^^^^^^^^^
.. autofunction:: worklab.com.load_spiro

load_spline
^^^^^^^^^^^
.. autofunction:: worklab.com.load_spline

load_sw
^^^^^^^
.. autofunction:: worklab.com.load_sw


Kinetics module
---------------
Contains functions for working with measurement wheel (Optipush and SMARTwheel) and ergometer (Esseda) data
You will usually only need the top-level function ``auto_process``.

auto_process
^^^^^^^^^^^^
.. autofunction:: worklab.kin.auto_process

filter_mw
^^^^^^^^^
.. autofunction:: worklab.kin.filter_mw

filter_ergo
^^^^^^^^^^^
.. autofunction:: worklab.kin.filter_ergo

process_mw
^^^^^^^^^^
.. autofunction:: worklab.kin.process_mw

process_ergo
^^^^^^^^^^^^
.. autofunction:: worklab.kin.process_ergo

push_by_push_mw
^^^^^^^^^^^^^^^
.. autofunction:: worklab.kin.push_by_push_mw

push_by_push_ergo
^^^^^^^^^^^^^^^^^
.. autofunction:: worklab.kin.push_by_push_ergo

Kinematics module
-----------------
.. automodule:: worklab.move
    :members:

Physiology module
-----------------
.. automodule:: worklab.physio
    :members:

Plotting module
---------------
.. automodule:: worklab.plots
    :members:

Utilities module
----------------
.. automodule:: worklab.utils
    :members:
