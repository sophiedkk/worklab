Python modules
==============


Rationale
---------
This is an attempt to make analysis of wheelchair biomechanics data more accessible and transparent. Previously all
analyses were performed with commercial software that is not available to everyone, especially to people not associated
with a university. Having the analysis in Python makes it accessible and more readable (hopefully) for everyone.
By sharing the code I hope to be transparent and to reduce the amount of times this code has to be written by other people.

Examples & audience
-------------------
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

* com: 		    Provides functions for reading and writing data, use ``load`` to infer filetype and automatically read it. If you use a different naming scheme you can always call the specific load functions.
* kinetics: 	Contains all essentials for measurement wheel and ergometer data. You only need the top-level function ``auto_process`` for most use-cases.
* move: 	    Contains kinematics and movement related functions for NGIMU and some functions for 3D kinematics.
* physio: 	    Contains physiological calculations, which for now is basically nothing as the spirometer does everything for you. Might include EMG and the likes later though.
* plots:        Contains some basic plotting functionalities for plots that become repetitive, needs some TLC to become really useful.
* utils:        Contains all functions that are useful for more than one application (e.g. filtering and interpolation).

The return of a function is a Pandas DataFrame in 9/10 cases. This means that you can also use all Pandas goodness.

Communication (.com)
---------------------------
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

load_wheelchair
^^^^^^^^^^^^^^^
.. autofunction:: worklab.com.load_wheelchair

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


Kinetics (.kin)
----------------------
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

Kinematics (.move)
-------------------------
Basic functions for movement related data such as from IMUs or optical tracking systems. IMU functions are
specifically made for the NGIMUs we use in the worklab.

resample_imu
^^^^^^^^^^^^
.. autofunction:: worklab.move.resample_imu

calc_wheelspeed
^^^^^^^^^^^^^^^
.. autofunction:: worklab.move.calc_wheelspeed

change_imu_orientation
^^^^^^^^^^^^^^^^^^^^^^
.. autofunction:: worklab.move.change_imu_orientation

push_detection
^^^^^^^^^^^^^^
.. autofunction:: worklab.move.push_detection

get_perp_vector
^^^^^^^^^^^^^^^
.. autofunction:: worklab.move.get_perp_vector

get_rotation_matrix
^^^^^^^^^^^^^^^^^^^
.. autofunction:: worklab.move.get_rotation_matrix

normalize
^^^^^^^^^
.. autofunction:: worklab.move.normalize

calc_marker_angles
^^^^^^^^^^^^^^^^^^
.. autofunction:: worklab.move.calc_marker_angles

Physiology module (.physio)
----------------------------------
Basics for working with physiological data. We only have a spirometer in the lab at the moment and this
involves very little processing. Might expand with EMG related function at some point in the future.

get_spirometer_units
^^^^^^^^^^^^^^^^^^^^
.. autofunction:: worklab.physio.get_spirometer_units

Plotting (.plots)
------------------------
Most variables can easily be plotted with matplotlib or pandas as most data in this package is contained
in dataframes. Some plotting is tedious however and these are functions for those plots.

plot_pushes
^^^^^^^^^^^
.. autofunction:: worklab.plots.plot_pushes

plot_pushes_ergo
^^^^^^^^^^^^^^^^
.. autofunction:: worklab.plots.plot_pushes_ergo

bland_altman_plot
^^^^^^^^^^^^^^^^^
.. autofunction:: worklab.plots.bland_altman_plot

Utilities (.utils)
-------------------------
This module contains utility functions used by all modules or functions that have multiple applications such as
filtering, finding zero-crossings, finding the nearest value in a signal.

pick_file
^^^^^^^^^
.. autofunction:: worklab.utils.pick_file

pick_files
^^^^^^^^^^
.. autofunction:: worklab.utils.pick_files

pick_directory
^^^^^^^^^^^^^^
.. autofunction:: worklab.utils.pick_directory

pick_save_file
^^^^^^^^^^^^^^
.. autofunction:: worklab.utils.pick_save_file

calc_weighted_average
^^^^^^^^^^^^^^^^^^^^^
.. autofunction:: worklab.utils.calc_weighted_average

make_calibration_spline
^^^^^^^^^^^^^^^^^^^^^^^
.. autofunction:: worklab.utils.make_calibration_spline

make_linear_calibration_spline
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. autofunction:: worklab.utils.make_linear_calibration_spline

pd_dt_to_s
^^^^^^^^^^
.. autofunction:: worklab.utils.pd_dt_to_s

lowpass_butter
^^^^^^^^^^^^^^
.. autofunction:: worklab.utils.lowpass_butter

interpolate_array
^^^^^^^^^^^^^^^^^
.. autofunction:: worklab.utils.interpolate_array

pd_interp
^^^^^^^^^
.. autofunction:: worklab.utils.pd_interp

merge_chars
^^^^^^^^^^^
.. autofunction:: worklab.utils.merge_chars

find_peaks
^^^^^^^^^^
.. autofunction:: worklab.utils.find_peaks

coast_down_velocity
^^^^^^^^^^^^^^^^^^^
.. autofunction:: worklab.utils.coast_down_velocity

nonlinear_fit_coast_down
^^^^^^^^^^^^^^^^^^^^^^^^
.. autofunction:: worklab.utils.nonlinear_fit_coast_down

mask_from_iterable
^^^^^^^^^^^^^^^^^^
.. autofunction:: worklab.utils.mask_from_iterable

calc_inertia
^^^^^^^^^^^^
.. autofunction:: worklab.utils.calc_inertia

zerocross1d
^^^^^^^^^^^
.. autofunction:: worklab.utils.zerocross1d

camel_to_snake
^^^^^^^^^^^^^^
.. autofunction:: worklab.utils.camel_to_snake

find_nearest
^^^^^^^^^^^^
.. autofunction:: worklab.utils.find_nearest

Timer
^^^^^
.. autoclass:: worklab.utils.Timer
