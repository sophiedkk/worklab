import datetime
import re
import time
from collections import defaultdict
from tkinter import Tk
from tkinter.filedialog import askopenfilename, askopenfilenames, asksaveasfilename, askdirectory

import numpy as np
import pandas as pd
from scipy.interpolate import InterpolatedUnivariateSpline, interp1d
from scipy.optimize import curve_fit
from scipy.signal import butter, sosfiltfilt


def pick_file(initialdir=None):
    """
    Open a window to select a single file

    Parameters
    ----------
    initialdir : str
        directory to start from

    Returns
    -------
    filename : str
        full path to picked file

    """
    root = Tk()
    root.withdraw()  # no root window
    filename = askopenfilename(initialdir=initialdir, title="Open data file or files")  # return path to selected file
    root.destroy()
    print("\n" + "=" * 80)
    print("You selected: ", filename)
    print("=" * 80, "\n")
    return filename


def pick_files(initialdir=None):
    """
    Open a window to select multiple files

    Parameters
    ----------
    initialdir : str
        directory to start from

    Returns
    -------
    filename : list
        full path to picked file

    """
    root = Tk()
    root.withdraw()  # no root window
    filenames = askopenfilenames(initialdir=initialdir, title="Open data file or files")  # path to selected files
    root.destroy()
    print("\n" + "=" * 80)
    print("You selected: ", filenames)
    print("=" * 80, "\n")
    return filenames


def pick_directory(initialdir=None):
    """
    Open a window to select a directory

    Parameters
    ----------
    initialdir : str
        directory to start from

    Returns
    -------
    directory : str
        full path to selected directory

    """
    root = Tk()
    root.withdraw()  # no root window
    directory = askdirectory(initialdir=initialdir, title="Select data directory")  # return path to selected directory
    root.destroy()
    print("\n" + "=" * 80)
    print("You selected: ", directory)
    print("=" * 80, "\n")
    return directory


def pick_save_file(initialdir=None):
    """
    Open a window to select a savefile

    Parameters
    ----------
    initialdir : str
        directory to start from

    Returns
    -------
    directory : str
        full path to selected savefile

    """
    root = Tk()
    root.withdraw()  # no root window
    filename = asksaveasfilename(initialdir=initialdir, title="Save file or files")  # return path to selected file
    root.destroy()
    print("\n" + "=" * 80)
    print("You selected: ", filename)
    print("=" * 80, "\n")
    return filename


def make_calibration_spline(calibration_points):
    """
    Makes a pre-1.0.4 calibration spline for the Esseda wheelchair ergometer.

    Parameters
    ----------
    calibration_points : dict
        dict with left: np.array, right: np.array

    Returns
    -------
    spl_line : dict
        dict with left: np.array, right: np.array containing the interpolated splines

    """
    spl_line = {"left": [], "right": []}
    for side in spl_line:
        x = np.arange(0, 10)
        spl = InterpolatedUnivariateSpline(x, calibration_points[side], k=2)
        spl_line[side] = spl(np.arange(0, 9.01, 0.01))  # Spline with 0.01 increments
        spl_line[side] = np.append(spl_line[side], np.full(99, spl_line[side][-1]))
    return spl_line


def make_linear_calibration_spline(calibration_points):
    """
    Makes a post-1.0.4 calibration spline for the Esseda wheelchair ergometer.

    Parameters
    ----------
    calibration_points : dict
        dict with left: np.array, right: np.array

    Returns
    -------
    spl_line : dict
        dict with left: np.array, right: np.array containing the interpolated splines

    """
    spl_line = {"left": [], "right": []}
    for side in spl_line:
        x = np.arange(0, 10)
        spl_line[side] = np.interp(np.arange(0, 9.01, 0.01), x, calibration_points[side])
    return spl_line


def lowpass_butter(array, sfreq=100., cutoff=20., order=2):
    """
    Apply a simple zero-phase low-pass Butterworth filter on an array.

    Parameters
    ----------
    array : np.array
        input array to be filtered
    sfreq : float
        sample frequency of the signal, default is 100
    cutoff : float
        cutoff frequency for the filter, default is 20
    order : int
        order of the filter, default is 2

    Returns
    -------
    array : np.array
        filtered array

    """
    # noinspection PyTupleAssignmentBalance
    array = np.asarray(array)
    sos = butter(order, cutoff, fs=sfreq, btype='low', output='sos')
    return sosfiltfilt(sos, array)


def interpolate_array(x, y, kind="linear", fill_value="extrapolate", assume=True):
    """
    Simple function to interpolate an array with Scipy's interp1d. Also extrapolates NaNs.

    Parameters
    ----------
    x : np.array
        time array (without NaNs)
    y : np.array
        array with potential NaNs
    kind : str
        kind of filter, default is "linear"
    fill_value : str
        fill value, default is "extrapolate"
    assume : bool
        assume that the array is sorted (performance), default is True

    Returns
    -------
    y : np.array
        interpolated y-array

    """
    y_fun = interp1d(x[~np.isnan(y)], y[~np.isnan(y)], kind=kind, fill_value=fill_value, assume_sorted=assume)
    return y_fun(x)


def pd_dt_to_s(dt):
    """
    Calculates time in seconds from datetime or string.

    Parameters
    ----------
    dt : pd.Series
        datetime instance or a string with H:m:s data

    Returns
    -------
    time : pd.Series
        time in seconds

    """
    if isinstance(dt, datetime.time):
        time = (dt.hour * 60 + dt.minute) * 60 + dt.second
    else:
        h, m, s = dt.split(":")
        time = int(h) * 3600 + int(m) * 60 + int(s)
    return time


def pd_interp(df, interp_column, at):
    """
    Resamples (and extrapolates) DataFrame with Scipy's interp1d, this was more performant than the pandas one for some
    reason.

    Parameters
    ----------
    df : pd.DataFrame
        target DataFrame
    interp_column : str
        column to interpolate on, e.g. "time"
    at : np.array
        column to interpolate to

    Returns
    -------
    interp_df : pd.DataFrame
        interpolated DataFrame

    """
    interp_df = pd.DataFrame()
    for col in df:
        f = interp1d(df[interp_column], df[col], fill_value="extrapolate")
        interp_df[col] = f(at)
    interp_df[interp_column] = at
    return interp_df


def merge_chars(chars):
    """
    Merges list or tuple of binary characters to single string

    Parameters
    ----------
    chars : list, tuple
        list or tuple of binary characters

    Returns
    -------
    str
        concatenated characters

    """
    return ''.join([char.decode("utf-8") for char in chars])


def find_peaks(data, cutoff=1.0, minpeak=5.0, min_dist=5):
    """
    Finds positive peaks in signal and returns indices of start and stop.

    Parameters
    ----------
    data : pd.Series, np.array
        any signal that contains peaks above minpeak that dip below cutoff
    cutoff : float
        where the peak gets cut off at the bottom, basically a hysteresis band
    minpeak : float
        minimum peak height of wave
    min_dist : int
        minimum sample distance between peak candidates, can be used to speed up algorithm

    Returns
    -------
    peaks : dict
        dictionary with start, end, and peak **index** of each peak

    """
    peaks = defaultdict(list)
    tmp = {"start": None, "stop": None}

    data = np.asarray(data)  # coercing to an array if necessary
    data_slice = np.nonzero(data > minpeak)[0]  # indices of nonzero values
    data_slice = data_slice[np.diff(data_slice, append=10e100) > min_dist]  # remove duplicate samples from push
    for prom in np.nditer(data_slice):
        tmp["stop"] = next((idx for idx, value in enumerate(data[prom:]) if value < cutoff), None)
        tmp["start"] = next((idx for idx, value in enumerate(reversed(data[:prom])) if value < cutoff), None)
        if tmp["stop"] and tmp["start"]:  # did we find a start and stop?
            peaks["stop"].append(tmp["stop"] + prom - 1)
            peaks["start"].append(prom - tmp["start"])
    for key, value in peaks.items():
        peaks[key] = np.unique(value)  # remove possible duplicates
    peaks["peak"] = [np.argmax(data[start:stop + 1]) + start for start, stop in zip(peaks["start"], peaks["stop"])]
    return peaks


def coast_down_velocity(t, v0, c1, c2, m):
    """
    Solution for the non-linear differential equation M(dv/dt) + c1*v**2 + c2 = 0. Returns the instantaneous
    velocity decreasing with time (t) for the friction coefficients c1 and c2 for an object with a fixed mass (M)

    Parameters
    ----------
    t : np.array
    v0 : float
    c1 : float
    c2 : float
    m : float

    Returns
    -------
    np.array

    """
    return np.sqrt(c2 / c1) * np.tan(np.arctan(v0 * np.sqrt(c1 / c2)) - t * (np.sqrt(c1 * c2) / m))


def nonlinear_fit_coast_down(time, vel, total_weight):
    """
    Performs a nonlinear fit on coast-down data, returning c1 and c2.

    Parameters
    ----------
    time : np.array
    vel : np.array
    total_weight : float

    Returns
    -------
    tuple
        c1, c2

    """
    # try to determine c1 and c2 with curve_fit for non-linear approach
    initial_velocity = vel[0]
    vel_func = lambda t, c1, c2: coast_down_velocity(t, initial_velocity, c1, c2, total_weight)  # lock variables
    (coef1, coef2), pcov = curve_fit(vel_func, time, vel, bounds=(0, [1, 20]))  # 'trf' method but 'lm' would also work
    return coef1, coef2


def mask_from_iterable(array, floor_values, ceil_values):
    """
    Combines multiple masks from iterable into one mask (e.g. can be used to select multiple time slices).

    Parameters
    ----------
    array : np.array
        array to apply mask on
    floor_values : list
        minimum values in array
    ceil_values : list
        maximum values in array

    Returns
    -------
    mask : np.array

    """
    mask = np.full(array.shape, False)  # start off with all data deselected
    for ceil, floor in zip(ceil_values, floor_values):
        mask = mask | ((array > floor) & (array < ceil))
    return mask


def calc_inertia(weight=0.8, radius=0.295, length=0.675, period=1.0):
    """
    Calculate the inertia of an object based on the trifilar pendulum equation.

    Parameters
    ----------
    weight : float
        total mass of the object, default is 0.8
    radius : float
        radius of the object, default is 0.295
    length : float
        length of the trifilar pendulum
    period : float
        observed oscillation period

    Returns
    -------
    inertia : float
        inertia [kgm2]

    """
    return (weight * 9.81 * radius ** 2 * period ** 2) / (4 * np.pi ** 2 * length)


def zerocross1d(x, y, indices=False):
    """
    Find the zero crossing points in 1d data.

    Find the zero crossing events in a discrete data set. Linear interpolation is used to determine the actual
    locations of the zero crossing between two data points showing a change in sign. Data point which are zero
    are counted in as zero crossings if a sign change occurs across them. Note that the first and last data point will
    not be considered whether or not they are zero.

    Parameters
    ----------
    x : np.array, pd.Series
        time/sample variable
    y : np.array, pd.Series
        y variable
    indices : bool
        return indices or not, default is False

    Returns
    -------
    np.array
        position in time and optionally the index of the sample before the zero-crossing

    """
    x = np.asarray(x)
    y = np.asarray(y)
    # Indices of points *before* zero-crossing
    indi = np.where(y[1:] * y[0:-1] < 0.0)[0]

    # Find the zero crossing by linear interpolation
    dx = x[indi + 1] - x[indi]
    dy = y[indi + 1] - y[indi]
    zc = -y[indi] * (dx / dy) + x[indi]

    # What about the points, which are actually zero
    zi = np.where(y == 0.0)[0]
    # Do nothing about the first and last point should they be zero
    zi = zi[np.where((zi > 0) & (zi < x.size - 1))]
    # Select those point, where zero is crossed (sign change across the point)
    zi = zi[np.where(y[zi - 1] * y[zi + 1] < 0.0)]

    # Concatenate indices
    zeroc_indices = np.concatenate((indi, zi))
    # Concatenate zc and locations corresponding to zi
    zeroc_xvalues = np.concatenate((zc, x[zi]))

    # Sort by x-value
    sind = np.argsort(zeroc_xvalues)
    zeroc_xvalues, zeroc_indices = zeroc_xvalues[sind], zeroc_indices[sind]

    return (zeroc_xvalues, zeroc_indices) if indices else zeroc_xvalues


def camel_to_snake(name: str):
    """
    Turns CamelCased text into snake_cased text.

    Parameters
    ----------
    name : str
        StringToConvert

    Returns
    -------
    str
        converted_string
    """

    s1 = re.sub('(.)([A-Z][a-z]+)', r'\1_\2', name)
    return re.sub('([a-z0-9])([A-Z])', r'\1_\2', s1).lower()


def find_nearest(array, value, index=False):
    """
    Find the nearest value in an array or the index thereof.

    Parameters
    ----------
    array : np.array
        array which has to be searched
    value : float
        value that you are looking for
    index : bool
        whether or not you want the index

    Returns
    -------
    np.array
        value or index of nearest value

    """
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return idx if index else array[idx]


def split_dataframe(df, inds):
    """
    Split a dataframe on a list of indices. For example a dataframe that contains multiple sessions of wheelchair
    ergometer data.

    Parameters
    ----------
    df : pd.DataFrame
        target dataframe
    inds : list
        list of indices where the dataframe should be split

    Returns
    -------
    list
        list of dataframes

    """
    df.reset_index(drop=True, inplace=True)
    inds = [0] + list(inds) + [len(df)]  # add first and last index for slicing
    return [df.iloc[start:stop, :].copy().reset_index(drop=True) for start, stop in zip(inds[0::], inds[1::])]


def binned_stats(array, bins=10, pad=True, func=np.mean, nan_func=np.nanmean):
    """
    Apply a compatible Numpy function to every bins samples (e.g. mean or std).

    Parameters
    ----------
    array : np.array
        array which has to be searched
    bins : int
        number of samples to be averaged
    pad : bool
        whether or not to pad the array with NaNs if needed
    func
        function that is used when no padding is applied
    nan_func
        function that is used when padding is applied

    Returns
    -------
    means: np.array
        array with the mean for every bins samples.

    """
    array = np.array(array, dtype=float)  # make sure we have an array
    if pad:
        array = np.pad(array, (0, bins - array.size % bins), mode='constant', constant_values=np.NaN)
        means = nan_func(array.reshape(-1, bins), axis=1)
    else:
        means = func(array[:(len(array) // bins) * bins].reshape(-1, bins), axis=1)
    return means


class TimerError(Exception):
    """A custom exception used to report errors in use of Timer class"""


class Timer:
    """
    Simple timer for timing code(blocks).

    Parameters
    ----------
    name : str
        name of timer, gets saved in Timer.timers optional
    text : str
        custom text, optional
    start : bool
        automatically start the timer when it's initialized, default is True

    Methods
    -------
    start
        start the timer
    stop
        stop the timer, prints and returns the time
    lap
        print the time between this lap and the previous one

    """
    timers = dict()

    def __init__(self, name="", text="Elapsed time: {:0.4f} seconds", start=True):
        self._start_time = None
        self._lap_time = 0.
        self.name = name
        self.text = text

        if name:
            self.timers.setdefault(name, 0)  # Add new named timers to dictionary of timers
        if start:
            self.start()

    def start(self):
        """Start a new timer"""
        if self._start_time is not None:
            raise TimerError(f"Timer is already running. Use .stop() to stop it")
        self._start_time = time.perf_counter()

    def lap(self, lap_name=""):
        """Report the elapsed time"""
        if self._start_time is None:
            raise TimerError(f"Timer is not running. Use .start() to start it")

        if self._lap_time:
            current_lap = time.perf_counter() - self._lap_time - self._start_time
            self._lap_time += current_lap
        else:
            self._lap_time = time.perf_counter() - self._start_time
            current_lap = self._lap_time

        if lap_name:
            print(lap_name)
        print(self.text.format(current_lap))

    def stop(self):
        """Stop the timer, and report the elapsed time"""
        if self._start_time is None:
            raise TimerError(f"Timer is not running. Use .start() to start it")

        elapsed_time = time.perf_counter() - self._start_time
        self._start_time = None

        print(self.text.format(elapsed_time))
        if self.name:
            self.timers[self.name] += elapsed_time
        return elapsed_time
