"""
-Helper and utility functions-
Description: Most variables can easily be plotted with matplotlib or pandas as most data in this package is contained
in dataframes. Some plotting is tedious however these are functions for those plots.
Author:     Rick de Klerk
Contact:    r.de.klerk@umcg.nl
Company:    University Medical Center Groningen
License:    GNU GPLv3.0
Date:       27/06/2019
"""
import datetime
import re
from collections import defaultdict
from tkinter import Tk
from tkinter.filedialog import askopenfilename, askopenfilenames, asksaveasfilename, askdirectory

import numpy as np
import pandas as pd
from scipy.interpolate import InterpolatedUnivariateSpline, interp1d
from scipy.optimize import curve_fit
from scipy.signal import butter, filtfilt


def pick_file(initialdir: str = None) -> object:
    """Open a window to select a single file

    :param initialdir: directory to start from
    :return: full path to picked file
    """
    root = Tk()
    root.withdraw()  # no root window
    filename = askopenfilename(initialdir=initialdir, title="Open data file or files")  # return path to selected file
    root.destroy()
    print("\n" + "=" * 80)
    print("You selected: ", filename)
    print("=" * 80, "\n")
    return filename


def pick_files(initialdir: str = None) -> list:
    """Open a window to select multiple files

    :param initialdir: directory to start from
    :return: list of paths to selected files
    """
    root = Tk()
    root.withdraw()  # no root window
    filenames = askopenfilenames(initialdir=initialdir, title="Open data file or files")  # path to selected files
    root.destroy()
    print("\n" + "=" * 80)
    print("You selected: ", filenames)
    print("=" * 80, "\n")
    return filenames


def pick_directory(initialdir: str = None) -> str:
    """Open a window to select a directory

    :param initialdir: directory to start from
    :return: path to selected directory
    """
    root = Tk()
    root.withdraw()  # no root window
    directory = askdirectory(initialdir=initialdir, title="Select data directory")  # return path to selected directory
    root.destroy()
    print("\n" + "=" * 80)
    print("You selected: ", directory)
    print("=" * 80, "\n")
    return directory


def pick_save_file(initialdir: str = None) -> str:
    """Open a window to select a savefile

    :param initialdir: directory to start from
    :return: path to selected savefile
    """
    root = Tk()
    root.withdraw()  # no root window
    filename = asksaveasfilename(initialdir=initialdir, title="Save file or files")  # return path to selected file
    root.destroy()
    print("\n" + "=" * 80)
    print("You selected: ", filename)
    print("=" * 80, "\n")
    return filename


def calc_weighted_average(dataframe: pd.DataFrame, weights: pd.Series) -> pd.DataFrame:
    """Calculate the weighted average of all columns in a dataframe

    :param dataframe: input dataframe
    :param weights: weights, can be any iterable of equal length
    :return: the weighted averages of each column
    """
    return dataframe.apply(lambda col: np.average(col, weights=weights), axis=0)


def make_calibration_spline(calibration_points: dict) -> dict:
    """Makes a pre-1.0.4 calibration spline for the Esseda wheelchair ergometer

    :param calibration_points: dict with left: np.array, right: np.array
    :return: dict with left: np.array, right: np.array containing the interpolated splines
    """
    spl_line = {"left": [], "right": []}
    for side in spl_line:
        x = np.arange(0, 10)
        spl = InterpolatedUnivariateSpline(x, calibration_points[side], k=2)
        spl_line[side] = spl(np.arange(0, 9.01, 0.01))  # Spline with 0.01 increments
        spl_line[side] = np.append(spl_line[side], np.full(99, spl_line[side][-1]))
    return spl_line


def make_linear_calibration_spline(calibration_points: dict) -> dict:
    """Makes a post-1.0.4 calibration spline for the Esseda wheelchair ergometer

    :param calibration_points: dict with left: np.array, right: np.array
    :return: dict with left: np.array, right: np.array containing the interpolated splines
    """
    spl_line = {"left": [], "right": []}
    for side in spl_line:
        x = np.arange(0, 10)
        spl_line[side] = np.interp(np.arange(0, 9.01, 0.01), x, calibration_points[side])
    return spl_line


def pd_dt_to_s(dt):
    """Calculates time in seconds from datetime or string

    :param dt: datetime instance or a string with H:m:s data
    :return: time in seconds
    """
    if isinstance(dt, datetime.time):
        time = (dt.hour * 60 + dt.minute) * 60 + dt.second
    else:
        h, m, s = dt.split(":")
        time = int(h) * 3600 + int(m) * 60 + int(s)
    return time


def lowpass_butter(array: np.array, sfreq: int, cutoff: int = 20, order: int = 2) -> np.array:
    """A simple low-pass Butterworth filter on an array
    :param array: input array
    :param sfreq: sample frequency of the signal
    :param cutoff: cutoff frequency for the filter
    :param order: filter order
    :return: filtered array
    """
    # noinspection PyTupleAssignmentBalance
    b, a = butter(order, cutoff / (0.5 * sfreq), 'low')
    return filtfilt(b, a, array)


def interpolate_array(x: np.array, y: np.array, kind: str = "linear", assume: bool = True) -> np.array:
    """Simple function to interpolate an array with Scipy's interp1d. Also extrapolates NaNs.

    :param x: time array (without NaNs)
    :param y: array with potential NaNs
    :param kind: look at scipy.interpolate.interp1d for options
    :param assume: assumes x is sorted and equally spaced
    :return: interpolated y array
    """
    y_fun = interp1d(x[~np.isnan(y)], y[~np.isnan(y)], kind=kind, fill_value="extrapolate", assume_sorted=assume)
    return y_fun(x)


def pd_interp(df: pd.DataFrame, interp_column: str, at: np.array) -> pd.DataFrame:
    """Resamples DataFrame with Scipy's interp1d, this was more performant than the pandas one for some reason

    :param df: original sessiondata structure
    :param interp_column: column to interpolate on
    :param at: column to interpolate on
    :return: interpolated DataFrame
    """
    interp_df = pd.DataFrame()
    for col in df:
        f = interp1d(df[interp_column], df[col], bounds_error=False, fill_value="extrapolate")
        interp_df[col] = f(at)
    interp_df[interp_column] = at
    return interp_df


def merge_chars(chars: tuple) -> str:
    """Merges list of binary characters to single string

    :param chars: list of binary characters
    :return: string of concatenated characters
    """
    return ''.join([char.decode("utf-8") for char in chars])


def find_peaks(data: pd.Series, cutoff: float = 1.0, minpeak: float = 5.0, min_dist: int = 5) -> dict:
    """Finds positive peaks in signal and returns indices of start and stop

    :param data: any signal that contains peaks above minpeak that dip below cutoff
    :param cutoff: where the peak gets cut off at the bottom, basically a hysteresis band
    :param minpeak: minimum peak height of wave
    :param min_dist: minimum sample distance between peaks, can be used to speed up algorithm
    :return: nested dictionary with start, end, and peak **index** of each peak
    """
    peaks = {"start": [], "stop": [], "peak": []}
    tmp = {"start": None, "stop": None}

    data = np.array(data)  # coercing to an array if necessary
    data_slice = np.nonzero(data > minpeak)[0]  # indices of nonzero values
    data_slice = data_slice[np.diff(data_slice, append=10e100) > min_dist]  # remove duplicate samples from push
    for prom in np.nditer(data_slice):
        tmp["stop"] = next((idx for idx, value in enumerate(data[prom:]) if value < cutoff), None)
        tmp["start"] = next((idx for idx, value in enumerate(reversed(data[:prom])) if value < cutoff), None)
        if tmp["stop"] and tmp["start"]:  # did we find a start and stop?
            peaks["stop"].append(tmp["stop"] + prom - 1)
            peaks["start"].append(prom - tmp["start"])
    peaks = {key: np.unique(value) for key, value in peaks.items()}  # remove possible duplicates
    peaks["peak"] = [np.argmax(data[start:stop + 1]) + start for start, stop in zip(peaks["start"], peaks["stop"])]
    return peaks


def coast_down_velocity(t: np.array, v0: float, c1: float, c2: float, m: float) -> np.array:
    """	Solution for the non-linear differential equation M(dv/dt) + c1*v**2 + c2 = 0. Returns the instantaneous
    velocity decreasing with time (t) for the friction coefficients c1 and c2 for an object with a fixed mass (M)"""
    return np.sqrt(c2 / c1) * np.tan(np.arctan(v0 * np.sqrt(c1 / c2)) - t * (np.sqrt(c1 * c2) / m))


def nonlinear_fit_coast_down(time: np.array, vel: np.array, total_weight: float) -> tuple:
    """Performs a nonlinear fit on coast-down data, returning c1 and c2"""
    # try to determine c1 and c2 with curve_fit for non-linear approach
    initial_velocity = vel[0]
    vel_func = lambda t, c1, c2: coast_down_velocity(t, initial_velocity, c1, c2, total_weight)  # lock variables
    (coef1, coef2), pcov = curve_fit(vel_func, time, vel, bounds=(0, [1, 20]))  # 'trf' method but 'lm' would also work
    return coef1, coef2


def mask_from_iterable(array: np.array, floor_values: list, ceil_values: list) -> np.array:
    """Combines multiple masks from iterable into one mask (e.g. can be used to select multiple time slices)
    :param array: array to apply mask on
    :param floor_values: minimum values in array
    :param ceil_values: maximum values in array
    :return:
    """
    mask = np.full(array.shape, False)  # start off with all data deselected
    for ceil, floor in zip(ceil_values, floor_values):
        mask = mask | ((array > floor) & (array < ceil))
    return mask


def calc_inertia(weight: float = 0.8, radius: float = 0.295, length: float = 0.675, period: float = 1.0) -> float:
    """Calculates the inertia from a trifilar pendulum test

    :param weight: total mass of the object
    :param radius: radius of the object
    :param length: length of the trifilar pendulum
    :param period: observed oscillation period
    :return: inertia [kgm2]
    """
    return (weight*9.81*radius**2*period**2)/(4*np.pi**2*length)


def zerocross1d(x: np.array, y: np.array, indices: bool = False):
    """Find the zero crossing points in 1d data.

      Find the zero crossing events in a discrete data set. Linear interpolation is used to determine the actual
      locations of the zero crossing between two data points showing a change in sign. Data point which are zero
      are counted in as zero crossings if a sign change occurs across them. Note that the first and last data point will
      not be considered whether or not they are zero.
      :param x: time variable
      :param y: y variable
      :param indices: return indices or not
      :return: position in time and optionally the index of the sample before the zero-crossing
    """
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

    if not indices:
        return zeroc_xvalues
    else:
        return zeroc_xvalues, zeroc_indices


def camel_to_snake(name: str):
    """Turns CamelCased text into snake_cased text.

    :param name: StringToConvert
    :return: converted_string
    """
    s1 = re.sub('(.)([A-Z][a-z]+)', r'\1_\2', name)
    return re.sub('([a-z0-9])([A-Z])', r'\1_\2', s1).lower()


def find_nearest(array: np.array, value: float, index: bool = False):
    """Find the nearest value in an array or the index thereof.

    :param array: array which has to be searched
    :param value: value that you are looking for
    :param index: whether or not you want the index
    :return: the closest value or the index of the value
    """
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return idx if index else array[idx]
