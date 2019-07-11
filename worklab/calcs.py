"""
-Ergometer/measurement processing-
Description: Basic functions for ergometer data processing such as peak detection,
general conversions, push by push analysis, summary statistics, and data exports
Author:     Rick de Klerk
Contact:    r.de.klerk@umcg.nl
Company:    University Medical Center Groningen
License:    GNU GPLv3.0
Date:       26/03/2018
"""
from collections import defaultdict
from warnings import warn

import numpy as np
import pandas as pd
from scipy.integrate import cumtrapz
from scipy.interpolate import InterpolatedUnivariateSpline, interp1d
from scipy.signal import butter, filtfilt

from .formats import get_sum_format


def autoprocess(data, wheelsize=0.31, rimsize=0.27, co_f=15, ord_f=2, sfreq=200, cutoff=1.0, minpeak=5.0):
    """Contains all signal processing steps in fixed order
    Input
        - data: raw data dict of ergometer or measurement wheel
        - rest: see other functions for specific kwargs
    Output
        - data: filtered and processed data dictionary
        - pbp: push-by-push data dictionary"""

    data = filter_data(data, co_f=co_f, ord_f=ord_f, sfreq=sfreq)
    data = process_data(data, wheelsize, rimsize, sfreq)
    pushes = find_pushes(data, cutoff=cutoff, minpeak=minpeak, neg=False)
    pbp = push_by_push(data, pushes)
    return data, pushes, pbp


def slice_data(data, start=0, end=100):
    """Slices wheelchair ergometer or measurement wheel data dictionary

    Input
        - data: wheelchair ergometer or measurement wheel data dict
        - start: starting index
        - end: end index
    Output
        - data: sliced data file from start *till* end"""

    if "right" in data:  # ergometer data
        for side in data:
            data[side] = {dkey: data[side][dkey][start:end] for dkey in data[side]}
    else:
        data = {dkey: data[dkey][start:end] for dkey in data}
    return data


def process_data(data, wheelsize=0.31, rimsize=0.27, sfreq=200):
    """Does all basic calculations and conversions on ergometer or measurement wheel data

    Input
        - data: filtered ergometer or measurement wheel data dict
    Optional
        - wheelsize: radius of wheelchair wheel
        - rimsize: handrim radius
        - sfreq: specific sample frequency of measurement wheel
    Output
        - data: extended data dictionary with regular outcome values"""

    if "right" in data:  # ergometer data
        sfreq = 100  # overrides default; ergometer is always 100Hz
        for side in data:
            data[side]["torque"] = data[side]["force"] * wheelsize
            data[side]["acc"] = np.gradient(data[side]["speed"]) / (1/sfreq)
            data[side]["power"] = data[side]["speed"] * data[side]["force"]
            data[side]["dist"] = cumtrapz(data[side]["speed"], initial=0.0) * (1/sfreq)
            data[side]["work"] = data[side]["power"] * (1/sfreq)
            data[side]["uforce"] = data[side]["force"] * (wheelsize/rimsize)
            data[side]["aspeed"] = data[side]["speed"] / wheelsize
            data[side]["angle"] = cumtrapz(data[side]["aspeed"], initial=0.0) * (1/sfreq)
    else:
        data["aspeed"] = np.gradient(data["angle"]) / (1/sfreq)
        data["speed"] = data["aspeed"] * wheelsize
        data["dist"] = cumtrapz(data["speed"], initial=0.0) * (1/sfreq)
        data["acc"] = np.gradient(data["speed"]) / (1/sfreq)
        data["ftot"] = (data["fx"]**2 + data["fy"]**2 + data["fz"]**2)**0.5
        data["uforce"] = data["torque"] / rimsize
        data["force"] = data["uforce"] / (wheelsize/rimsize)
        data["power"] = data["torque"] * data["aspeed"]
        data["work"] = data["power"] * (1/sfreq)
    return data


# noinspection PyTupleAssignmentBalance
def filter_data(data, sfreq=200, co_f=15, ord_f=2, force=True, co_s=6, ord_s=2, speed=True):
    """Filters ergometer or measurement wheel data dict; should be used before processing

    Input
        - data:  ergometer or measurement wheel data dict
    Optional
        - sfreq: specific sample freq for measurement wheel
        - co_f: cut off frequency for force related variables
        - ord_f: filter order for force related variables
        - force: filter force toggle
        - co_s: cut off frequency for speed related variables
        - ord_s: filter order for speed related variables
        - speed: filter speed toggle
    Output
        - data: same data file but filtered"""

    if "right" in data:  # ergometer data
        sfreq = 100  # overrides default; ergometer is always 100Hz
        for side in data:
            if force:
                b, a = butter(ord_f, co_f / (0.5 * sfreq), 'low')
                if "force" in data[side]:  # not from LEM
                    data[side]["force"] = filtfilt(b, a, data[side]["force"])
                else:  # from LEM
                    data[side]["uforce"] = filtfilt(b, a, data[side]["uforce"])
            if speed:
                b, a = butter(ord_s, co_s / (0.5 * sfreq), 'low')
                data[side]["speed"] = filtfilt(b, a, data[side]["speed"])
    else:
        if force:
            frel = ["fx", "fy", "fz", "mx", "my", "torque"]
            for var in frel:
                b, a = butter(ord_f, co_f / (0.5 * sfreq), 'low')
                data[var] = filtfilt(b, a, data[var])
        if speed:
            b, a = butter(ord_s, co_s / (0.5 * sfreq), 'low')
            data["angle"] = filtfilt(b, a, data["angle"])
    return data


def find_peaks(data, cutoff=1.0, minpeak=5.0, neg=False):
    """Finds positive peaks in signal and returns indices

    Input
        - Data: any signal that contains peaks above minpeak that dip below cutoff
    Optional
        - cutoff: where the push gets cut off at the bottom
        - minpeak: minimum peak height of push
        - neg: whether to include the negative portion at start/end of each peak
    Returns
        - pushes nested dictionary with start, end, and peak **index** of each peak"""

    peaks = {"start": [], "end": [], "peak": []}
    tmp = {"start": None, "end": None, "peak": None}
    if neg:  # optionally
        peaks.update({"endneg": [], "startneg": []})
        tmp.update({"endneg": None, "startneg": None})

    data_slice = np.nonzero(data > minpeak)
    for prom in np.nditer(data_slice):
        if peaks["end"]:
            if prom < peaks["end"][-1]:
                continue  # skip if a push has already been found for that index
        tmp["end"] = next((index for index, value in enumerate(data[prom:]) if value < cutoff), None)
        tmp["start"] = next((index for index, value in enumerate(reversed(data[:prom])) if value < cutoff), None)
        if tmp["end"] and tmp["start"]:
            if neg:  # optionally
                tmp["endneg"] = next((index for index, value in enumerate(data[tmp["end"]+prom:])
                                      if value > cutoff), None)
                tmp["startneg"] = next((index for index, value in enumerate(reversed(data[:prom-tmp["start"]-1]))
                                        if value > cutoff), None)
                if tmp["endneg"] and tmp["startneg"]:
                    peaks["end"].append(tmp["end"] + prom - 1)
                    peaks["start"].append(prom - tmp["start"])
                    peaks["endneg"].append(tmp["endneg"] + peaks["end"][-1])
                    peaks["startneg"].append(peaks["start"][-1] - tmp["startneg"])
            else:
                peaks["end"].append(tmp["end"] + prom - 1)
                peaks["start"].append(prom - tmp["start"])
    peaks["peak"] = [np.argmax(data[start:end]) + start for start, end in zip(peaks["start"], peaks["end"])]
    peaks = {key: np.unique(peak) for key, peak in peaks.items()}
    return peaks


def find_pushes(data, cutoff=1.0, minpeak=5.0, neg=False):
    """Uses find peak method to find **index** of pushes

    Input
        - data: processed ergometer or measurement wheel data dict
    Optional
        - cutoff: where the push gets cut off at the bottom
        - minpeak: minimum peak height of push
        - var: what variable should be used to find pushes
        - neg: whether to include the negative portion at each push
    Returns
        - pushes nested dictionary with start, end, and peak **index** of each push"""

    if "right" in data:  # ergometer data
        pushes = {"left": find_peaks(data["left"]["torque"], cutoff=cutoff, minpeak=minpeak, neg=neg),
                  "right": find_peaks(data["right"]["torque"], cutoff=cutoff, minpeak=minpeak, neg=neg)}
    else:
        pushes = find_peaks(data["torque"], cutoff=cutoff, minpeak=minpeak, neg=neg)
    return pushes


# noinspection PyTypeChecker
def push_by_push(data, pushes):
    """Calculates push-by-push statistics such as push time and power per push.

    Input
        - data: processed ergometer or measurement wheel data dict
        - pushes: start, end, peak indices of all pushes
    Returns
        - push-by-push nested dictionary with outcome parameters in arrays"""

    if "right" in data:  # ergometer data
        pbp = {"left": [], "right": []}
        for side in data:  # left and right side
            pbp[side] = defaultdict(list)
            pbp[side]["start"] = pushes[side]["start"]
            pbp[side]["stop"] = pushes[side]["end"]
            pbp[side]["peak"] = pushes[side]["peak"]
            for ind, (start, stop, peak) in enumerate(zip(pbp[side]["start"], pbp[side]["stop"], pbp[side]["peak"])):
                stop += 1  # inclusive of last sample
                pbp[side]["tstart"].append(data[side]["time"][start])
                pbp[side]["tstop"].append(data[side]["time"][stop])
                pbp[side]["tpeak"].append(data[side]["time"][peak])
                pbp[side]["ptime"].append(pbp[side]["tstop"][-1] - pbp[side]["tstart"][-1])
                pbp[side]["pout"].append(np.mean(data[side]["power"][start:stop]))
                pbp[side]["maxpout"].append(np.max(data[side]["power"][start:stop]))
                pbp[side]["maxtorque"].append(np.max(data[side]["torque"][start:stop]))
                pbp[side]["meantorque"].append(np.mean(data[side]["torque"][start:stop]))
                pbp[side]["cangle"].append(data[side]["angle"][stop] - data[side]["angle"][start])
                pbp[side]["work"].append(np.cumsum(data[side]["work"][start:stop])[-1])
                pbp[side]["fpeak"].append(np.max(data[side]["uforce"][start:stop]))
                pbp[side]["fmean"].append(np.mean(data[side]["uforce"][start:stop]))
                pbp[side]["slope"].append(pbp[side]["maxtorque"][-1] /
                                          (pbp[side]["tpeak"][-1] - pbp[side]["tstart"][-1]))
                if start != pushes[side]["start"][0]:  # only after first push
                    pbp[side]["ctime"].append(pbp[side]["tstart"][-1] - pbp[side]["tstart"][-2])
                    pbp[side]["reltime"].append(pbp[side]["ptime"][-2] / pbp[side]["ctime"][-1] * 100)
                    pbp[side]["negwork/cycle"].append(np.cumsum(data[side]["work"][pbp[side]["start"][-2]:
                                                                                   pbp[side]["stop"][-1] + 1])[-1])
                if "startneg" in pushes[side]:
                    pbp[side]["neg"].append((np.cumsum(data[side]["work"][pushes[side]["startneg"][ind]:
                                                                          pushes[side]["start"][ind]])[-1]))
                    pbp[side]["neg"][ind] += np.cumsum(data[side]["work"][pushes[side]["end"][ind]:
                                                                          pushes[side]["endneg"][ind]])[-1]
            pbp[side] = pd.DataFrame.from_dict(pbp[side], orient='index').T
        return pbp
    else:  # measurement wheel data
        pbp = defaultdict(list)
        pbp["start"] = pushes["start"]
        pbp["stop"] = pushes["end"]
        pbp["peak"] = pushes["peak"]
        for ind, (start, stop, peak) in enumerate(zip(pbp["start"], pbp["stop"], pbp["peak"])):  # for each push
            stop += 1  # inclusive of last sample
            pbp["tstart"].append(data["time"][start])
            pbp["tstop"].append(data["time"][stop])
            pbp["tpeak"].append(data["time"][peak])
            pbp["ptime"].append(pbp["tstop"][-1]-pbp["tstart"][-1])
            pbp["pout"].append(np.mean(data["power"][start:stop]))
            pbp["maxpout"].append(np.max(data["power"][start:stop]))
            pbp["maxtorque"].append(np.max(data["torque"][start:stop]))
            pbp["meantorque"].append(np.mean(data["torque"][start:stop]))
            pbp["cangle"].append(data["angle"][stop] - data["angle"][start])
            pbp["work"].append(np.cumsum(data["work"][start:stop])[-1])
            pbp["fpeak"].append(np.max(data["uforce"][start:stop]))
            pbp["fmean"].append(np.mean(data["uforce"][start:stop]))
            pbp["feff"].append(np.mean(data["uforce"][start:stop] / ((data["fx"][start:stop]**2 +
                                                                      data["fy"][start:stop]**2 +
                                                                      data["fz"][start:stop]**2)**0.5)) * 100)
            pbp["slope"].append(pbp["maxtorque"][-1]/(pbp["tpeak"][-1]-pbp["tstart"][-1]))
            if start != pushes["start"][0]:  # only after first push
                pbp["ctime"].append(pbp["tstart"][-1] - pbp["tstart"][-2])
                pbp["reltime"].append(pbp["ptime"][-2]/pbp["ctime"][-1] * 100)
                pbp["negwork/cycle"].append(np.cumsum(data["work"][pbp["start"][-2]:pbp["stop"][-1] + 1])[-1])
            if "startneg" in pushes:
                pbp["neg"].append((np.cumsum(data["work"][pushes["startneg"][ind]:pushes["start"][ind]])[-1]))
                pbp["neg"][ind] += np.cumsum(data["work"][pushes["end"][ind]:pushes["endneg"][ind]])[-1]
    return pd.DataFrame.from_dict(pbp, orient='index').T


def make_calibration_spline(calibration_points):
    spl_line = {"left": [], "right": []}
    for side in spl_line:
        x = np.arange(0, 10)
        spl = InterpolatedUnivariateSpline(x, calibration_points[side], k=2)
        spl_line[side] = spl(np.arange(0, 9.01, 0.01))  # Spline with 0.01 increments
        spl_line[side] = np.append(spl_line[side], np.full(99, spl_line[side][-1]))
    return spl_line


def summary_statistics(pbp):
    """Gets the mean and standard deviation of all push-by-push parameters.

    Input
        - pbp: push-by-push nested dictionary
    Returns
        - summary: nested dictionary with means and standard deviations"""

    if "right" in pbp:  # ergometer data
        summary = {"left": get_sum_format(), "right": get_sum_format()}
        for side in pbp:
            summary[side] = {dkey: [np.mean(pbp[side][dkey]), np.std(pbp[side][dkey])] for dkey in summary[side]}
    else:
        summary = get_sum_format()
        summary = {dkey: [np.mean(pbp[dkey]), np.std(pbp[dkey])] for dkey in summary}
    return summary


def pd_interp(df, interp_column, at):
    """
    Resamples DataFrame with Scipy's interp1d
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


def resample_imu(sessiondata, samplefreq=400):
    """
    Resample all devices and sensors to new sample frequency. Translated from xio-Technologies.
    :param sessiondata: original sessiondata structure
    :param samplefreq: new intended sample frequency
    :return: resampled sessiondata
    """
    end_time = 0
    for device in sessiondata:
        for sensor in sessiondata[device]:
            max_time = sessiondata[device][sensor]["Time"].max()
            end_time = max_time if max_time > end_time else end_time

    new_time = np.arange(0, end_time, 1/samplefreq)

    for device in sessiondata:
        for sensor in sessiondata[device]:
            if sensor == "quaternion":  # TODO: xio-tech has TODO here to replace this part with slerp
                sessiondata[device][sensor] = pd_interp(sessiondata[device][sensor], "Time", new_time)
                sessiondata[device][sensor] *= (1 / np.linalg.norm(sessiondata[device][sensor], axis=0))
            elif sensor == "matrix":
                sessiondata[device].pop(sensor)
                warn("Rotation matrix cannot be resampled. This dataframe has been removed")
            else:
                sessiondata[device][sensor] = pd_interp(sessiondata[device][sensor], "Time", new_time)
    return sessiondata


def lowpass_butter(array, sfreq, co=20, order=2):
    """Butterworth filter that takes sample-freq, cutoff, and order as input."""
    # noinspection PyTupleAssignmentBalance
    b, a = butter(order, co / (0.5 * sfreq), 'low')
    return filtfilt(b, a, array)


def calc_wheelspeed(sessiondata, camber=15, wsize=0.31, wbase=0.60, sfreq=400):
    """
    Calculate wheelchair velocity based on NGIMU data, modifies dataframes inplace.
    :param sessiondata: original sessiondata structure
    :param camber: camber angle in degrees
    :param wsize: radius of the wheels
    :param wbase: width of wheelbase
    :param sfreq: sample frequency
    """
    frame = sessiondata["Frame"]["sensors"]  # view into dataframe, edits will be inplace
    left = sessiondata["Left"]["sensors"]  # most variables will be added to df except for some temp variables
    right = sessiondata["Right"]["sensors"]

    # Wheelchair camber correction
    deg2rad = np.pi / 180
    right["GyroCor"] = right["GyroscopeY"] + np.tan(camber * deg2rad) * (frame["GyroscopeZ"] * np.cos(camber * deg2rad))
    left["GyroCor"] = left["GyroscopeY"] - np.tan(camber * deg2rad) * (frame["GyroscopeZ"] * np.cos(camber * deg2rad))
    frame["GyroCor"] = (right["GyroCor"] + left["GyroCor"]) / 2

    # Calculation of wheelspeed and displacement
    right["GyroVel"] = right["GyroCor"] * wsize * deg2rad  # angular velocity to linear velocity
    right["GyroDist"] = cumtrapz(right["GyroVel"] / sfreq, initial=0.0)  # integral of velocity gives distance

    left["GyroVel"] = left["GyroCor"] * wsize * deg2rad
    left["GyroDist"] = cumtrapz(left["GyroVel"] / sfreq, initial=0.0)

    frame["CombVel"] = (right["GyroVel"] + left["GyroVel"]) / 2  # mean velocity
    frame["CombDist"] = (right["GyroDist"] + left["GyroDist"]) / 2  # mean velocity

    """Perform skid correction from Rienk vd Slikke, please refer and reference to: Van der Slikke, R. M. A., et. al. 
    Wheel skid correction is a prerequisite to reliably measure wheelchair sports kinematics based on inertial sensors. 
    Procedia Engineering, 112, 207-212."""
    frame["CombVelRight"] = np.gradient(right["GyroDist"]) * sfreq  # Calculate frame centre displacement
    frame["CombVelRight"] -= np.tan(np.deg2rad(frame["GyroscopeZ"]/sfreq)) * wbase/2 * sfreq
    frame["CombVelLeft"] = np.gradient(left["GyroDist"]) * sfreq
    frame["CombVelLeft"] += np.tan(np.deg2rad(frame["GyroscopeZ"]/sfreq)) * wbase/2 * sfreq

    r_ratio0 = np.abs(right["GyroVel"]) / (np.abs(right["GyroVel"]) + np.abs(left["GyroVel"]))  # Ratio left and right
    l_ratio0 = np.abs(left["GyroVel"]) / (np.abs(right["GyroVel"]) + np.abs(left["GyroVel"]))
    r_ratio1 = np.abs(np.gradient(left["GyroVel"])) / (np.abs(np.gradient(right["GyroVel"]))
                                                        + np.abs(np.gradient(left["GyroVel"])))
    l_ratio1 = np.abs(np.gradient(right["GyroVel"])) / (np.abs(np.gradient(right["GyroVel"]))
                                                        + np.abs(np.gradient(left["GyroVel"])))

    comb_ratio = (r_ratio0 * r_ratio1) / ((r_ratio0 * r_ratio1) + (l_ratio0 * l_ratio1))  # Combine speed ratios
    comb_ratio = lowpass_butter(comb_ratio, sfreq=sfreq, co=20)  # Filter the signal
    comb_ratio = np.clip(comb_ratio, 0, 1)  # clamp Combratio values, not in df
    frame["CombSkidVel"] = (frame["CombVelRight"] * comb_ratio) + (frame["CombVelLeft"] * (1-comb_ratio))
    frame["CombSkidDist"] = cumtrapz(frame["CombSkidVel"], initial=0.0) / sfreq  # Combined skid displacement
    return sessiondata
