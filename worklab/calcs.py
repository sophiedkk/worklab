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


import numpy as np
from scipy import interpolate
from scipy.integrate import cumtrapz
from scipy.signal import butter, filtfilt
from .formats import get_pbp_format, get_sum_format


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
        for side in data.keys():
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
            if "uforce" in data[side].keys():  # LEM
                data[side]["force"] = data[side]["uforce"] / (wheelsize/rimsize)
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
        tmp["end"] = next((index for index, value in enumerate(data[prom:-1]) if value < cutoff), None)
        tmp["start"] = next((index for index, value in enumerate(reversed(data[0:prom])) if value < cutoff), None)
        if tmp["end"] and tmp["start"]:
            if neg:  # optionally
                tmp["endneg"] = next((index for index, value in enumerate(data[tmp["end"]+prom:-1])
                                      if value > cutoff), None)
                tmp["startneg"] = next((index for index, value in enumerate(reversed(data[0:prom-tmp["start"]-1]))
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

    tmp = 0  # needed for cycle time
    if "right" in data:  # ergometer data
        pbp = {"left": [], "right": []}
        for side in data:  # left and right side
            pbp[side] = get_pbp_format()
            pbp[side]["start"] = pushes[side]["start"]
            pbp[side]["stop"] = pushes[side]["end"]
            if "startneg" in pushes[side]:
                pbp[side]["neg"] = []
            for ind, (start, stop) in enumerate(zip(pbp[side]["start"], pbp[side]["stop"])):  # for each push
                pbp[side]["tstart"].append(data[side]["time"][start])
                pbp[side]["tstop"].append(data[side]["time"][stop])
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
                                          (pbp[side]["tstop"][-1] - pbp[side]["tstart"][-1]))
                if start != pushes[side]["start"][0]:  # only after first push
                    pbp[side]["ctime"].append(pbp[side]["tstart"][-1] - tmp)
                    pbp[side]["reltime"].append(pbp[side]["ptime"][-2] / pbp[side]["ctime"][-1] * 100)
                tmp = pbp[side]["tstart"][-1]  # hold for cycle time calc
                if "startneg" in pushes[side].keys():
                    pbp[side]["neg"].append((np.cumsum(data[side]["work"][pushes[side]["startneg"][ind]:
                                                                          pushes[side]["start"][ind]])[-1]))
                    pbp[side]["neg"][ind] += np.cumsum(data[side]["work"][pushes[side]["end"][ind]:
                                                                          pushes[side]["endneg"][ind]])[-1]
            pbp[side] = {dkey: np.asarray(pbp[side][dkey]) for dkey in pbp[side]}
    else:
        pbp = get_pbp_format()
        pbp["start"] = pushes["start"]
        pbp["stop"] = pushes["end"]
        if "startneg" in pushes:
            pbp["neg"] = []
        for ind, (start, stop) in enumerate(zip(pbp["start"], pbp["stop"])):  # for each push
            pbp["tstart"].append(data["time"][start])
            pbp["tstop"].append(data["time"][stop])
            pbp["ptime"].append(pbp["tstop"][-1]-pbp["tstart"][-1])
            pbp["pout"].append(np.mean(data["power"][start:stop]))
            pbp["maxpout"].append(np.max(data["power"][start:stop]))
            pbp["maxtorque"].append(np.max(data["torque"][start:stop]))
            pbp["meantorque"].append(np.mean(data["torque"][start:stop]))
            pbp["cangle"].append(data["angle"][stop] - data["angle"][start])
            pbp["work"].append(np.cumsum(data["work"][start:stop])[-1])
            pbp["fpeak"].append(np.max(data["uforce"][start:stop]))
            pbp["fmean"].append(np.mean(data["uforce"][start:stop]))
            pbp["slope"].append(pbp["maxtorque"][-1]/(pbp["tstop"][-1]-pbp["tstart"][-1]))
            if start != pushes["start"][0]:  # only after first push
                pbp["ctime"].append(pbp["tstart"][-1] - tmp)
                pbp["reltime"].append(pbp["ptime"][-2]/pbp["ctime"][-1] * 100)
            tmp = pbp["tstart"][-1]  # hold for cycle time calc
            if "startneg" in pushes:
                pbp["neg"].append((np.cumsum(data["work"][pushes["startneg"][ind]:pushes["start"][ind]])[-1]))
                pbp["neg"][ind] += np.cumsum(data["work"][pushes["end"][ind]:pushes["endneg"][ind]])[-1]
        pbp = {dkey: np.asarray(pbp[dkey]) for dkey in pbp}
    return pbp


def make_calibration_spline(calibration_points):
    spl_line = {"left": [], "right": []}
    for side in spl_line:
        x = np.arange(0, 10)
        spl = interpolate.InterpolatedUnivariateSpline(x, calibration_points[side], k=2)
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
            summary[side] = {dkey: [np.mean(pbp[side][dkey]), np.std(pbp[side][dkey])]
                             for dkey in summary[side].keys()}
    else:
        summary = get_sum_format()
        summary = {dkey: [np.mean(pbp[dkey]), np.std(pbp[dkey])] for dkey in summary.keys()}
    return summary
