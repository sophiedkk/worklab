"""
-Kinetics module-
Description: Contains functions for working with measurement wheel (Optipush and SMARTwheel) and ergometer (Esseda) data
You will usually only need the top-level function autoprocess.
Author:     Rick de Klerk
Contact:    r.de.klerk@umcg.nl
Company:    University Medical Center Groningen
License:    GNU GPLv3.0
Date:       27/06/2019
"""
import numpy as np
import pandas as pd
from scipy.integrate import cumtrapz

from .utils import lowpass_butter, find_peaks


def auto_process(data, wheelsize=0.31, rimsize=0.27, sfreq=200, co_f=15, ord_f=2, co_s=6, ord_s=2, force=True,
                 speed=True, variable="torque", cutoff=0.0, minpeak=5.0):
    """Contains all signal processing steps in fixed order. It is advised to use this function for all (pre-)processing.
    If needed take a look at a specific function to see how it works.

        :param data: raw ergometer or measurement wheel data
        :param wheelsize: wheel radius in m
        :param rimsize: rim radius in m
        :param sfreq: sample frequency
        :param co_f: cutoff frequency for force filter
        :param ord_f: order for force filter
        :param co_s: cutoff frequency for speed filter
        :param ord_s: order for speed filter
        :param force: force filter toggle
        :param speed: speed filter toggle
        :param variable: variable used for peak (push) detection
        :param cutoff: noise level for peak (push) detection
        :param minpeak: min peak for peak (push) detection
        :return: filtered and processed data, and push-by-push data
        """
    if "right" in data:
        data = filter_ergo_data(data, co_f, ord_f, force, co_s, ord_s, speed)
        data = process_ergo_data(data, wheelsize, rimsize)
        pushes = push_by_push_ergo(data, variable, cutoff, minpeak)
    else:
        data = filter_mw_data(data, sfreq, co_f, ord_f, force, co_s, ord_s, speed)
        data = process_mw_data(data, wheelsize, rimsize, sfreq)
        pushes = push_by_push_mw(data, variable, cutoff, minpeak)
    return data, pushes


def filter_mw_data(data, sfreq=200, co_f=15, ord_f=2, force=True, co_s=6, ord_s=2, speed=True):
    """Filters measurement wheel data; should be used before processing

    :param data: measurement wheel data
    :param sfreq:  specific sample freq for measurement wheel
    :param co_f: cut off frequency for force related variables
    :param ord_f: filter order for force related variables
    :param force: filter force toggle
    :param co_s: cut off frequency for speed related variables
    :param ord_s: filter order for speed related variables
    :param speed: filter speed toggle
    :return: same data but filtered
    """
    if force:
        frel = ["fx", "fy", "fz", "mx", "my", "torque"]
        for var in frel:
            data[var] = lowpass_butter(data[var], cutoff=co_f, order=ord_f, sfreq=sfreq)
    if speed:
        data["angle"] = lowpass_butter(data["angle"], cutoff=co_s, order=ord_s, sfreq=sfreq)
    return data


def filter_ergo_data(data, co_f=15, ord_f=2, force=True, co_s=6, ord_s=2, speed=True):
    """Filters ergometer data; should be used before processing

    :param data: measurement wheel data
    :param co_f: cut off frequency for force related variables
    :param ord_f: filter order for force related variables
    :param force: filter force toggle
    :param co_s: cut off frequency for speed related variables
    :param ord_s: filter order for speed related variables
    :param speed: filter speed toggle
    :return: same data but filtered
    """
    sfreq = 100
    for side in data:
        if force:
            data[side]["force"] = lowpass_butter(data[side]["force"], cutoff=co_f, order=ord_f, sfreq=sfreq)
        if speed:
            data[side]["speed"] = lowpass_butter(data[side]["speed"], cutoff=co_s, order=ord_s, sfreq=sfreq)
    return data


def process_ergo_data(data: dict, wheelsize: float = 0.31, rimsize: float = 0.275) -> dict:
    """Basic processing for ergometer data (e.g. speed to distance)

    :param data: ergometer data dictionary
    :param wheelsize: wheel radius in m
    :param rimsize: handrim radius in m
    :return: processed ergometer data dictionary
    """
    sfreq = 100  # ergometer is always 100Hz
    for side in data:
        data[side]["torque"] = data[side]["force"] * wheelsize
        data[side]["acc"] = np.gradient(data[side]["speed"]) * sfreq
        data[side]["power"] = data[side]["speed"] * data[side]["force"]
        data[side]["dist"] = cumtrapz(data[side]["speed"], initial=0.0) / sfreq
        data[side]["work"] = data[side]["power"] / sfreq
        data[side]["uforce"] = data[side]["force"] * (wheelsize / rimsize)
        data[side]["aspeed"] = data[side]["speed"] / wheelsize
        data[side]["angle"] = cumtrapz(data[side]["aspeed"], initial=0.0) / sfreq
    return data


def process_mw_data(data, wheelsize: float = 0.31, rimsize: float = 0.275, sfreq: int = 200) -> pd.DataFrame:
    """Basic processing for measurment wheel data (e.g. speed to distance)

    :param data: measurement wheel dataframe
    :param wheelsize: wheel radius in m
    :param rimsize: handrim radius in m
    :param sfreq: sample frequency in Hz
    :return: processed measurement wheel dataframe
    """
    data["aspeed"] = np.gradient(data["angle"]) / (1 / sfreq)
    data["speed"] = data["aspeed"] * wheelsize
    data["dist"] = cumtrapz(data["speed"], initial=0.0) * (1 / sfreq)
    data["acc"] = np.gradient(data["speed"]) / (1 / sfreq)
    data["ftot"] = (data["fx"] ** 2 + data["fy"] ** 2 + data["fz"] ** 2) ** 0.5
    data["uforce"] = data["torque"] / rimsize
    data["force"] = data["uforce"] / (wheelsize / rimsize)
    data["power"] = data["torque"] * data["aspeed"]
    data["work"] = data["power"] * (1 / sfreq)
    return data


def push_by_push_ergo(data: dict, variable: str = "torque", cutoff: float = 0.0, minpeak: float = 5.0) -> dict:
    """Push detection and push-by-push analysis for ergometer data

    :param data: ergometer data dictionary
    :param variable: variable used for peak (push) detection
    :param cutoff: noise level for peak (push) detection
    :param minpeak: minimum peak (push) height
    :return: push-by-push data dictionary with left: pd.DataFrame and right: pd.DataFrame
    """
    pbp = {"left": [], "right": []}
    for side in data:
        pbp[side] = find_peaks(data[side][variable], cutoff, minpeak)
        for ind, (start, stop, peak) in enumerate(zip(pbp[side]["start"], pbp[side]["stop"], pbp[side]["peak"])):
            pbp[side]["tstart"].append(data[side]["time"][start])
            pbp[side]["tstop"].append(data[side]["time"][stop])
            pbp[side]["tpeak"].append(data[side]["time"][peak])
            pbp[side]["cangle"].append(data[side]["angle"][stop] - data[side]["angle"][start])
            pbp[side]["ptime"].append(pbp[side]["tstop"][-1] - pbp[side]["tstart"][-1])
            stop += 1  # inclusive of last sample for slicing
            pbp[side]["meanpower"].append(np.mean(data[side]["power"][start:stop]))
            pbp[side]["maxpower"].append(np.max(data[side]["power"][start:stop]))
            pbp[side]["meantorque"].append(np.mean(data[side]["torque"][start:stop]))
            pbp[side]["maxtorque"].append(np.max(data[side]["torque"][start:stop]))
            pbp[side]["meanforce"].append(np.mean(data[side]["uforce"][start:stop]))
            pbp[side]["maxforce"].append(np.max(data[side]["uforce"][start:stop]))
            pbp[side]["work"].append(np.cumsum(data[side]["work"][start:stop]).iloc[-1])
            pbp[side]["slope"].append(pbp[side]["maxtorque"][-1] /
                                      (pbp[side]["tpeak"][-1] - pbp[side]["tstart"][-1]))
            if start != pbp[side]["start"][0]:  # only after first push
                pbp[side]["ctime"].append(pbp[side]["tstart"][-1] - pbp[side]["tstart"][-2])
                pbp[side]["reltime"].append(pbp[side]["ptime"][-2] / pbp[side]["ctime"][-1] * 100)
        pbp[side]["ctime"].append(np.NaN)
        pbp[side]["reltime"].append(np.NaN)
        pbp[side] = pd.DataFrame(pbp[side])
    print("\n" + "=" * 80 + f"\nFound left: {len(pbp['left'])} and right: {len(pbp['right'])} pushes!\n"
          + "=" * 80 + "\n")
    return pbp


def push_by_push_mw(data, variable: str = "torque", cutoff: float = 0.0, minpeak: float = 5.0) -> pd.DataFrame:
    """Push detection and push-by-push analysis for measurement wheel data

    :param data: measurement wheel dataframe
    :param variable: variable used for peak (push) detection
    :param cutoff: noise level for peak (push) detection
    :param minpeak: minimum peak (push) height
    :return: push-by-push dataframe
    """
    pbp = find_peaks(data[variable], cutoff, minpeak)
    for ind, (start, stop, peak) in enumerate(zip(pbp["start"], pbp["stop"], pbp["peak"])):  # for each push
        pbp["tstart"].append(data["time"][start])
        pbp["tstop"].append(data["time"][stop])
        pbp["tpeak"].append(data["time"][peak])
        pbp["cangle"].append(data["angle"][stop] - data["angle"][start])
        pbp["ptime"].append(pbp["tstop"][-1] - pbp["tstart"][-1])
        stop += 1  # inclusive of last sample for slices
        pbp["meanpower"].append(np.mean(data["power"][start:stop]))
        pbp["maxpower"].append(np.max(data["power"][start:stop]))
        pbp["meantorque"].append(np.mean(data["torque"][start:stop]))
        pbp["maxtorque"].append(np.max(data["torque"][start:stop]))
        pbp["meanforce"].append(np.mean(data["uforce"][start:stop]))
        pbp["maxforce"].append(np.max(data["uforce"][start:stop]))
        pbp["work"].append(np.cumsum(data["work"][start:stop]).iloc[-1])
        pbp["feff"].append(np.mean(data["uforce"][start:stop] / data["ftot"][start:stop]) * 100)
        pbp["slope"].append(pbp["maxtorque"][-1] / (pbp["tpeak"][-1] - pbp["tstart"][-1]))
        if start != pbp["start"][0]:  # only after first push
            pbp["ctime"].append(pbp["tstart"][-1] - pbp["tstart"][-2])
            pbp["reltime"].append(pbp["ptime"][-2] / pbp["ctime"][-1] * 100)
    pbp["ctime"].append(np.NaN)
    pbp["reltime"].append(np.NaN)
    pbp = pd.DataFrame(pbp)
    print("\n" + "=" * 80 + f"\nFound {len(pbp)} pushes!\n" + "=" * 80 + "\n")
    return pbp
