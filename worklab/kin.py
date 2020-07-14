import numpy as np
import pandas as pd
from scipy.integrate import cumtrapz

from .utils import lowpass_butter, find_peaks


def auto_process(data, wheelsize=0.31, rimsize=0.27, sfreq=200, co_f=15, ord_f=2, co_s=6, ord_s=2, force=True,
                 speed=True, variable="torque", cutoff=0.0, minpeak=5.0):
    """
    Top level processing function that performs all processing steps for mw/ergo data.

    Contains all signal processing steps in fixed order. It is advised to use this function for all (pre-)processing.
    If needed take a look at a specific function to see how it works.

    Parameters
    ----------
    data : pd.DataFrame, dict
        raw ergometer or measurement wheel data
    wheelsize : float
        wheel radius [m]
    rimsize : float
        handrim radius [m]
    sfreq : int
        sample frequency [Hz]
    co_f : int
        cutoff frequency force filter [Hz]
    ord_f : int
        order force filter [..]
    co_s : int
        cutoff frequency force filter [Hz]
    ord_s : int
        order speed filter [..]
    force : bool
        force filter toggle, default is True
    speed : bool
        speed filter toggle, default is True
    variable : str
        variable name used for peak (push) detection
    cutoff : float
        noise level for peak (push) detection
    minpeak : float
        min peak height for peak (push) detection

    Returns
    -------
    data : pd.DataFrame, dict
    pushes : pd.DataFrame, dict

    See Also
    --------
    filter_mw, process_mw, push_by_push_mw, filter_ergo, process_ergo, push_by_push_ergo

    """
    if "right" in data:
        data = filter_ergo(data, co_f, ord_f, co_s, ord_s, force, speed)
        data = process_ergo(data, wheelsize, rimsize)
        pushes = push_by_push_ergo(data, variable, cutoff, minpeak)
    else:
        data = filter_mw(data, sfreq, co_f, ord_f, co_s, ord_s, force, speed)
        data = process_mw(data, wheelsize, rimsize, sfreq)
        pushes = push_by_push_mw(data, variable, cutoff, minpeak)
    return data, pushes


def filter_mw(data, sfreq=200., co_f=15., ord_f=2, co_s=6., ord_s=2, force=True, speed=True):
    """
    Filters measurement wheel data.

    Filters raw measurement wheel data. Should be used before further processing.

    Parameters
    ----------
    data : pd.DataFrame
        raw measurement wheel data
    sfreq : float
        sample frequency [Hz]
    co_f : float
        cutoff frequency force filter [Hz]
    ord_f : int
        order force filter [..]
    co_s : float
        cutoff frequency force filter [Hz]
    ord_s : int
        order speed filter [..]
    force : bool
        force filter toggle, default is True
    speed : bool
        speed filter toggle, default is True

    Returns
    -------
    data : pd.DataFrame
    Same data but filtered.

    See Also
    --------
    .utils.lowpass_butter

    """
    if force:
        frel = ["fx", "fy", "fz", "mx", "my", "torque"]
        for var in frel:
            data[var] = lowpass_butter(data[var], cutoff=co_f, order=ord_f, sfreq=sfreq)
    if speed:
        data["angle"] = lowpass_butter(data["angle"], cutoff=co_s, order=ord_s, sfreq=sfreq)
    return data


def filter_ergo(data, co_f=15., ord_f=2, co_s=6., ord_s=2, force=True, speed=True):
    """
    Filters ergometer data.

    Filters raw ergometer data. Should be used before further processing.

    Parameters
    ----------
    data : dict
        raw measurement wheel data
    co_f : float
        cutoff frequency force filter [Hz]
    ord_f : int
        order force filter [..]
    co_s : float
        cutoff frequency speed filter [Hz]
    ord_s : int
        order speed filter [..]
    force : bool
        force filter toggle, default is True
    speed : bool
        speed filter toggle, default is True

    Returns
    -------
    data : dict
        Same data but filtered.

    See Also
    --------
    .utils.lowpass_butter

    """
    sfreq = 100
    for side in data:
        if force:
            data[side]["force"] = lowpass_butter(data[side]["force"], cutoff=co_f, order=ord_f, sfreq=sfreq)
        if speed:
            data[side]["speed"] = lowpass_butter(data[side]["speed"], cutoff=co_s, order=ord_s, sfreq=sfreq)
    return data


def process_mw(data, wheelsize=0.31, rimsize=0.275, sfreq=200):
    """
    Basic processing for measurement wheel data.

    Basic processing for measurement wheel data (e.g. speed to distance). Should be performed after filtering.
    Added columns:

    +------------+----------------------+-----------+
    | Column     | Data                 | Unit      |
    +============+======================+===========+
    | aspeed     | angular velocity     | rad/s     |
    +------------+----------------------+-----------+
    | speed      | velocity             | m/s       |
    +------------+----------------------+-----------+
    | dist       | cumulative distance  | m         |
    +------------+----------------------+-----------+
    | acc        | acceleration         | m/s^2     |
    +------------+----------------------+-----------+
    | ftot       | total combined force | N         |
    +------------+----------------------+-----------+
    | uforce     | effective force      | N         |
    +------------+----------------------+-----------+
    | force      | force on wheel       | N         |
    +------------+----------------------+-----------+
    | power      | power                | W         |
    +------------+----------------------+-----------+
    | work       | instantanious work   | J         |
    +------------+----------------------+-----------+

    Parameters
    ----------
    data : pd.DataFrame
        raw measurement wheel data
    wheelsize : float
        wheel radius [m]
    rimsize : float
        handrim radius [m]
    sfreq : int
        sample frequency [Hz]

    Returns
    -------
    data : pd.DataFrame

    See Also
    --------
    .com.load_opti, .com.load_sw

    """
    data["aspeed"] = np.gradient(data["angle"]) * sfreq
    data["speed"] = data["aspeed"] * wheelsize
    data["dist"] = cumtrapz(data["speed"], initial=0.0) / sfreq
    data["acc"] = np.gradient(data["speed"]) * sfreq
    data["ftot"] = (data["fx"] ** 2 + data["fy"] ** 2 + data["fz"] ** 2) ** 0.5
    data["uforce"] = data["torque"] / rimsize
    data["feff"] = (data["uforce"] / data["ftot"]) * 100
    data["force"] = data["torque"] / wheelsize
    data["power"] = data["torque"] * data["aspeed"]
    data["work"] = data["power"] / sfreq
    return data


def process_ergo(data, wheelsize=0.31, rimsize=0.275):
    """
    Basic processing for ergometer data.

    Basic processing for ergometer data (e.g. speed to distance). Should be performed after filtering.
    Added columns:

    +------------+----------------------+-----------+
    | Column     | Data                 | Unit      |
    +============+======================+===========+
    | angle      | angle                | rad       |
    +------------+----------------------+-----------+
    | aspeed     | angular velocity     | rad/s     |
    +------------+----------------------+-----------+
    | acc        | acceleration         | m/s^2     |
    +------------+----------------------+-----------+
    | dist       | cumulative distance  | m         |
    +------------+----------------------+-----------+
    | power      | power                | W         |
    +------------+----------------------+-----------+
    | work       | instantanious work   | J         |
    +------------+----------------------+-----------+
    | uforce     | effective force      | N         |
    +------------+----------------------+-----------+
    | torque     | torque around wheel  | Nm        |
    +------------+----------------------+-----------+

    .. note:: the force column contains force on the wheels, uforce (user force) is force on the handrim

    Parameters
    ----------
    data : dict
        raw ergometer data
    wheelsize : float
        wheel radius [m]
    rimsize : float
        handrim radius [m]

    Returns
    -------
    data : dict

    See Also
    --------
    .com.load_esseda

    """
    sfreq = 100  # ergometer is always 100Hz
    for side in data:
        data[side]["aspeed"] = data[side]["speed"] / wheelsize
        data[side]["angle"] = cumtrapz(data[side]["aspeed"], initial=0.0) / sfreq
        data[side]["torque"] = data[side]["force"] * wheelsize
        data[side]["acc"] = np.gradient(data[side]["speed"]) * sfreq
        data[side]["power"] = data[side]["speed"] * data[side]["force"]
        data[side]["dist"] = cumtrapz(data[side]["speed"], initial=0.0) / sfreq
        data[side]["work"] = data[side]["power"] / sfreq
        data[side]["uforce"] = data[side]["force"] * (wheelsize / rimsize)
    return data


def push_by_push_mw(data, variable="torque", cutoff=0.0, minpeak=5.0):
    """
    Push-by-push analysis for measurement wheel data.

    Push detection and push-by-push analysis for measurement wheel data. Returns a pandas DataFrame with:

    +--------------------+----------------------+-----------+
    | Column             | Data                 | Unit      |
    +====================+======================+===========+
    | start/stop/peak    | respective indices   |           |
    +--------------------+----------------------+-----------+
    | tstart/tstop/tpeak | respective samples   | s         |
    +--------------------+----------------------+-----------+
    | cangle             | contact angle        | rad       |
    +--------------------+----------------------+-----------+
    | cangle_deg         | contact angle        | degrees   |
    +--------------------+----------------------+-----------+
    | meanpower          | power per push       | W         |
    +--------------------+----------------------+-----------+
    | maxpower           | peak power per push  | W         |
    +--------------------+----------------------+-----------+
    | meantorque         | torque per push      | Nm        |
    +--------------------+----------------------+-----------+
    | maxtorque          | peak torque per push | Nm        |
    +--------------------+----------------------+-----------+
    | meanforce          | mean force per push  | N         |
    +--------------------+----------------------+-----------+
    | maxforce           | peak force per push  | N         |
    +--------------------+----------------------+-----------+
    | meanuforce         | (rim) force per push | N         |
    +--------------------+----------------------+-----------+
    | maxuforce          | peak force per push  | N         |
    +--------------------+----------------------+-----------+
    | work               | work per push        | J         |
    +--------------------+----------------------+-----------+
    | cwork              | work per cycle       | J         |
    +--------------------+----------------------+-----------+
    | negwork            | negative work/cycle  | J         |
    +--------------------+----------------------+-----------+
    | meanfeff           | mean feff per push   | %         |
    +--------------------+----------------------+-----------+
    | maxfeff            | max feff per push    | %         |
    +--------------------+----------------------+-----------+
    | slope              | slope onset to peak  | Nm/s      |
    +--------------------+----------------------+-----------+
    | smoothness         | mean/peak force      |           |
    +--------------------+----------------------+-----------+
    | ptime              | push time            | s         |
    +--------------------+----------------------+-----------+
    | ctime              | cycle time           | s         |
    +--------------------+----------------------+-----------+
    | reltime            | relative push/cycle  | %         |
    +--------------------+----------------------+-----------+

    Parameters
    ----------
    data : pd.DataFrame
        measurement wheel DataFrame
    variable : str
        variable name used for peak (push) detection
    cutoff : float
        noise level for peak (push) detection
    minpeak : float
        min peak height for peak (push) detection

    Returns
    -------
    pbp : pd.DataFrame
        push-by-push DataFrame
    """
    pbp = find_peaks(data[variable], cutoff, minpeak)
    for ind, (start, stop, peak) in enumerate(zip(pbp["start"], pbp["stop"], pbp["peak"])):  # for each push
        pbp["tstart"].append(data["time"][start])
        pbp["tstop"].append(data["time"][stop])
        pbp["tpeak"].append(data["time"][peak])
        pbp["cangle"].append(data["angle"][stop] - data["angle"][start])
        pbp["cangle_deg"].append(np.rad2deg(pbp["cangle"][-1]))
        pbp["ptime"].append(pbp["tstop"][-1] - pbp["tstart"][-1])
        window = data.iloc[start:stop+1, :]
        pbp["meanpower"].append(np.mean(window["power"]))
        pbp["maxpower"].append(np.max(window["power"]))
        pbp["meantorque"].append(np.mean(window["torque"]))
        pbp["maxtorque"].append(np.max(window["torque"]))
        pbp["meanuforce"].append(np.mean(window["uforce"]))
        pbp["maxuforce"].append(np.max(window["uforce"]))
        pbp["meanforce"].append(np.mean(window["force"]))
        pbp["maxforce"].append(np.max(window["force"]))
        pbp["work"].append(np.sum(window["work"]))
        pbp["meanfeff"].append(np.mean(window["feff"]))
        pbp["maxfeff"].append(np.max(window["feff"]))
        pbp["slope"].append(pbp["maxtorque"][-1] / (pbp["tpeak"][-1] - pbp["tstart"][-1]))
        pbp["smoothness"].append(pbp["meanforce"][-1]/pbp["maxforce"][-1])

        if ind:  # only after first push
            pbp["ctime"].append(pbp["tstart"][-1] - pbp["tstart"][-2])
            pbp["reltime"].append(pbp["ptime"][-2] / pbp["ctime"][-1] * 100)

            window = data.loc[pbp["start"][ind-1]:pbp["start"][ind], "work"]  # select cycle
            pbp["cwork"].append(np.sum(window))
            window = window[window <= 0]  # only negative samples
            pbp["negwork"].append(np.sum(window))

    pbp["ctime"].append(None)  # ensure equal length of arrays
    pbp["reltime"].append(None)
    pbp["cwork"].append(None)
    pbp["negwork"].append(None)

    pbp = pd.DataFrame(pbp)

    print("\n" + "=" * 80 + f"\nFound {len(pbp)} pushes!\n" + "=" * 80 + "\n")
    return pbp


def push_by_push_ergo(data, variable="torque", cutoff=0.0, minpeak=5.0, mindist=5):
    """
    Push-by-push analysis for measurement wheel data.

    Push detection and push-by-push analysis for measurement wheel data. Returns a pandas DataFrame with:

    +--------------------+----------------------+-----------+
    | Column             | Data                 | Unit      |
    +====================+======================+===========+
    | start/stop/peak    | respective indices   |           |
    +--------------------+----------------------+-----------+
    | tstart/tstop/tpeak | respective samples   | s         |
    +--------------------+----------------------+-----------+
    | cangle             | contact angle        | rad       |
    +--------------------+----------------------+-----------+
    | cangle_deg         | contact angle        | degrees   |
    +--------------------+----------------------+-----------+
    | meanpower          | power per push       | W         |
    +--------------------+----------------------+-----------+
    | maxpower           | peak power per push  | W         |
    +--------------------+----------------------+-----------+
    | meantorque         | torque per push      | Nm        |
    +--------------------+----------------------+-----------+
    | maxtorque          | peak torque per push | Nm        |
    +--------------------+----------------------+-----------+
    | meanforce          | mean force per push  | N         |
    +--------------------+----------------------+-----------+
    | maxforce           | peak force per push  | N         |
    +--------------------+----------------------+-----------+
    | meanuforce         | (rim) force per push | N         |
    +--------------------+----------------------+-----------+
    | maxuforce          | peak force per push  | N         |
    +--------------------+----------------------+-----------+
    | work               | work per push        | J         |
    +--------------------+----------------------+-----------+
    | cwork              | work per cycle       | J         |
    +--------------------+----------------------+-----------+
    | negwork            | negative work/cycle  | J         |
    +--------------------+----------------------+-----------+
    | slope              | slope onset to peak  | Nm/s      |
    +--------------------+----------------------+-----------+
    | smoothness         | mean/peak force      |           |
    +--------------------+----------------------+-----------+
    | ptime              | push time            | s         |
    +--------------------+----------------------+-----------+
    | ctime              | cycle time           | s         |
    +--------------------+----------------------+-----------+
    | reltime            | relative push/cycle  | %         |
    +--------------------+----------------------+-----------+

    Parameters
    ----------
    data : dict
        wheelchair ergometer dictionary
    variable : str
        variable name used for peak (push) detection
    cutoff : float
        noise level for peak (push) detection
    minpeak : float
        min peak height for peak (push) detection
    mindist : int
        minimum sample distance between peak candidates, can be used to speed up algorithm

    Returns
    -------
    pbp : dict
        dictionary with left and right push-by-push DataFrame
    """
    pbp = {"left": [], "right": []}
    for side in data:
        tmp = find_peaks(data[side][variable], cutoff, minpeak, mindist)
        for ind, (start, stop, peak) in enumerate(zip(tmp["start"], tmp["stop"], tmp["peak"])):
            tmp["tstart"].append(data[side]["time"][start])
            tmp["tstop"].append(data[side]["time"][stop])
            tmp["tpeak"].append(data[side]["time"][peak])
            tmp["cangle"].append(data[side]["angle"][stop] - data[side]["angle"][start])
            tmp["cangle_deg"].append(np.rad2deg(tmp["cangle"][-1]))
            tmp["ptime"].append(tmp["tstop"][-1] - tmp["tstart"][-1])
            view = data[side].iloc[start:stop+1, :]
            tmp["meanpower"].append(np.mean(view["power"]))
            tmp["maxpower"].append(np.max(view["power"]))
            tmp["meantorque"].append(np.mean(view["torque"]))
            tmp["maxtorque"].append(np.max(view["torque"]))
            tmp["meanuforce"].append(np.mean(view["uforce"]))
            tmp["maxuforce"].append(np.max(view["uforce"]))
            tmp["meanforce"].append(np.mean(view["force"]))
            tmp["maxforce"].append(np.max(view["force"]))
            tmp["work"].append(np.sum(view["work"]))
            tmp["slope"].append(tmp["maxtorque"][-1] / (tmp["tpeak"][-1] - tmp["tstart"][-1]))
            tmp["smoothness"].append(tmp["meanforce"][-1]/tmp["maxforce"][-1])

            if ind:  # only after first push
                tmp["ctime"].append(tmp["tstart"][-1] - tmp["tstart"][-2])
                tmp["reltime"].append(tmp["ptime"][-2] / tmp["ctime"][-1] * 100)

                window = data[side].iloc[tmp["start"][ind - 1]:tmp["start"][ind], :]  # select cycle
                tmp["cwork"].append(np.sum(window["work"]))
                window = window[window <= 0]  # only negative samples
                tmp["negwork"].append(np.sum(window["work"]))

        tmp["ctime"].append(None)  # ensure equal length arrays
        tmp["reltime"].append(None)
        tmp["cwork"].append(None)
        tmp["negwork"].append(None)

        pbp[side] = pd.DataFrame(tmp)

    print("\n" + "=" * 80 + f"\nFound left: {len(pbp['left'])} and right: {len(pbp['right'])} pushes!\n"
          + "=" * 80 + "\n")
    return pbp
