import numpy as np
import pandas as pd
from scipy.integrate import cumulative_trapezoid
from scipy.signal import savgol_filter
from .utils import lowpass_butter, find_peaks
from .move import rotate_matrix


def auto_process(
    data,
    wheelsize=0.31,
    rimsize=0.27,
    sfreq=200,
    co_f=15,
    ord_f=2,
    co_s=6,
    ord_s=2,
    force=True,
    speed=True,
    variable="torque",
    cutoff=0.0,
    wl=201,
    ord_a=2,
    minpeak=5.0,
):
    """
    Top level processing function that performs all processing steps for mw/ergo data.

    Contains all signal processing steps in fixed order. It is advised to use this function for all (pre-)processing.
    If needed to take a look at a specific function to see how it works.

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
    wl : int
        window length angle filter
    ord_a : int
        order angle filter [..]
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
        data = filter_mw(data, sfreq, co_f, ord_f, wl, ord_a, force, speed)
        data = process_mw(data, wheelsize, rimsize, sfreq)
        pushes = push_by_push_mw(data, variable, cutoff, minpeak)
    return data, pushes


def filter_mw(data, sfreq=200.0, co_f=15.0, ord_f=2, wl=201, ord_a=2, force=True, speed=True):
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
    wl : int
        window length angle filter
    ord_a : int
        order angle filter [..]
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
        data["angle"] = savgol_filter(data["angle"], window_length=wl, polyorder=ord_a)
    return data


def filter_ergo(data, co_f=15.0, ord_f=2, co_s=6.0, ord_s=2, force=True, speed=True):
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
    | work       | instantaneous work   | J         |
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
    data["dist"] = cumulative_trapezoid(data["speed"], initial=0.0) / sfreq
    data["acc"] = np.gradient(data["speed"]) * sfreq
    data["ftot"] = (data["fx"] ** 2 + data["fy"] ** 2 + data["fz"] ** 2) ** 0.5
    data["uforce"] = data["torque"] / rimsize
    data["feff"] = (data["uforce"] / data["ftot"]) * 100
    data["force"] = data["torque"] / wheelsize
    data["power"] = data["torque"] * data["aspeed"]
    data["work"] = data["power"] / sfreq
    return data


def process_ergo(data, wheelsize=0.31, rimsize=0.275, unit="ms"):
    """
    Basic processing for ergometer data.

    Basic processing for ergometer data (e.g. speed to distance). Should be performed after filtering.
    Returned columns:

    +------------+----------------------+-----------+
    | Column     | Data                 | Unit      |
    +============+======================+===========+
    | time       | time                 | s         |
    +------------+----------------------+-----------+
    | force      | force (on wheel)     | N         |
    +------------+----------------------+-----------+
    | speed      | speed                | m/s       |
    +------------+----------------------+-----------+
    | acc        | acceleration         | m/s^2     |
    +------------+----------------------+-----------+
    | aspeed     | angular velocity     | rad/s     |
    +------------+----------------------+-----------+
    | angle      | angle                | rad       |
    +------------+----------------------+-----------+
    | dist       | cumulative distance  | m         |
    +------------+----------------------+-----------+
    | power      | power                | W         |
    +------------+----------------------+-----------+
    | torque     | torque around wheel  | Nm        |
    +------------+----------------------+-----------+
    | uforce     | effective force      | N         |
    +------------+----------------------+-----------+
    | work       | instantaneous work   | J         |
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
    unit : str
        unit of measured 'speed' column; ms, kmh or mph

    Returns
    -------
    data : dict

    See Also
    --------
    .com.load_esseda

    """
    sfreq = 100  # ergometer is always 100Hz
    for side in data:
        if unit == "kmh":
            data[side]["speed"] /= 3.6
        elif unit == "mph":
            data[side]["speed"] /= 2.23694
        elif unit == "ms":
            data[side]["speed"] = data[side]["speed"]
        else:
            raise Exception("Please specify either 'ms', 'kmh' or 'mph', data not processed!")

        data[side]["acc"] = np.gradient(data[side]["speed"]) * sfreq
        data[side]["aspeed"] = data[side]["speed"] / wheelsize
        data[side]["angle"] = cumulative_trapezoid(data[side]["aspeed"], initial=0.0) / sfreq
        data[side]["dist"] = cumulative_trapezoid(data[side]["speed"], initial=0.0) / sfreq
        data[side]["power"] = data[side]["speed"] * data[side]["force"]
        data[side]["torque"] = data[side]["force"] * wheelsize
        data[side]["uforce"] = data[side]["force"] * (wheelsize / rimsize)
        data[side]["work"] = data[side]["power"] / sfreq
    return data


def push_by_push_mw(data, variable="torque", cutoff=0.0, minpeak=5.0, mindist=5, verbose=True):
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
    | mean/maxpower      | power per push       | W         |
    +--------------------+----------------------+-----------+
    | mean/maxtorque     | torque per push      | Nm        |
    +--------------------+----------------------+-----------+
    | mean/maxforce      | force per push       | N         |
    +--------------------+----------------------+-----------+
    | mean/maxuforce     | (rim) force per push | N         |
    +--------------------+----------------------+-----------+
    | mean/maxfeff       | feffective per push  | %         |
    +--------------------+----------------------+-----------+
    | mean/maxftot       | ftotal per push      | N         |
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
    data : pd.DataFrame
        measurement wheel DataFrame
    variable : str
        variable name used for peak (push) detection
    cutoff : float
        noise level for peak (push) detection
    minpeak : float
        min peak height for peak (push) detection
    mindist : int
        minimum sample distance between peak candidates, can be used to speed up algorithm
    verbose : Boolean
        can be used to print out the number of pushes, default = True

    Returns
    -------
    pbp : pd.DataFrame
        push-by-push DataFrame
    """
    peaks = find_peaks(data[variable], cutoff, minpeak, mindist)

    keys = [
        "start",
        "stop",
        "peak",
        "tstart",
        "tstop",
        "tpeak",
        "cangle",
        "cangle_deg",
        "ptime",
        "meanpower",
        "maxpower",
        "meantorque",
        "maxtorque",
        "meanuforce",
        "maxuforce",
        "meanforce",
        "maxforce",
        "work",
        "meanfeff",
        "maxfeff",
        "meanftot",
        "maxftot",
        "slope",
        "smoothness",
        "ctime",
        "reltime",
        "cwork",
        "negwork",
    ]
    pbp = pd.DataFrame(data=np.full((len(peaks["start"]), len(keys)), np.nan), columns=keys)  # preallocate dataframe

    pbp["start"] = peaks["start"]
    pbp["peak"] = peaks["peak"]
    pbp["stop"] = peaks["stop"]
    pbp["tstart"] = data["time"][pbp["start"]].values  # get .values so indices align
    pbp["tstop"] = data["time"][pbp["stop"]].values
    pbp["tpeak"] = data["time"][pbp["peak"]].values
    pbp["ptime"] = pbp["tstop"] - pbp["tstart"].values
    pbp["ctime"] = pbp["tstart"].iloc[1:].reset_index(drop=True) - pbp["tstart"].iloc[:-1].reset_index(drop=True)
    pbp["reltime"] = (pbp["ptime"] / pbp["ctime"]) * 100
    pbp["cangle"] = data["angle"][pbp["stop"]].values - data["angle"][pbp["start"]].values
    pbp["cangle_deg"] = np.rad2deg(pbp["cangle"])

    bins = pbp[["start", "stop"]].values
    bins[:, 1] += 1
    push_bins = np.digitize(data.index, bins.ravel())  # slice dataframe from push start:stop
    push_group = data.groupby(push_bins)
    grouped = push_group.agg(["mean", "max"])[1::2].reset_index(drop=True)
    grouped.columns = [f"{col[1]}{col[0]}" for col in grouped.columns]  # collapse multiindex to match cols with pbp

    mean_max_variables = [var for var in keys if "mean" in var or "max" in var]
    pbp[mean_max_variables] = grouped[mean_max_variables]
    pbp["slope"] = pbp["maxtorque"] / (pbp["tpeak"] - pbp["tstart"])
    pbp["smoothness"] = pbp["meanforce"] / pbp["maxforce"]
    pbp["work"] = push_group["work"].sum()[1::2].reset_index(drop=True)

    cycle_bins = np.digitize(data.index, pbp["start"].values)
    pbp["cwork"] = data[["work"]].groupby(cycle_bins).sum()[1:].reset_index(drop=True)
    negative_work = data["work"].copy()
    negative_work[negative_work >= 0] = 0
    pbp["negwork"] = negative_work.groupby(cycle_bins).sum()[1:].reset_index(drop=True)
    pbp.loc[len(pbp) - 1, ["cwork", "negwork"]] = np.nan

    if verbose:
        print("\n" + "=" * 80 + f"\nFound {len(pbp)} pushes!\n" + "=" * 80 + "\n")
    return pbp


def push_by_push_ergo(data, variable="power", cutoff=0.0, minpeak=50.0, mindist=5, verbose=True):
    """
    Push-by-push analysis for wheelchair ergometer data.

    Push detection and push-by-push analysis for ergometer data. Returns a pandas DataFrame with:

    +--------------------+-----------------------+-----------+
    | Column             | Data                  | Unit      |
    +====================+=======================+===========+
    | start/stop/peak    | respective indices    |           |
    +--------------------+-----------------------+-----------+
    | tstart/tstop/tpeak | respective samples    | s         |
    +--------------------+-----------------------+-----------+
    | cangle             | contact angle         | rad       |
    +--------------------+-----------------------+-----------+
    | cangle_deg         | contact angle         | degrees   |
    +--------------------+-----------------------+-----------+
    | mean/maxpower      | power per push        | W         |
    +--------------------+-----------------------+-----------+
    | mean/maxtorque     | torque per push       | Nm        |
    +--------------------+-----------------------+-----------+
    | mean/maxforce      | force per push        | N         |
    +--------------------+-----------------------+-----------+
    | mean/maxuforce     | (rim) force per push  | N         |
    +--------------------+-----------------------+-----------+
    | mean/maxspeed      | velocity per push     | ms        |
    +--------------------+-----------------------+-----------+
    | work               | work per push         | J         |
    +--------------------+-----------------------+-----------+
    | cwork              | work per cycle        | J         |
    +--------------------+-----------------------+-----------+
    | negwork            | negative work/cycle   | J         |
    +--------------------+-----------------------+-----------+
    | slope              | slope onset to peak   | Nm/s      |
    +--------------------+-----------------------+-----------+
    | smoothness         | mean/peak force       | N         |
    +--------------------+-----------------------+-----------+
    | ptime              | push time             | s         |
    +--------------------+-----------------------+-----------+
    | ctime              | cycle time            | s         |
    +--------------------+-----------------------+-----------+
    | reltime            | relative push/cycle   | %         |
    +--------------------+-----------------------+-----------+
    | pnegpos            | neg power start push  | index    |
    +--------------------+-----------------------+-----------+
    | negpos             | neg power start push  | W        |
    +--------------------+-----------------------+-----------+
    | pnegpoe            | neg power end push    | index    |
    +--------------------+-----------------------+-----------+
    | negpoe             | neg power end push    | W        |
    +--------------------+-----------------------+-----------+

    Parameters
    ----------
    data : dict
        wheelchair ergometer dictionary  with left, right and mean DataFrame
    variable : str
        variable name used for peak (push) detection, default = power
    cutoff : float
        noise level for peak (push) detection, default = 0
    minpeak : float
        min peak height for peak (push) detection, default = 50.0
    mindist : int
        minimum sample distance between peak candidates, can be used to speed up algorithm
    verbose : Boolean
        can be used to print out the number of pushes for left, right and mean, default = True

    Returns
    -------
    pbp : dict
        dictionary with left, right and mean push-by-push DataFrame
    """
    pbp_sides = {"left": [], "right": [], "mean": []}
    keys = [
        "start",
        "stop",
        "peak",
        "tstart",
        "tstop",
        "tpeak",
        "cangle",
        "cangle_deg",
        "ptime",
        "meanpower",
        "maxpower",
        "meantorque",
        "maxtorque",
        "meanuforce",
        "maxuforce",
        "meanforce",
        "maxforce",
        "meanspeed",
        "maxspeed",
        "work",
        "slope",
        "smoothness",
        "ctime",
        "reltime",
        "cwork",
        "negwork",
        "pnegpos",
        "negpos",
        "pnegpoe",
        "negpoe",
    ]

    for side in data:
        if (side == "left") | (side == "right"):
            peaks = find_peaks(data[side][variable], cutoff, minpeak, mindist)
        else:
            peaks = find_peaks(data[side][variable], cutoff, (minpeak * 2), mindist)
        pbp = pd.DataFrame(data=np.full((len(peaks["start"]), len(keys)), np.nan), columns=keys)  # preallocate

        pbp["start"] = peaks["start"]
        pbp["peak"] = peaks["peak"]
        pbp["stop"] = peaks["stop"]
        pbp["tstart"] = data[side]["time"][pbp["start"]].values  # get .values so indices align
        pbp["tstop"] = data[side]["time"][pbp["stop"]].values
        pbp["tpeak"] = data[side]["time"][pbp["peak"]].values
        pbp["ptime"] = pbp["tstop"] - pbp["tstart"].values
        pbp["ctime"] = pbp["tstart"].iloc[1:].reset_index(drop=True) - pbp["tstart"].iloc[:-1].reset_index(drop=True)
        pbp["reltime"] = (pbp["ptime"] / pbp["ctime"]) * 100
        pbp["cangle"] = data[side]["angle"][pbp["stop"]].values - data[side]["angle"][pbp["start"]].values
        pbp["cangle_deg"] = np.rad2deg(pbp["cangle"])

        bins = pbp[["start", "stop"]].values
        bins[:, 1] += 1
        push_bins = np.digitize(data[side].index, bins.ravel())  # slice dataframe from push start:stop
        push_group = data[side].groupby(push_bins)
        grouped = push_group.agg(["mean", "max"])[1::2].reset_index(drop=True)
        grouped.columns = [f"{col[1]}{col[0]}" for col in grouped.columns]  # collapse multiindex to match cols with pbp

        mean_max_variables = [var for var in keys if "mean" in var or "max" in var]
        pbp[mean_max_variables] = grouped[mean_max_variables]
        pbp["slope"] = pbp["maxtorque"] / (pbp["tpeak"] - pbp["tstart"])
        pbp["smoothness"] = pbp["meanforce"] / pbp["maxforce"]
        pbp["work"] = push_group["work"].sum()[1::2].reset_index(drop=True)

        cycle_bins = np.digitize(data[side].index, pbp["start"].values)
        pbp["cwork"] = data[side][["work"]].groupby(cycle_bins).sum()[1:].reset_index(drop=True)
        negative_work = data[side]["work"].copy()
        negative_work[negative_work >= 0] = 0
        pbp["negwork"] = negative_work.groupby(cycle_bins).sum()[1:].reset_index(drop=True)
        pbp.loc[len(pbp) - 1, ["cwork", "negwork"]] = np.nan

        neg_before_all = []
        neg_after_all = []

        for sample, sample2 in zip(pbp['start'], pbp['stop']):
            neg_before = data['mean'].loc[sample - 10:sample]
            neg_before_min = neg_before[neg_before['power'] == min(neg_before['power'])]
            neg_before_all.append(neg_before_min)
            neg_after = data['mean'].loc[sample2:sample2 + 10]
            neg_after_min = neg_after[neg_after['power'] == min(neg_after['power'])]
            neg_after_all.append(neg_after_min)

        for push in range(0, len(pbp)):
            pbp.loc[push, "pnegpos"] = neg_before_all[push].index.item()
            pbp.loc[push, "negpos"] = neg_before_all[push]['power'].item()
            pbp.loc[push, "pnegpoe"] = neg_after_all[push].index.item()
            pbp.loc[push, "negpoe"] = neg_after_all[push]['power'].item()

        pbp_sides[side] = pd.DataFrame(pbp)

    if verbose:
        print(
            "\n" + "=" * 80 + f"\nFound left: {len(pbp_sides['left'])} , "
            f"right: {len(pbp_sides['right'])} and "
            f"mean: {len(pbp_sides['mean'])} pushes!\n" + "=" * 80 + "\n"
        )
    return pbp_sides


def camber_correct(data, ang):
    """Correct for camber angle in measurement wheel data

    Parameters
    ----------
    data : pd.DataFrame()
        measurement wheel data with forces/torques in 3D

    ang : int
        camber angle to correct

    Returns
    -------
    data: pd.DataFrame()
        measurement wheel data with forces/torques in 3D
        corrected for camber angle
    """

    force = data[["fx", "fy", "fz"]].T
    torques = data[["mx", "my", "torque"]].T

    rotmat = rotate_matrix(ang / (180 * np.pi), axis="x")
    data[["fx", "fy", "fz"]] = np.dot(rotmat, force).T
    data[["mx", "my", "torque"]] = np.dot(rotmat, torques).T

    return data
