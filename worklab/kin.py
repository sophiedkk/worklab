import numpy as np
import pandas as pd
from scipy.integrate import cumtrapz
from scipy.signal import savgol_filter
from .utils import lowpass_butter, find_peaks
from .move import rotate_matrix

def auto_process(data, wheelsize=0.31, rimsize=0.27, sfreq=200, co_f=15, ord_f=2, co_s=6, ord_s=2, force=True,
                 speed=True, variable="torque", cutoff=0.0, wl=201, ord_a=2, minpeak=5.0):
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
    wl : float
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


def filter_mw(data, sfreq=200., co_f=15., ord_f=2, wl=201, ord_a=2, force=True, speed=True):
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
    wl : float
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

    Returns
    -------
    pbp : pd.DataFrame
        push-by-push DataFrame
    """
    peaks = find_peaks(data[variable], cutoff, minpeak, mindist)

    keys = ["start", "stop", "peak", "tstart", "tstop", "tpeak", "cangle", "cangle_deg", "ptime", "meanpower",
            "maxpower", "meantorque", "maxtorque", "meanuforce", "maxuforce", "meanforce", "maxforce", "work",
            "meanfeff", "maxfeff", "meanftot", "maxftot", "slope", "smoothness", "ctime", "reltime", "cwork", "negwork"]
    pbp = pd.DataFrame(data=np.full((len(peaks["start"]), len(keys)), np.NaN), columns=keys)  # preallocate dataframe

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
    pbp.loc[len(pbp) - 1, ["cwork", "negwork"]] = np.NaN

    if verbose:
        print("\n" + "=" * 80 + f"\nFound {len(pbp)} pushes!\n" + "=" * 80 + "\n")
    return pbp


def push_by_push_ergo(data, variable="power", cutoff=0.0, minpeak=50.0, mindist=5, verbose=True):
    """
    Push-by-push analysis for wheelchair ergometer data.

    Push detection and push-by-push analysis for ergometer data. Returns a pandas DataFrame with:

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
        variable name used for peak (push) detection, default = power
    cutoff : float
        noise level for peak (push) detection, default = 0
    minpeak : float
        min peak height for peak (push) detection, default = 50.0
    mindist : int
        minimum sample distance between peak candidates, can be used to speed up algorithm

    Returns
    -------
    pbp : dict
        dictionary with left, right and mean push-by-push DataFrame
    """
    pbp_sides = {"left": [], "right": [], "mean": []}
    keys = ["start", "stop", "peak", "tstart", "tstop", "tpeak", "cangle", "cangle_deg", "ptime", "meanpower",
            "maxpower", "meantorque", "maxtorque", "meanuforce", "maxuforce", "meanforce", "maxforce", "work",
            "slope", "smoothness", "ctime", "reltime", "cwork", "negwork"]

    for side in data:
        if (side == 'left') | (side == 'right'):
            peaks = find_peaks(data[side][variable], cutoff, minpeak, mindist)
        if (side == 'mean'):
            peaks = find_peaks(data[side][variable], cutoff, (minpeak*2), mindist)
        pbp = pd.DataFrame(data=np.full((len(peaks["start"]), len(keys)), np.NaN), columns=keys)  # preallocate

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
        pbp.loc[len(pbp) - 1, ["cwork", "negwork"]] = np.NaN

        pbp_sides[side] = pd.DataFrame(pbp)

    if verbose:
        print("\n" + "=" * 80 + f"\nFound left: {len(pbp_sides['left'])} , right: {len(pbp_sides['right'])} and mean: {len(pbp_sides['mean'])} pushes!\n"
              + "=" * 80 + "\n")
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

    force = data[['fx', 'fy', 'fz']].T
    torques = data[['mx', 'my', 'torque']].T

    rotmat = rotate_matrix(ang/(180*np.pi), axis='x')
    data[['fx', 'fy', 'fz']] = np.dot(rotmat, force).T
    data[['mx', 'my', 'torque']] = np.dot(rotmat, torques).T

    return data
