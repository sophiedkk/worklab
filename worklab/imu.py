import copy
from warnings import warn

import numpy as np
import pandas as pd
from scipy.integrate import cumtrapz
from scipy.signal import periodogram, find_peaks

from .utils import lowpass_butter, pd_interp


def resample_imu(sessiondata, sfreq=400.):
    """
    Resample all devices and sensors to new sample frequency.

    Resamples all devices and sensors to new sample frequency. Sample intervals are not fixed with NGIMU's so resampling
    before further analysis is recommended. Translated from xio-Technologies [2]_.

    Parameters
    ----------
    sessiondata : dict
        original session data structure to be resampled
    sfreq : float
        new intended sample frequency

    Returns
    -------
    sessiondata : dict
        resampled session data structure

    References
    ----------
    .. [2] https://github.com/xioTechnologies/NGIMU-MATLAB-Import-Logged-Data-Example

    """
    end_time = 0
    for device in sessiondata:
        for sensor in sessiondata[device]:
            max_time = sessiondata[device][sensor]["time"].max()
            end_time = max_time if max_time > end_time else end_time

    new_time = np.arange(0, end_time, 1 / sfreq)

    for device in sessiondata:
        for sensor in sessiondata[device]:
            if sensor == "quaternion":  # TODO: xio-tech has TODO here to replace this part with slerp
                sessiondata[device][sensor] = pd_interp(sessiondata[device][sensor], "time", new_time)
                sessiondata[device][sensor] *= (1 / np.linalg.norm(sessiondata[device][sensor], axis=0))
            elif sensor == "matrix":
                sessiondata[device].pop(sensor)
                warn("Rotation matrix cannot be resampled. This dataframe has been removed")
            else:
                sessiondata[device][sensor] = pd_interp(sessiondata[device][sensor], "time", new_time)
    return sessiondata


def process_imu(sessiondata, camber=15, wsize=0.31, wbase=0.60, inplace=False):
    """
    Calculate wheelchair velocity based on NGIMU data with skid correction.

    Parameters
    ----------
    sessiondata : dict
        original sessiondata structure
    camber : float
        camber angle in degrees
    wsize : float
        radius of the wheels
    wbase : float
        width of wheelbase
    inplace : bool
        performs operation inplace

    Returns
    -------
    sessiondata : dict
        sessiondata structure with processed data

    """
    if not inplace:
        sessiondata = copy.deepcopy(sessiondata)
    frame = sessiondata["frame"] = sessiondata["frame"]["sensors"]  # view into DataFrame, ditch sensors
    left = sessiondata["left"] = sessiondata["left"]["sensors"]
    right = sessiondata["right"] = sessiondata["right"]["sensors"]

    sfreq = 1 / frame["time"].diff().mean()

    # Wheelchair camber correction
    deg2rad = np.pi / 180
    right["gyro_cor"] = right["gyroscope_y"] + np.tan(camber * deg2rad) * (
            frame["gyroscope_z"] * np.cos(camber * deg2rad))
    left["gyro_cor"] = left["gyroscope_y"] - np.tan(camber * deg2rad) * (
            frame["gyroscope_z"] * np.cos(camber * deg2rad))
    frame["gyro_cor"] = (right["gyro_cor"] + left["gyro_cor"]) / 2

    # Calculation of rotations, rotational velocity and acceleration
    frame["rot_vel"] = lowpass_butter(frame["gyroscope_z"], sfreq=sfreq, cutoff=20)
    frame["rot"] = cumtrapz(abs(frame["rot_vel"]) / sfreq, initial=0.0)
    frame["rot_acc"] = lowpass_butter(np.gradient(frame["rot_vel"]) * sfreq, sfreq=sfreq, cutoff=20)

    # Calculation of speed, acceleration and distance
    right["vel"] = right["gyro_cor"] * wsize * deg2rad  # angular velocity to linear velocity
    right["dist"] = cumtrapz(right["vel"] / sfreq, initial=0.0)  # integral of velocity gives distance

    left["vel"] = left["gyro_cor"] * wsize * deg2rad
    left["dist"] = cumtrapz(left["vel"] / sfreq, initial=0.0)

    frame["vel"] = (right["vel"] + left["vel"]) / 2  # mean velocity both sides
    frame["vel"] = lowpass_butter(frame["vel"], sfreq=sfreq, cutoff=20)
    frame["acc"] = lowpass_butter(np.gradient(frame["vel"])*sfreq, sfreq=sfreq, cutoff=20) #mean acceleration from velocity
    frame["dist"] = (right["dist"] + left["dist"]) / 2  # mean distance
    frame["accelerometer_x"] = frame["accelerometer_x"]*9.81

    # distance in the x and y direction
    frame["dist_y"] = cumtrapz(
        np.gradient(frame["dist"]) * np.sin(np.deg2rad(cumtrapz(frame["rot_vel"] / sfreq, initial=0.0))),
        initial=0.0)
    frame["dist_x"] = cumtrapz(
        np.gradient(frame["dist"]) * np.cos(np.deg2rad(cumtrapz(frame["rot_vel"] / sfreq, initial=0.0))),
        initial=0.0)

    """Perform skid correction from Rienk vd Slikke, please refer and reference to: Van der Slikke, R. M. A., et. al. 
    Wheel skid correction is a prerequisite to reliably measure wheelchair sports kinematics based on inertial sensors. 
    Procedia Engineering, 112, 207-212."""
    frame["skid_vel_right"] = right["vel"]  # Calculate frame centre distance
    frame["skid_vel_right"] -= np.tan(np.deg2rad(frame["gyroscope_z"] / sfreq)) * wbase / 2 * sfreq
    frame["skid_vel_left"] = left["vel"]
    frame["skid_vel_left"] += np.tan(np.deg2rad(frame["gyroscope_z"] / sfreq)) * wbase / 2 * sfreq

    r_ratio0 = np.abs(right["vel"]) / (np.abs(right["vel"]) + np.abs(left["vel"]))  # Ratio left and right
    l_ratio0 = np.abs(left["vel"]) / (np.abs(right["vel"]) + np.abs(left["vel"]))
    r_ratio1 = np.abs(np.gradient(left["vel"])) / (np.abs(np.gradient(right["vel"]))
                                                   + np.abs(np.gradient(left["vel"])))
    l_ratio1 = np.abs(np.gradient(right["vel"])) / (np.abs(np.gradient(right["vel"]))
                                                    + np.abs(np.gradient(left["vel"])))

    comb_ratio = (r_ratio0 * r_ratio1) / ((r_ratio0 * r_ratio1) + (l_ratio0 * l_ratio1))  # Combine speed ratios
    comb_ratio.fillna(value=0., inplace=True)
    comb_ratio = lowpass_butter(comb_ratio, sfreq=sfreq, cutoff=20)  # Filter the signal
    comb_ratio = np.clip(comb_ratio, 0, 1)  # clamp Combratio values, not in df
    frame["skid_vel"] = (frame["skid_vel_right"] * comb_ratio) + (frame["skid_vel_left"] * (1 - comb_ratio))
    frame["skid_vel"] = lowpass_butter(frame["skid_vel"], sfreq=sfreq, cutoff=20)
    frame["skid_dist"] = cumtrapz(frame["skid_vel"], initial=0.0) / sfreq  # Combined skid distance
    return sessiondata


def change_imu_orientation(sessiondata, inplace=False):
    """
    Changes IMU orientation from in-wheel to on-wheel

    Parameters
    ----------
    sessiondata : dict
        original sessiondata structure
    inplace : bool
        perform operation inplace

    Returns
    -------
    sessiondata : dict
        sessiondata with reoriented gyroscope data

    """
    if not inplace:
        sessiondata = copy.deepcopy(sessiondata)

    order = {"gyroscope_x": "gyroscope_z", "gyroscope_z": "gyroscope_y", "gyroscope_y": "gyroscope_x"}
    sessiondata["left"]["sensors"].rename(columns=order, inplace=True)
    sessiondata["right"]["sensors"].rename(columns=order, inplace=True)
    sessiondata["right"]["sensors"]["gyroscope_y"] *= -1
    return sessiondata


def push_imu(acceleration: np.array, sfreq=400.):
    """
    Push detection based on velocity signal of IMU on a wheelchair [3]_.

    Parameters
    ----------
    acceleration : np.array
    sfreq : float

    Returns
    -------
        push_acc_fr, frame_acceleration_p, n_pushes, cycle_time, push_freq

    References
    ----------
    .. [3] van der Slikke, R., Berger, M., Bregman, D., & Veeger, D. (2016). Push characteristics in wheelchair court sport sprinting. Procedia engineering, 147, 730-734.

    """
    min_freq = 1.2
    f, pxx = periodogram(acceleration - np.mean(acceleration), sfreq)
    min_freq_f = len(f[f < min_freq])
    max_freq_ind_temp = np.argmax(pxx[min_freq_f:min_freq_f * 5])
    max_freq = f[min_freq_f + max_freq_ind_temp]
    max_freq = min(max_freq, 3.)
    cutoff_freq = 1.5 * max_freq
    acc_filt = lowpass_butter(acceleration, sfreq=sfreq, cutoff=cutoff_freq)
    std_acc = np.std(acc_filt)
    push_idx, peak_char = find_peaks(acc_filt, height=std_acc / 2,
                                              distance=round(1 / (max_freq * 1.5) * sfreq), prominence=std_acc / 2)
    n_pushes = len(push_idx)
    push_freq = n_pushes / (len(acceleration) / sfreq)
    cycle_time = pd.DataFrame([])

    for n in range(0, len(push_idx) - 1):
        cycle_time = cycle_time.append([(push_idx[n + 1] / sfreq) - (push_idx[n] / sfreq)])

    return push_idx, acc_filt, n_pushes, cycle_time, push_freq


def vel_zones(velocity, time):
    """
    Calculate wheelchair velocity zones

    Parameters
    ----------
    velocity : dict
        velocity data structure
    time : dict
        time data structure

    Returns
    -------
    velocity_zones : dict
        velocity zones (m/s), 1-2, 2-3, 3-4, 4-5, 5 and above

    """

    vel_1 = velocity[velocity > 1]
    vel_2 = velocity[velocity > 2]
    vel_3 = velocity[velocity > 3]
    vel_4 = velocity[velocity > 4]
    vel_5 = velocity[velocity > 5]
    per_bet_1_and_2 = abs((len(vel_2) - len(vel_1)) / len(time) * 100)
    per_bet_2_and_3 = abs((len(vel_3) - len(vel_2)) / len(time) * 100)
    per_bet_3_and_4 = abs((len(vel_4) - len(vel_3)) / len(time) * 100)
    per_bet_4_and_5 = abs((len(vel_5) - len(vel_4)) / len(time) * 100)
    per_above_5 = abs(len(vel_5) / len(time) * 100)

    zones = [per_bet_1_and_2, per_bet_2_and_3, per_bet_3_and_4,
             per_bet_4_and_5, per_above_5]
    return zones


def butterfly(sessiondata, sfreq: float = 400., skid=False) -> dict:
    """
    Calculate butterfly sprint test outcome measures.

    Parameters
    ----------
    sessiondata : dict
        processed sessiondata structure

    Returns
    -------
    sessiondata : dict
        sessiondata structure with butterfly sprint test data
    outcomes_bs : dict
        structure with most important outcome variables of the butterfly sprint test

    """
    if skid == True:
        vel = "skid_vel"
    else:
        vel = "vel"
    sessiondata["frame"][vel] = lowpass_butter(sessiondata["frame"][vel], sfreq=sfreq, cutoff=10)

    m = int(len(sessiondata["frame"][vel]) - (0.5 * sfreq))
    st = 1
    for st in range(1, m):
        if sessiondata["frame"][vel][st] > 0.1:
            if sessiondata["frame"][vel][int(st + (0.5 * sfreq))] > 1.0:
                start_value = st
                break
    sessiondata["frame"] = sessiondata["frame"][start_value:].reset_index(drop=True)
    sessiondata["frame"]["dist"] = cumtrapz(sessiondata["frame"][vel], initial=0.0) / sfreq

    sessiondata["frame"]["dist_y"] = cumtrapz(np.gradient(sessiondata["frame"]["dist"]) * np.sin(
        np.deg2rad(cumtrapz(sessiondata["frame"]["rot_vel"] / sfreq, initial=0.0))), initial=0.0)
    sessiondata["frame"]["dist_x"] = cumtrapz(np.gradient(sessiondata["frame"]["dist"]) * np.cos(
        np.deg2rad(cumtrapz(sessiondata["frame"]["rot_vel"] / sfreq, initial=0.0))), initial=0.0)

    end_point = int(pd.DataFrame(sessiondata["frame"]["dist_x"]).idxmin())

    dist_x_zero = sessiondata["frame"]["dist_x"][end_point:]
    find_end = dist_x_zero[dist_x_zero > 0]
    end_value = end_point + (len(dist_x_zero) - len(find_end))

    sessiondata["frame"] = sessiondata["frame"][:end_value]
    sessiondata["frame"]["time"] -= sessiondata['frame']['time'][0]
    sessiondata["frame"]["rot_vel_left"] = sessiondata["frame"]["rot_vel"][sessiondata["frame"]["rot_vel"] > 30]
    sessiondata["frame"]["rot_vel_right"] = sessiondata["frame"]["rot_vel"][sessiondata["frame"]["rot_vel"] < -30]

    outcomes_bs = pd.DataFrame([])
    outcomes_bs = outcomes_bs.append([{'endtime': (end_value / sfreq),
                                       'vel_mean': np.mean(sessiondata["frame"][vel]),
                                       'vel_peak': np.max(sessiondata["frame"][vel]),
                                       'acc_peak': np.max(sessiondata["frame"]["acc"]),
                                       'rot_vel_mean_right': np.mean(sessiondata["frame"]["rot_vel_right"]),
                                       'rot_vel_mean_left': np.mean(sessiondata["frame"]["rot_vel_left"]),
                                       'rot_vel_peak_right': np.min(sessiondata["frame"]["rot_vel_right"]),
                                       'rot_vel_peak_left': np.max(sessiondata["frame"]["rot_vel_left"]),
                                       'rot_acc_peak': np.max(sessiondata["frame"]["rot_acc"])}], ignore_index=True)
    outcomes_bs = round(outcomes_bs, 2)
    return sessiondata, outcomes_bs


def sprint_10m(sessiondata, sfreq: float = 400., skid=False) -> dict:
    """
    Calculate 10m sprint test outcomes measures.

    Parameters
    ----------
    sessiondata : dict
        processed sessiondata structure

    Returns
    -------
    sessiondata : dict
        sessiondata structure with 10m sprint test data
    outcomes_sprint : dict
        structure with most important outcome variables of the sprint test

    """
    if skid == True:
        vel = "skid_vel"
    else:
        vel = "vel"

    sessiondata["frame"][vel] = lowpass_butter(sessiondata["frame"][vel], sfreq=sfreq, cutoff=10)

    m = int(len(sessiondata["frame"][vel]) - (0.5 * sfreq))
    for st in range(1, m):
        if sessiondata["frame"][vel][st] > 0.1:
            if sessiondata["frame"][vel][int(st + (0.5 * sfreq))] > 1.0:
                start_value = st
                break
    sessiondata["frame"] = sessiondata["frame"][start_value:].reset_index(drop=True)
    sessiondata["frame"]["dist"] = cumtrapz(sessiondata["frame"][vel], initial=0.0) / sfreq

    sessiondata["frame"]["dist_y"] = cumtrapz(np.gradient(sessiondata["frame"]["dist"]) * np.sin(
        np.deg2rad(cumtrapz(sessiondata["frame"]["rot_vel"] / sfreq, initial=0.0))), initial=0.0)
    sessiondata["frame"]["dist_x"] = cumtrapz(np.gradient(sessiondata["frame"]["dist"]) * np.cos(
        np.deg2rad(cumtrapz(sessiondata["frame"]["rot_vel"] / sfreq, initial=0.0))), initial=0.0)

    n10 = int(len(sessiondata["frame"]["dist_x"]))
    for val2 in range(0, n10):
        if sessiondata["frame"]["dist_x"][val2] > 2:
            two_value = val2
            break
    for val5 in range(0, n10):
        if sessiondata["frame"]["dist_x"][val5] > 5:
            five_value = val5
            break
    for val10 in range(0, n10):
        if sessiondata["frame"]["dist_x"][val10] > 10:
            end_value = val10
            break

    sessiondata["frame"] = sessiondata["frame"][:end_value]
    sessiondata["frame"]["time"] -= sessiondata['frame']['time'][0]
    push_ind, frame_acceleration_p, n_pushes, cycle_time, push_freq = push_imu(
        sessiondata["frame"]["accelerometer_x"] * 9.81, sfreq=400)

    outcomes_sprint = pd.DataFrame([])
    outcomes_sprint = outcomes_sprint.append([{'time_2m': two_value / sfreq,
                                               'time_5m': five_value / sfreq,
                                               'time_10m': end_value / sfreq,
                                               'vel_2m_peak': np.max(sessiondata["frame"][vel][:two_value]),
                                               'vel_5m_peak': np.max(
                                                   sessiondata["frame"][vel][two_value:five_value]),
                                               'vel_10m_peak': np.max(
                                                   sessiondata["frame"][vel][five_value:end_value]),
                                               'pos_vel_peak': sessiondata["frame"]["dist_x"][
                                                   sessiondata["frame"][vel].idxmax()],
                                               'vel_mean': np.mean(sessiondata["frame"][vel]),
                                               'vel_peak': np.max(sessiondata["frame"][vel]),
                                               'acc_2m_peak': np.max(sessiondata["frame"]["acc"][:two_value]),
                                               'acc_5m_peak': np.max(sessiondata["frame"]["acc"][two_value:five_value]),
                                               'acc_10m_peak': np.max(
                                                   sessiondata["frame"]["acc"][five_value:end_value]),
                                               'acc_peak': np.max(sessiondata["frame"]["acc"]),
                                               'pos_acc_peak': sessiondata["frame"]["dist_x"][
                                                   sessiondata["frame"]["acc"].idxmax()],
                                               'n_pushes': n_pushes,
                                               'dist_push1': sessiondata["frame"]["dist_x"][push_ind[0]],
                                               'dist_push2': sessiondata["frame"]["dist_x"][push_ind[1]],
                                               'dist_push3': sessiondata["frame"]["dist_x"][push_ind[2]],
                                               'cycle_time': np.mean(cycle_time[0])
                                               }], ignore_index=True)
    outcomes_sprint = round(outcomes_sprint, 2)
    return sessiondata, outcomes_sprint


def sprint_20m(sessiondata, sfreq: float = 400., skid=False) -> dict:
    """
    Calculate 20m sprint outcomes measures.

    Parameters
    ----------
    sessiondata : dict
        processed sessiondata structure

    Returns
    -------
    sessiondata : dict
        sessiondata structure with 20m sprint data
    outcomes_sprint : dict
        structure with most important outcome variables of the sprint test

    """
    if skid == True:
        vel = "skid_vel"
    else:
        vel = "vel"

    sessiondata["frame"][vel] = lowpass_butter(sessiondata["frame"][vel], sfreq=sfreq, cutoff=10)

    m = int(len(sessiondata["frame"][vel]) - (0.5 * sfreq))
    st = 1
    for st in range(1, m):
        if sessiondata["frame"][vel][st] > 0.1:
            if sessiondata["frame"][vel][int(st + (0.5 * sfreq))] > 1.0:
                start_value = st
                break
    sessiondata["frame"] = sessiondata["frame"][start_value:].reset_index(drop=True)
    sessiondata["frame"]["dist"] = cumtrapz(sessiondata["frame"][vel], initial=0.0) / sfreq

    sessiondata["frame"]["dist_y"] = cumtrapz(np.gradient(sessiondata["frame"]["dist"]) * np.sin(
        np.deg2rad(cumtrapz(sessiondata["frame"]["rot_vel"] / sfreq, initial=0.0))), initial=0.0)
    sessiondata["frame"]["dist_x"] = cumtrapz(np.gradient(sessiondata["frame"]["dist"]) * np.cos(
        np.deg2rad(cumtrapz(sessiondata["frame"]["rot_vel"] / sfreq, initial=0.0))), initial=0.0)

    n20 = int(len(sessiondata["frame"]["dist_x"]))
    for val5 in range(0, n20):
        if sessiondata["frame"]["dist_x"][val5] > 5:
            five_value = val5
            break
    for val10 in range(0, n20):
        if sessiondata["frame"]["dist_x"][val10] > 10:
            ten_value = val10
            break
    for val20 in range(0, n20):
        if sessiondata["frame"]["dist_x"][val20] > 20:
            end_value = val20
            break

    sessiondata["frame"] = sessiondata["frame"][:end_value]
    sessiondata["frame"]["time"] -= sessiondata['frame']['time'][0]
    push_ind, frame_acceleration_p, n_pushes, cycle_time, push_freq = push_imu(
        sessiondata["frame"]["accelerometer_x"] * 9.81, sfreq=400)

    outcomes_sprint = pd.DataFrame([])
    outcomes_sprint = outcomes_sprint.append([{'time_2m': five_value / sfreq,
                                               'time_10m': ten_value / sfreq,
                                               'time_20m': end_value / sfreq,
                                               'vel_5m_peak': np.max(sessiondata["frame"][vel][:five_value]),
                                               'vel_10m_peak': np.max(
                                                   sessiondata["frame"][vel][five_value:ten_value]),
                                               'vel_20m_peak': np.max(
                                                   sessiondata["frame"][vel][ten_value:end_value]),
                                               'pos_vel_peak': sessiondata["frame"]["dist_x"][
                                                   sessiondata["frame"][vel].idxmax()],
                                               'vel_mean': np.mean(sessiondata["frame"][vel]),
                                               'vel_peak': np.max(sessiondata["frame"][vel]),
                                               'acc_5m_peak': np.max(sessiondata["frame"]["acc"][:five_value]),
                                               'acc_10m_peak': np.max(
                                                   sessiondata["frame"]["acc"][five_value:ten_value]),
                                               'acc_20m_peak': np.max(sessiondata["frame"]["acc"][ten_value:end_value]),
                                               'acc_peak': np.max(sessiondata["frame"]["acc"]),
                                               'pos_acc_peak': sessiondata["frame"]["dist_x"][
                                                                sessiondata["frame"]["acc"].idxmax()],
                                               'n_pushes': n_pushes,
                                               'dist_push1': sessiondata["frame"]["dist_x"][push_ind[0]],
                                               'dist_push2': sessiondata["frame"]["dist_x"][push_ind[1]],
                                               'dist_push3': sessiondata["frame"]["dist_x"][push_ind[2]],
                                               'cycle_time': np.mean(cycle_time[0])
                                               }], ignore_index=True)
    outcomes_sprint = round(outcomes_sprint, 2)
    return sessiondata, outcomes_sprint


def spider(sessiondata, sfreq: float = 400., skid=False) -> dict:
    """
    Calculate spider outcomes measures.

    Parameters
    ----------
    sessiondata : dict
        processed sessiondata structure

    Returns
    -------
    sessiondata : dict
        sessiondata structure with spider data
    outcomes_spider : dict
        structure with most important outcome variables of the spider test

    """
    if skid == True:
        vel = "skid_vel"
    else:
        vel = "vel"

    sessiondata["frame"][vel] = lowpass_butter(sessiondata["frame"][vel], sfreq=sfreq, cutoff=10)

    m = int(len(sessiondata["frame"][vel]) - (0.5 * sfreq))
    st = 1
    for st in range(1, m):
        if sessiondata["frame"][vel][st] > 0.1:
            if sessiondata["frame"][vel][int(st + (0.5 * sfreq))] > 1.0:
                start_value = st
                break
    sessiondata["frame"] = sessiondata["frame"][start_value:].reset_index(drop=True)
    sessiondata["frame"]["dist"] = cumtrapz(sessiondata["frame"][vel], initial=0.0) / sfreq

    sessiondata["frame"]["dist_y"] = cumtrapz(np.gradient(sessiondata["frame"]["dist"]) * np.sin(
        np.deg2rad(cumtrapz(sessiondata["frame"]["rot_vel"] / sfreq, initial=0.0))), initial=0.0)
    sessiondata["frame"]["dist_x"] = cumtrapz(np.gradient(sessiondata["frame"]["dist"]) * np.cos(
        np.deg2rad(cumtrapz(sessiondata["frame"]["rot_vel"] / sfreq, initial=0.0))), initial=0.0)

    end_point = int(pd.DataFrame(sessiondata["frame"]["dist_x"]).idxmax())

    dist_x_zero = sessiondata["frame"]["dist_x"][end_point:]
    find_end = dist_x_zero[dist_x_zero < 0]
    end_value = end_point + (len(dist_x_zero) - len(find_end))

    sessiondata["frame"] = sessiondata["frame"][:end_value]
    sessiondata["frame"]["time"] -= sessiondata['frame']['time'][0]
    sessiondata["frame"]["rot_vel_left"] = sessiondata["frame"]["rot_vel"][sessiondata["frame"]["rot_vel"] > 30]
    sessiondata["frame"]["rot_vel_right"] = sessiondata["frame"]["rot_vel"][sessiondata["frame"]["rot_vel"] < -30]
    sessiondata["frame"]["dist_x"] = -sessiondata["frame"]["dist_x"]

    outcomes_spider = pd.DataFrame([])
    outcomes_spider = outcomes_spider.append([{'endtime': (end_value / sfreq),
                                               'vel_mean': np.mean(sessiondata["frame"][vel]),
                                               'vel_peak': np.max(sessiondata["frame"][vel]),
                                               'acc_peak': np.max(sessiondata["frame"]["acc"]),
                                               'rot_vel_mean_right': np.mean(
                                                   sessiondata["frame"]["rot_vel_right"]),
                                               'rot_vel_mean_left': np.mean(sessiondata["frame"]["rot_vel_left"]),
                                               'rot_vel_peak_right': np.min(
                                                   sessiondata["frame"]["rot_vel_right"]),
                                               'rot_vel_peak_left': np.max(sessiondata["frame"]["rot_vel_left"]),
                                               'rot_acc_peak': np.max(sessiondata["frame"]["rot_acc"])}],
                                             ignore_index=True)
    outcomes_spider = round(outcomes_spider, 2)
    return sessiondata, outcomes_spider
