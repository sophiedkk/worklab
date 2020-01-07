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
            max_time = sessiondata[device][sensor]["Time"].max()
            end_time = max_time if max_time > end_time else end_time

    new_time = np.arange(0, end_time, 1 / sfreq)

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


def calc_wheelspeed(sessiondata, camber=15, wsize=0.31, wbase=0.60, inplace=False):
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
    frame = sessiondata["Frame"] = sessiondata["Frame"]["sensors"]  # view into dataframe, ditch sensors
    left = sessiondata["Left"] = sessiondata["Left"]["sensors"]
    right = sessiondata["Right"] = sessiondata["Right"]["sensors"]

    sfreq = 1 / frame["Time"].diff().mean()

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
    frame["CombDist"] = (right["GyroDist"] + left["GyroDist"]) / 2  # mean distance

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
    comb_ratio = lowpass_butter(comb_ratio, sfreq=sfreq, cutoff=20.)  # Filter the signal
    comb_ratio = np.clip(comb_ratio, 0, 1)  # clamp Combratio values, not in df
    frame["CombSkidVel"] = (frame["CombVelRight"] * comb_ratio) + (frame["CombVelLeft"] * (1-comb_ratio))
    frame["CombSkidDist"] = cumtrapz(frame["CombSkidVel"], initial=0.0) / sfreq  # Combined skid displacement
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

    order = {"GyroscopeX": "GyroscopeZ", "GyroscopeZ": "GyroscopeY", "GyroscopeY": "GyroscopeX"}
    sessiondata["Left"]["sensors"].rename(columns=order, inplace=True)
    sessiondata["Right"]["sensors"].rename(columns=order, inplace=True)
    sessiondata["Right"]["sensors"]["GyroscopeY"] *= -1
    return sessiondata


def push_detection(acceleration: np.array, sfreq=400.):
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
    frame_acceleration_p = lowpass_butter(acceleration, sfreq=sfreq, cutoff=cutoff_freq)
    std_fr_acc = np.std(frame_acceleration_p)
    push_acc_fr, push_acc_fr_ind = find_peaks(frame_acceleration_p, height=std_fr_acc/2,
                                              distance=round(1 / (max_freq*1.5) * sfreq), prominence=std_fr_acc / 2)
    n_pushes = len(push_acc_fr)
    push_freq = n_pushes/(len(acceleration) / sfreq)
    cycle_time = pd.DataFrame([])
    
    for n in range(0, len(push_acc_fr) - 1):
        cycle_time = cycle_time.append([(push_acc_fr[n + 1] / sfreq) - (push_acc_fr[n] / sfreq)])

    return push_acc_fr, frame_acceleration_p, n_pushes, cycle_time, push_freq


def get_perp_vector(vector2d: np.array, clockwise=True) -> np.array:
    """
    Get the vector perpendicular to the input vector. Only works in 2D as 3D has infinite solutions.

    Parameters
    ----------
    vector2d : np.array
        [n, 3] vector data
    clockwise : bool
        clockwise or counterclockwise rotation

    Returns
    -------
    perp_vector2d : np.array
        rotated vector

    """
    if clockwise:
        """Gets 2D vector perpendicular to input vector, rotated clockwise"""
        perp_vector2d = np.empty(vector2d.shape)
        perp_vector2d[:, 0] = vector2d[:, 1] * -1
        perp_vector2d[:, 1] = vector2d[:, 0]
        perp_vector2d[:, 2] = vector2d[:, 2]
    else:
        """Gets 2D vector perpendicular to input vector, rotated counterclockwise"""
        perp_vector2d = np.empty(vector2d.shape)
        perp_vector2d[:, 0] = vector2d[:, 1]
        perp_vector2d[:, 1] = vector2d[:, 0] * -1
        perp_vector2d[:, 2] = vector2d[:, 2]
    return perp_vector2d


def get_rotation_matrix(new_frame, local_to_world=True):
    """
    Get the rotation matrix between a new reference frame and the global reference frame or the other way around.

    Parameters
    ----------
    new_frame : np.array
        3x3 array specifying the new reference frame
    local_to_world : bool
        global to local or local to global

    Returns
    -------
    rotation_matrix : np.array
        rotation matrix that can be used to rotate marker data, e.g.: rotation_matrix @ marker

    """
    x1 = np.array([1, 0, 0])  # X axis of the world
    x2 = np.array([0, 1, 0])  # Y axis of the world
    x3 = np.array([0, 0, 1])  # Z axis of the world

    x1_prime = new_frame[:, 0]  # X axis of the local frame
    x2_prime = new_frame[:, 1]  # Y axis of the local frame
    x3_prime = new_frame[:, 2]  # Z axis of the local frame

    if local_to_world:
        rotation_matrix = np.array([[np.dot(x1, x1_prime), np.dot(x1, x2_prime), np.dot(x1, x3_prime)],
                                    [np.dot(x2, x1_prime), np.dot(x2, x2_prime), np.dot(x2, x3_prime)],
                                    [np.dot(x3, x1_prime), np.dot(x3, x2_prime), np.dot(x3, x3_prime)]])
    else:
        rotation_matrix = np.array([[np.dot(x1_prime, x1), np.dot(x1_prime, x2), np.dot(x1_prime, x3)],
                                    [np.dot(x2_prime, x1), np.dot(x2_prime, x2), np.dot(x2_prime, x3)],
                                    [np.dot(x3_prime, x1), np.dot(x3_prime, x2), np.dot(x3_prime, x3)]])
    return rotation_matrix


def normalize(x):
    """
    Normalizes [n, 3] marker data using an l2 norm.

    Parameters
    ----------
    x : np.array
        marker data to be normalized

    Returns
    -------
    x : np.array
        normalized marker data

    """
    if isinstance(x, pd.DataFrame):
        return x.div(np.linalg.norm(x), axis=0)
    else:
        return x / np.linalg.norm(x)


def calc_marker_angles(v_1: np.array, v_2: np.array, deg=False):
    """
    Calculates n angles between two [n, 3] markers.

    Parameters
    ----------
    v_1 : np.array
        [n, 3] array or DataFrame for marker 1
    v_2 : np.array
        [n, 3] array or DataFrame for marker 2
    deg : bool
        return radians or degrees, default is radians

    Returns
    -------
    x : np.array
        returns [n, 1] array with the angle for each sample

    """
    p1 = np.einsum('ij,ij->i', v_1, v_2)
    p2 = np.cross(v_1, v_2, axis=1)
    p3 = np.linalg.norm(p2, axis=1)
    angles = np.arctan2(p3, p1)
    return np.rad2deg(angles) if deg else angles
