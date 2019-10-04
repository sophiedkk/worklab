"""
-Move(kinematics) module-
Description: Basic functions for movement related data such as from IMUs or optical tracking systems. IMU functions are
specifically made for the NGIMUs we use in the worklab.
Author:     Rick de Klerk
Contact:    r.de.klerk@umcg.nl
Company:    University Medical Center Groningen
License:    GNU GPLv3.0
Date:       27/06/2019
"""
import copy
from warnings import warn

import numpy as np
import pandas as pd
from scipy.integrate import cumtrapz
from scipy.signal import periodogram, find_peaks

from .utils import lowpass_butter, pd_interp


def resample_imu(sessiondata, sfreq: float = 400.) -> dict:
    """Resample all devices and sensors to new sample frequency. Translated from xio-Technologies.
    :param sessiondata: original sessiondata structure
    :param sfreq: new intended sample frequency
    :return: resampled sessiondata
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


def calc_wheelspeed(sessiondata, camber=15, wsize=0.31, wbase=0.60, inplace: bool = False) -> dict:
    """Calculate wheelchair velocity based on NGIMU data.
    :param sessiondata: original sessiondata structure
    :param camber: camber angle in degrees
    :param wsize: radius of the wheels
    :param wbase: width of wheelbase
    :param inplace: performs operation inplace
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
    comb_ratio = lowpass_butter(comb_ratio, sfreq=sfreq, cutoff=20)  # Filter the signal
    comb_ratio = np.clip(comb_ratio, 0, 1)  # clamp Combratio values, not in df
    frame["CombSkidVel"] = (frame["CombVelRight"] * comb_ratio) + (frame["CombVelLeft"] * (1-comb_ratio))
    frame["CombSkidDist"] = cumtrapz(frame["CombSkidVel"], initial=0.0) / sfreq  # Combined skid displacement
    return sessiondata


def change_imu_orientation(sessiondata: dict, inplace: bool = False) -> dict:
    """Changes IMU orientation from in-wheel to on-wheel

    :param sessiondata: original sessiondata structure
    :param inplace: perform operation inplace
    :return: sessiondata with reoriented gyroscope data
    """
    if not inplace:
        sessiondata = copy.deepcopy(sessiondata)

    order = {"GyroscopeX": "GyroscopeZ", "GyroscopeZ": "GyroscopeY", "GyroscopeY": "GyroscopeX"}
    sessiondata["Left"]["sensors"].rename(columns=order, inplace=True)
    sessiondata["Right"]["sensors"].rename(columns=order, inplace=True)
    sessiondata["Right"]["sensors"]["GyroscopeY"] *= -1
    return sessiondata


def push_detection(acceleration: np.array, fs: int = 400):
    """
    Push detection based on velocity signal of IMU on a wheelchair.
    Adapted from: van der Slikke, R., Berger, M., Bregman, D., & Veeger, D. (2016). Push characteristics in wheelchair
    court sport sprinting. Procedia engineering, 147, 730-734.

    :param acceleration:
    :param fs:
    :return: push location index,
    """
    min_freq = 1.2
    f, pxx = periodogram(acceleration-np.mean(acceleration), fs)
    min_freq_f = len(f[f < min_freq])
    max_freq_ind_temp = np.argmax(pxx[min_freq_f:min_freq_f * 5])
    max_freq = f[min_freq_f + max_freq_ind_temp]
    if max_freq > 3:
        max_freq = 3
    cutoff_freq = 1.5 * max_freq
    frame_acceleration_p = lowpass_butter(acceleration, sfreq=fs, cutoff=cutoff_freq)
    std_fr_acc = np.std(frame_acceleration_p)
    push_acc_fr, push_acc_fr_ind = find_peaks(frame_acceleration_p, height=std_fr_acc/2,
                                              distance=round(1/(max_freq*1.5)*fs), prominence=std_fr_acc/2)
    n_pushes = len(push_acc_fr)
    push_freq = n_pushes/(len(acceleration)/fs)
    cycle_time = pd.DataFrame([])
    
    for n in range(0, len(push_acc_fr) - 1):
        cycle_time = cycle_time.append([(push_acc_fr[n + 1]/fs) - (push_acc_fr[n]/fs)])

    return push_acc_fr, frame_acceleration_p, n_pushes, cycle_time, push_freq