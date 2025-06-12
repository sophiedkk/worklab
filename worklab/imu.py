import copy
from warnings import warn

import numpy as np
from scipy.integrate import cumulative_trapezoid
from scipy.signal import periodogram, find_peaks, savgol_filter

from .utils import lowpass_butter, pd_interp


def resample_imu(sessiondata, sfreq=400.0):
    """
    Resample all devices and sensors to new sample frequency.

    Resamples all devices and sensors to new sample frequency. Sample intervals are not fixed with ngimu so resampling
    before further analysis is recommended. Translated from xio-Technologies.

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
    https://github.com/xioTechnologies/NGIMU-MATLAB-Import-Logged-Data-Example

    """
    end_time = np.inf
    for device in sessiondata:
        max_time = sessiondata[device]["time"].max()
        end_time = max_time if max_time < end_time else end_time

    new_time = np.arange(0, end_time, 1 / sfreq)

    for device in sessiondata:
        if device == "quaternion":
            sessiondata[device] = pd_interp(sessiondata[device], "time", new_time)
            sessiondata[device] *= 1 / np.linalg.norm(sessiondata[device], axis=0)
        elif device == "matrix":
            warn("Rotation matrix cannot be resampled. This dataframe has been removed")
        else:
            sessiondata[device] = pd_interp(sessiondata[device], "time", new_time)
    return sessiondata


def process_imu(sessiondata, camber=18, wsize=0.32, wbase=0.80, n_sensors=3, sensor_type='ngimu', inplace=False):
    """
    Calculate wheelchair kinematic variables based on NGIMU data

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
    n_sensors: float
        number of sensors used: 2: right wheel and frame, 3: right, left wheel and frame
    sensor_type: string
        type of sensor, 'ngimu' or 'ximu3' is for xio-technologies, 'move' is for movesense
    inplace : bool
        performs operation inplace


    Returns
    -------
    sessiondata : dict
        sessiondata structure with processed data

    """
    if not inplace:
        sessiondata = copy.deepcopy(sessiondata)
    frame = sessiondata["frame"]
    right = sessiondata["right"]

    sfreq = 1 / frame["time"].diff().mean()
    frame["rot_vel"] = lowpass_butter(frame["gyroscope_z"], sfreq=sfreq, cutoff=6)
    frame['rot_vel'] = savgol_filter(frame['rot_vel'], window_length=100, polyorder=3)
    right['gyroscope_y'] = lowpass_butter(right['gyroscope_y'], sfreq=sfreq, cutoff=10)

    # Wheelchair camber correction
    deg2rad = np.pi / 180
    right["gyro_cor"] = right["gyroscope_y"] + np.tan(camber * deg2rad) * (
        frame["rot_vel"] * np.cos(camber * deg2rad))
    if n_sensors == 3:
        left = sessiondata["left"]
        left['gyroscope_y'] = lowpass_butter(left['gyroscope_y'], sfreq=sfreq, cutoff=10)
        left["gyro_cor"] = left["gyroscope_y"] - np.tan(camber * deg2rad) * (
            frame["rot_vel"] * np.cos(camber * deg2rad))
        frame["gyro_cor"] = (right["gyro_cor"] + left["gyro_cor"]) / 2
    else:
        frame["gyro_cor"] = right["gyro_cor"]

    # Calculation of rotations, rotational velocity and rotational acceleration
    frame["rot"] = cumulative_trapezoid(abs(frame["rot_vel"]) / sfreq, initial=0.0)
    frame["rot_acc"] = np.gradient(frame["rot_vel"]) * sfreq

    # Calculation of velocity, acceleration and distance
    right["vel"] = right["gyro_cor"] * wsize * deg2rad  # angular velocity to linear velocity
    right["dist"] = cumulative_trapezoid(right["vel"] / sfreq, initial=0.0)  # integral of velocity gives distance

    if n_sensors == 3:
        left["vel"] = left["gyro_cor"] * wsize * deg2rad
        left["dist"] = cumulative_trapezoid(left["vel"] / sfreq, initial=0.0)
        frame["vel_wheel"] = (right["vel"] + left["vel"]) / 2  # mean velocity both sides
        frame["dist_wheel"] = (right["dist"] + left["dist"]) / 2  # mean distance
    else:
        frame["vel_wheel"] = right["vel"]
        frame["dist_wheel"] = right["dist"]

    frame["vel_wheel"] = lowpass_butter(frame["vel_wheel"], sfreq=sfreq, cutoff=10)
    frame["acc_wheel"] = np.gradient(frame["vel_wheel"]) * sfreq  # mean acceleration from velocity
    frame['acc_wheel'] = lowpass_butter(frame['acc_wheel'], sfreq=sfreq, cutoff=10)

    if sensor_type == 'ngimu' or sensor_type == 'ximu3':  # Acceleration for NGIMU/XIMU3 is in g
        frame["accelerometer_x"] = frame["accelerometer_x"] * 9.81
    frame['acc'] = lowpass_butter(frame['accelerometer_x'], sfreq=sfreq, cutoff=10)

    """Perform skid correction from Rienk vd Slikke, please refer and reference to: Van der Slikke, R. M. A., et. al.
    Wheel skid correction is a prerequisite to reliably measure wheelchair sports kinematics based on inertial sensors.
    Procedia Engineering, 112, 207-212."""
    frame["vel_right"] = right["vel"]  # Calculate frame centre distance
    frame["vel_right"] -= np.tan(np.deg2rad(frame["rot_vel"] / sfreq)) * wbase / 2 * sfreq

    if n_sensors == 3:
        frame["vel_left"] = left["vel"]
        frame["vel_left"] += np.tan(np.deg2rad(frame["rot_vel"] / sfreq)) * wbase / 2 * sfreq

        r_ratio0 = np.abs(right["vel"]) / (np.abs(right["vel"]) + np.abs(left["vel"]))  # Ratio left and right
        l_ratio0 = np.abs(left["vel"]) / (np.abs(right["vel"]) + np.abs(left["vel"]))
        r_ratio1 = np.abs(np.gradient(left["vel"])) / (np.abs(np.gradient(right["vel"]))
                                                       + np.abs(np.gradient(left["vel"])))
        l_ratio1 = np.abs(np.gradient(right["vel"])) / (np.abs(np.gradient(right["vel"]))
                                                        + np.abs(np.gradient(left["vel"])))

        comb_ratio = (r_ratio0 * r_ratio1) / ((r_ratio0 * r_ratio1) + (l_ratio0 * l_ratio1))  # Combine speed ratios
        comb_ratio.fillna(value=0., inplace=True)
        comb_ratio = lowpass_butter(comb_ratio, sfreq=sfreq, cutoff=20)  # Filter the signal
        comb_ratio = np.clip(comb_ratio, 0, 1)  # clamp Combine ratio values, not in df
        frame["skid_vel"] = (frame["vel_right"] * comb_ratio) + (frame["vel_left"] * (1 - comb_ratio))
        frame["vel"] = (frame["vel_right"] + frame["vel_left"]) / 2
        frame['dist'] = cumulative_trapezoid(frame["skid_vel"], initial=0.0) / sfreq
    else:
        frame["vel"] = frame["vel_right"]
        frame["dist"] = cumulative_trapezoid(frame["vel"], initial=0.0) / sfreq  # Combined distance

    # distance in the x and y direction
    frame["dist_y"] = cumulative_trapezoid(
        frame['vel'] / sfreq * np.sin(np.deg2rad(cumulative_trapezoid(frame["rot_vel"] / sfreq, initial=0.0))),
        initial=0.0)
    frame["dist_x"] = cumulative_trapezoid(
        frame['vel'] / sfreq * np.cos(np.deg2rad(cumulative_trapezoid(frame["rot_vel"] / sfreq, initial=0.0))),
        initial=0.0)

    return sessiondata


def process_imu_left(sessiondata, camber=18, wsize=0.32, wbase=0.80,
                     sensor_type='ngimu', inplace=False):
    """
    Calculate wheelchair kinematic variables based on NGIMU data

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
    sensor_type: string
        type of sensor, 'ngimu' or 'ximu3' is xio-technologies, 'move' is movesense
    inplace : bool
        performs operation inplace


    Returns
    -------
    sessiondata : dict
        sessiondata structure with processed data

    """
    if not inplace:
        sessiondata = copy.deepcopy(sessiondata)
    frame = sessiondata["frame"]
    left = sessiondata["left"]
    sfreq = int(1 / frame["time"].diff().mean())

    # Calculation of rotations, rotational velocity and acceleration
    frame["rot_vel"] = lowpass_butter(frame["gyroscope_z"],
                                      sfreq=sfreq, cutoff=10)
    frame['rot_vel'] = savgol_filter(frame['rot_vel'], window_length=100, polyorder=3)
    frame["rot"] = cumulative_trapezoid(abs(frame["rot_vel"]) / sfreq, initial=0.0)
    frame["rot_acc"] = np.gradient(frame["rot_vel"]) * sfreq

    # Wheelchair camber correction
    deg2rad = np.pi / 180
    left['gyroscope_y'] = lowpass_butter(left['gyroscope_y'], sfreq=sfreq, cutoff=10)
    left["gyro_cor"] = left["gyroscope_y"] - np.tan(camber * deg2rad) * (
        frame["rot_vel"] * np.cos(camber * deg2rad))
    frame["gyro_cor"] = left["gyro_cor"]

    left["vel"] = left["gyro_cor"] * wsize * deg2rad
    left["dist"] = cumulative_trapezoid(left["vel"] / sfreq, initial=0.0)
    frame["vel_wheel"] = left["vel"]
    frame["vel_wheel"] = lowpass_butter(frame["vel_wheel"], sfreq=sfreq, cutoff=10)
    frame["dist_wheel"] = cumulative_trapezoid(frame["vel_wheel"] / sfreq, initial=0.0)

    frame["acc_wheel"] = np.gradient(frame["vel_wheel"]) * sfreq
    frame['acc_wheel'] = lowpass_butter(frame['acc_wheel'],
                                        sfreq=sfreq, cutoff=10)

    if sensor_type == 'ngimu' or sensor_type == 'ximu3':  # Acceleration for NGIMU/XIMU3 is in g
        frame["accelerometer_x"] = frame["accelerometer_x"] * 9.81
    frame['acc'] = lowpass_butter(frame['accelerometer_x'],
                                  sfreq=sfreq, cutoff=10)

    frame["vel_left"] = left["vel"]
    frame["vel_left"] += np.tan(np.deg2rad(frame["rot_vel"] / sfreq)) * wbase / 2 * sfreq
    frame["vel"] = frame["vel_left"]
    frame["dist"] = cumulative_trapezoid(frame["vel"], initial=0.0) / sfreq

    # distance in the x and y direction
    frame["dist_y"] = cumulative_trapezoid(
        frame['vel'] / sfreq * np.sin(np.deg2rad(cumulative_trapezoid(frame["rot_vel"] / sfreq, initial=0.0))),
        initial=0.0)
    frame["dist_x"] = cumulative_trapezoid(
        frame['vel'] / sfreq * np.cos(np.deg2rad(cumulative_trapezoid(frame["rot_vel"] / sfreq, initial=0.0))),
        initial=0.0)

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


def push_imu(acceleration, sfreq=400.0):
    """
    Push detection based on velocity signal of IMU on a wheelchair.

    Parameters
    ----------
    acceleration : np.array, pd.Series
        acceleration data structure
    sfreq : float
        sampling frequency

    Returns
    -------
        push_idx, acc_filt, n_pushes, cycle_time, push_freq

    References
    ----------
    van der Slikke, R., Berger, M., Bregman, D., & Veeger, D. (2016). Push characteristics in wheelchair court sport
    sprinting. Procedia engineering, 147, 730-734.

    """
    min_freq = 1.2
    f, pxx = periodogram(acceleration - np.mean(acceleration), sfreq)
    min_freq_f = len(f[f < min_freq])
    max_freq_ind_temp = np.argmax(pxx[min_freq_f: min_freq_f * 5])
    max_freq = f[min_freq_f + max_freq_ind_temp]
    max_freq = min(max_freq, 3.0)
    cutoff_freq = 1.5 * max_freq
    acc_filt = lowpass_butter(acceleration, sfreq=sfreq, cutoff=cutoff_freq)
    std_acc = np.std(acc_filt)
    push_idx, peak_char = find_peaks(
        acc_filt, height=std_acc / 2, distance=round(1 / (max_freq * 1.5) * sfreq), prominence=std_acc / 2
    )
    n_pushes = len(push_idx)
    push_freq = n_pushes / (len(acceleration) / sfreq)
    cycle_time = list()

    for n in range(0, len(push_idx) - 1):
        cycle_time.append((push_idx[n + 1] / sfreq) - (push_idx[n] / sfreq))

    return push_idx, acc_filt, n_pushes, cycle_time, push_freq


def movesense_offset(sessiondata, n_sensors=2, right_wheel=True, gyro_offset=False):
    """
    Remove offset MoveSense sensors

    Parameters
    ----------
    sessiondata : dict
        resampled sessiondata structure
    right_wheel: boolean
        if set to True, right wheel is used, if set to False, left wheel is used
    n_sensors: float
        number of sensors used, 2: right wheel and frame,
        3: right, left wheel and frame
    gyro_offset: boolean
        if set to True, an additional gyroscope offset will be used

    Returns
    -------
    sessiondata : dict
        sessiondata with offset removed

    """
    if right_wheel is True:
        offset_indices = (np.abs(sessiondata['frame']['gyroscope_z']) < 5) & (
            np.abs(sessiondata['right']['gyroscope_y']) < 5)
    else:
        offset_indices = (np.abs(sessiondata['frame']['gyroscope_z']) < 5) & (
            np.abs(sessiondata['left']['gyroscope_y']) < 5)

    if sum(offset_indices) > 10:
        offset_frame_x = np.mean(sessiondata['frame']['gyroscope_x'][offset_indices])
        offset_frame_y = np.mean(sessiondata['frame']['gyroscope_y'][offset_indices])
        offset_frame_z = np.mean(sessiondata['frame']['gyroscope_z'][offset_indices])
        sessiondata['frame']['gyroscope_x'] -= offset_frame_x
        sessiondata['frame']['gyroscope_y'] -= offset_frame_y
        sessiondata['frame']['gyroscope_z'] -= offset_frame_z

        if right_wheel is True:
            offset_right_y = np.mean(sessiondata['right']['gyroscope_y'][offset_indices])
            offset_right_z = np.mean(sessiondata['right']['gyroscope_z'][offset_indices])
            offset_right_x = np.mean(sessiondata['right']['gyroscope_x'][offset_indices])
            sessiondata['right']['gyroscope_y'] -= offset_right_y
            sessiondata['right']['gyroscope_z'] -= offset_right_z
            sessiondata['right']['gyroscope_x'] -= offset_right_x

        if n_sensors == 3 or right_wheel is False:
            offset_left_y = np.mean(sessiondata['left']['gyroscope_y'][offset_indices])
            offset_left_z = np.mean(sessiondata['left']['gyroscope_z'][offset_indices])
            offset_left_x = np.mean(sessiondata['left']['gyroscope_x'][offset_indices])
            sessiondata['left']['gyroscope_y'] -= offset_left_y
            sessiondata['left']['gyroscope_z'] -= offset_left_z
            sessiondata['left']['gyroscope_x'] -= offset_left_x
    else:
        print('No offset corrected')
    if gyro_offset is True:
        sessiondata['frame']['gyroscope_z'] = np.sign(
            sessiondata['frame']['gyroscope_z']) * np.sqrt(sessiondata['frame']['gyroscope_x']**2
                                                           + sessiondata['frame']['gyroscope_y']**2
                                                           + sessiondata['frame']['gyroscope_z']**2)

    return sessiondata
