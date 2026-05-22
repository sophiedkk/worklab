import copy
from warnings import warn
import numpy as np
from scipy.integrate import cumulative_trapezoid
from scipy.signal import periodogram, find_peaks, savgol_filter, correlate
from scipy import signal
from .utils import lowpass_butter, pd_interp
import imufusion


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


def cut_imu(sessiondata, slice_1=1, slice_2=5, side='right'):
    """
    Align IMU signals based on correlation of sensors

    Parameters
    ----------
    sessiondata : dict
        original sessiondata structure
    slice_1 : np.int
        starting time to cut data (s)
    slice_2: np.int
        stopping time to cut data (s), if input is 'end' (string), it will cut till the end
    side: string
        location of wheel sensor

    Returns
    -------
    sessiondata : dict
        cut sessiondata

    """
    sfreq = 1 / sessiondata[side]["time"].diff().mean()
    slice_1 *= sfreq

    if type(slice_2) is str:
       slice_2 = sessiondata[side]['time'].iloc[-1]
    slice_2 *= sfreq

    for sensor in sessiondata:
        sessiondata[sensor] = sessiondata[sensor].iloc[slice_1.astype('int64'):slice_2.astype('int64'), :]
        sessiondata[sensor] = sessiondata[sensor].reset_index(drop=True)
        sessiondata[sensor]['time'] -= sessiondata[sensor]['time'][0]

    return sessiondata


def align_signals(signal1, signal2):
    """
    Align IMU signals based on correlation of sensors

    Parameters
    ----------
    signal1 : pd.Series
        signal to align with
    signal2: pd.Series
        signal to align to

    Returns
    -------
    signal2: pd.Series
        signal to align to
    lag: np.int
        number of samples to shift signals to align sensors

    """
    corr = correlate(signal1, signal2, mode='full')
    lag = np.argmax(corr) - len(signal2) + 1

    return signal2, lag


def camber_angle(wheel_data, accel_units='g'):
    """
    Camber estimated from wheel IMU

    Parameters
    ----------
    wheel_data : pd.DataFrame
        DataFrame with wheeldata
    accel_units: string
        default is g, other inputs can be m/s2

    Returns
    -------
    wheel_data : pd.DataFrame
        DataFrame with processed wheeldata

    """
    # convert paper thresholds (rad/s) into deg/s for your raw gyroscope units
    sfreq = 1 / wheel_data["time"].diff().mean()

    gyro_y_thresh_deg = np.rad2deg(5.0)  # ~286.48 deg/s
    gyro_xz_thresh_deg = np.rad2deg(0.2)  # ~11.459 deg/s

    # selection mask uses raw gyroscope values (deg/s)
    mask = (wheel_data.gyroscope_y > gyro_y_thresh_deg) & \
           (np.abs(wheel_data.gyroscope_x) < gyro_xz_thresh_deg) & \
           (np.abs(wheel_data.gyroscope_z) < gyro_xz_thresh_deg)
    subset = wheel_data[mask].head(int(2 * sfreq))

    # Fallback: first 2 s if mask empty
    if len(subset) == 0:
        subset = wheel_data.head(int(2 * sfreq))

    ay = subset.accelerometer_y.values

    # convert to g if needed
    if accel_units.lower() == 'm/s2':
        ay = ay / 9.81

    # Use mean offset of Y-axis (normalized to g) to compute camber
    # paper described arctan(ay/g) — here ay already normalized if accel_units='m/s2' handled above
    ay_mean = np.mean(ay) if len(ay) > 0 else 0.0
    # Compute camber angle in radians
    theta = abs(np.rad2deg(np.arctan(ay_mean)))
    return theta


def gyro_bias(wheel_data):
    """
    Gyroscope bias based on Klimstra

    Parameters
    ----------
    wheel_data : pd.DataFrame
        DataFrame with wheeldata

    Returns
    -------
    wheel_data : pd.DataFrame
        DataFrame with processed wheeldata

    """
    for axis in ['gyroscope_x', 'gyroscope_y', 'gyroscope_z']:
        wheel_data[axis] = np.deg2rad(wheel_data[axis])
        mask = np.abs(wheel_data[axis]) < np.percentile(np.abs(wheel_data[axis]), 10)
        wheel_data[axis] -= np.mean(wheel_data[axis].values[mask])

    return wheel_data


def mounting_misallignment(wheel_data):
    """
    Mounting misallignment based on Klimstra

    Parameters
    ----------
    wheel_data : pd.DataFrame
        DataFrame with wheeldata

    Returns
    -------
    wheel_data : pd.Dataframe
        Dataframe with processed wheeldata

    """
    sfreq = 1 / wheel_data["time"].diff().mean()

    mask = (np.abs(wheel_data.gyroscope_y) > 5) & \
           (np.abs(wheel_data.gyroscope_x) < 0.2) & \
           (np.abs(wheel_data.gyroscope_z) < 0.2)
    subset = wheel_data[mask].head(int(2 * sfreq))

    sx = np.mean(subset.gyroscope_x / subset.gyroscope_y) if len(subset) > 0 else 0
    sz = np.mean(subset.gyroscope_z / subset.gyroscope_y) if len(subset) > 0 else 0

    wheel_data.gyroscope_x = lowpass_butter(wheel_data.gyroscope_x, sfreq, cutoff=6)
    wheel_data.gyroscope_z = lowpass_butter(wheel_data.gyroscope_z, sfreq, cutoff=6)
    wheel_data.gyroscope_y = lowpass_butter(wheel_data.gyroscope_y, sfreq, cutoff=6)
    gyro_x_corr = wheel_data.gyroscope_x - sx * wheel_data.gyroscope_y
    gyro_z_corr = wheel_data.gyroscope_z - sz * wheel_data.gyroscope_y
    wheel_data.gyroscope_x = gyro_x_corr
    wheel_data.gyroscope_z = gyro_z_corr

    return wheel_data


def frame_rot(sessiondata, ca=20, ws=0.34, side='right', method='ahrs'):
    """
    Estimate frame rotational velocity with one IMU

    Parameters
    ----------
    sessiondata : dict
        original sessiondata structure
    ca : float
        camber angle (degrees)
    ws : float
        radius of the wheels (meters)
    side: string
        wheel is situated on which side, default is 'right'
    method: string
        One IMU processing method, default is 'ahrs' (can be changed to Klimstra)


    Returns
    -------
    sessiondata : dict
        sessiondata structure with estimated frame rotational data

    """
    if method == 'ahrs':
        wheel_data = np.array(sessiondata[side])
        sfreq = 1 / sessiondata[side]["time"].diff().mean()
        timestamp = wheel_data[:, 0]
        gyroscope = wheel_data[:, 1:4]
        accelerometer = wheel_data[:, 4:7]

        gyroscope[:, 0] = lowpass_butter(gyroscope[:, 0], sfreq, cutoff=6)
        gyroscope[:, 1] = lowpass_butter(gyroscope[:, 1], sfreq, cutoff=6)
        gyroscope[:, 2] = lowpass_butter(gyroscope[:, 2], sfreq, cutoff=6)

        accelerometer[:, 0] = lowpass_butter(accelerometer[:, 0], sfreq, cutoff=6)
        accelerometer[:, 1] = lowpass_butter(accelerometer[:, 1], sfreq, cutoff=6)
        accelerometer[:, 2] = lowpass_butter(accelerometer[:, 2], sfreq, cutoff=6)

        # Process sensor data
        ahrs = imufusion.Ahrs()
        euler = np.empty((len(timestamp), 3))

        for index in range(len(timestamp)):
            ahrs.update_no_magnetometer(gyroscope[index], accelerometer[index], 1 / sfreq)
            euler[index] = ahrs.quaternion.to_euler()
        euler2 = np.deg2rad(euler)

        gyro_x_corr = sessiondata[side].gyroscope_x
        gyro_z_corr = sessiondata[side].gyroscope_z
        if side == 'right':
            frame_rot_euler2 = (-euler2[:, 1] / np.deg2rad(90 - ca)) * gyro_x_corr + (
                        (euler2[:, 0] + np.deg2rad(90)) / np.deg2rad(90 - ca)) * gyro_z_corr
        else:
            frame_rot_euler2 = (-euler2[:, 1] / np.deg2rad(90 - ca)) * gyro_x_corr + (
                        (-euler2[:, 0] + np.deg2rad(90)) / np.deg2rad(90 - ca)) * gyro_z_corr

        frame_rot_euler2_filt = lowpass_butter(frame_rot_euler2, sfreq, cutoff=10)
        sessiondata[side]['rot_vel'] = frame_rot_euler2_filt
    else:
        wheel_data = sessiondata[side]
        sfreq = 1 / sessiondata[side]["time"].diff().mean()

        # Gyro bias
        wheel_data = gyro_bias(wheel_data)

        # Ensure omega_y in rad/s for ar = omega^2 * r
        gyro_units = 'deg/s'
        omega_y = wheel_data['gyroscope_y'].values
        if gyro_units.lower() in ['deg/s', 'dps', 'deg/s.']:
            omega_y_rad = omega_y * (np.pi / 180.0)
        else:
            omega_y_rad = omega_y.copy()

        # radial acceleration in m/s^2
        a_r_mss = (omega_y_rad ** 2) * ws

        # convert radial acceleration to accel input units (g or m/s2)
        accel_units = 'g'
        if accel_units.lower() == 'g':
            a_r = a_r_mss / 9.81  # now in g
        else:
            a_r = a_r_mss  # already m/s^2

        # subtract radial accel from measured accel z to isolate gravity component
        wheel_data['accelerometer_z_gf'] = wheel_data['accelerometer_z'] - a_r

        # --- Camber correction using gyro Y-axis ---
        theta = camber_angle(wheel_data, accel_units='g')
        wheel_data[['accelerometer_x', 'accelerometer_y', 'accelerometer_z']] /= np.cos(np.deg2rad(theta))
        wheel_data[['gyroscope_x', 'gyroscope_y', 'gyroscope_z']] /= np.cos(np.deg2rad(theta))

        # Mounting misalignment
        wheel_data = mounting_misallignment(wheel_data)

        # --- Project axes using orientation ---
        # --- Compute orientation using gravity-isolated accelerometer z ---
        # Use ax and az_gf to compute phi (Klimstra)
        phi = np.arctan2(wheel_data.accelerometer_x.values, wheel_data['accelerometer_z_gf'].values)
        gyro_x_corr = wheel_data.gyroscope_x
        gyro_z_corr = wheel_data.gyroscope_z

        # --- Project corrected gyro axes into frame rotation ---
        frame_rot_klim = gyro_x_corr * np.sin(phi) + gyro_z_corr * np.cos(phi)

        # # --- Savgol filter
        wl = int(0.25 * sfreq)
        if wl % 2 == 0: wl += 1
        # frame_rot_filtered = savgol_filter(frame_rot, wl, polyorder=3)
        frame_rot_filtered = lowpass_butter(frame_rot_klim, sfreq, cutoff=6)
        for axis in ['gyroscope_x', 'gyroscope_y', 'gyroscope_z']:
            wheel_data[axis] = np.rad2deg(wheel_data[axis])

        sessiondata[side]['rot_vel'] = np.rad2deg(frame_rot_filtered)

    return sessiondata


def process_imu(sessiondata, camber=18, wsize=0.32, wbase=0.80, n_sensors=3, sensor_type='ximu3', side='right',
                inplace=False, method='ahrs', alignment_correction=False):
    """
    Calculate wheelchair kinematic variables based on NGIMU data

    Parameters
    ----------
    sessiondata : dict
        original sessiondata structure
    camber : float
        camber angle in degrees
    wsize : float
        radius of the wheels (meters)
    wbase : float
        width of wheelbase (meters)
    n_sensors: float
        number of sensors used: 1: wheel 2: wheel and frame, 3: right wheel, left wheel and frame
    sensor_type: string
        type of sensor, 'ngimu' or 'ximu3' is for xio-technologies, 'move' is for movesense
    side: string
        wheel is situated on which side, default is 'right' (only change with one or two sensors)
    inplace : bool
        performs operation inplace
    method: string
        One IMU processing method, default is 'ahrs' (can be changed to Klimstra)
    alignment_correction: bool
        Additional alignment correction based on rotational velocity


    Returns
    -------
    sessiondata : dict
        sessiondata structure with processed data

    """
    if not inplace:
        sessiondata = copy.deepcopy(sessiondata)
    if n_sensors == 3:
        side = 'right'

    if side == 'right':
        right = sessiondata[side]
        sfreq = 1 / right["time"].diff().mean()
        right['gyroscope_y'] = lowpass_butter(right['gyroscope_y'], sfreq=sfreq, cutoff=10)
    else:
        left = sessiondata[side]
        sfreq = 1 / left["time"].diff().mean()
        left['gyroscope_y'] = lowpass_butter(left['gyroscope_y'], sfreq=sfreq, cutoff=10)

    if n_sensors > 1:
        frame = sessiondata["frame"]
        # frame["rot_vel"] = lowpass_butter(frame["gyroscope_z"], sfreq=sfreq, cutoff=6)
        frame['rot_vel'] = savgol_filter(frame['gyroscope_z'], window_length=100, polyorder=3)
        # Wheelchair camber correction
        if side == 'right':
            right["gyro_cor"] = right["gyroscope_y"] + np.tan(np.deg2rad(camber)) * (
                    frame["rot_vel"] * np.cos(np.deg2rad(camber)))
            sessiondata = frame_rot(sessiondata, side='right')
            if alignment_correction:
                sessiondata['right']['rot_vel'], lag_right = align_signals(sessiondata['frame']['rot_vel'],
                                                                           sessiondata['right']['rot_vel'])
                for column in sessiondata['right']:
                    if lag_right > 0:
                        sessiondata['right'][column] = np.pad(sessiondata['right'][column], (lag_right, 0))[
                            :len(sessiondata['frame'])]
                    elif lag_right < 0:
                        sessiondata['right'][column] = np.pad(sessiondata['right'][column], (0, -lag_right))[
                            -lag_right:len(sessiondata['frame']) - lag_right]
        else:
            left["gyro_cor"] = left["gyroscope_y"] + np.tan(np.deg2rad(camber)) * (
                    frame["rot_vel"] * np.cos(np.deg2rad(camber)))
            sessiondata = frame_rot(sessiondata, side='left')
            if alignment_correction:
                sessiondata['left']['rot_vel'], lag_left = align_signals(sessiondata['frame']['rot_vel'],
                                                                         sessiondata['left']['rot_vel'])
                for column in sessiondata['left']:
                    if lag_left > 0:
                        sessiondata['left'][column] = np.pad(sessiondata['left'][column], (lag_left, 0))[
                            :len(sessiondata['frame'])]
                    elif lag_left < 0:
                        sessiondata['left'][column] = np.pad(sessiondata['left'][column], (0, -lag_left))[
                            -lag_left:len(sessiondata['frame']) - lag_left]
    else:
        # Estimate frame rotation from wheel sensor
        # Wheelchair camber correction
        if side == 'right':
            sessiondata = frame_rot(sessiondata, method=method)
            right['rot_vel'] = savgol_filter(right['rot_vel'], window_length=100, polyorder=3)
            if method == 'klimstra':
                right['gyro_cor'] = right['gyroscope_y']
            elif method == 'ahrs':
                right["gyro_cor"] = right["gyroscope_y"] + np.tan(np.deg2rad(camber)) * (
                        right["rot_vel"] * np.cos(np.deg2rad(camber)))

        else:
            sessiondata = frame_rot(sessiondata, side='left')
            left['rot_vel'] = savgol_filter(left['rot_vel'], window_length=100, polyorder=3)
            if method == 'klimstra':
                left['gyro_cor'] = left['gyroscope_y']
            elif method == 'ahrs':
                left["gyro_cor"] = left["gyroscope_y"] + np.tan(np.deg2rad(camber)) * (
                        left["rot_vel"] * np.cos(np.deg2rad(camber)))

    if n_sensors == 3:
        left = sessiondata["left"]
        sessiondata = frame_rot(sessiondata, side='left')
        if alignment_correction:
            sessiondata['left']['rot_vel'], lag_left = align_signals(sessiondata['frame']['rot_vel'],
                                                                     sessiondata['left']['rot_vel'])
            for column in sessiondata['right']:
                if lag_left > 0:
                    sessiondata['right'][column] = np.pad(sessiondata['right'][column], (lag_left, 0))[
                        :len(sessiondata['frame'])]
                elif lag_left < 0:
                    sessiondata['right'][column] = np.pad(sessiondata['right'][column], (0, -lag_left))[
                        -lag_left:len(sessiondata['frame']) - lag_left]

        left['gyroscope_y'] = lowpass_butter(left['gyroscope_y'], sfreq=sfreq, cutoff=10)
        left["gyro_cor"] = left["gyroscope_y"] + np.tan(np.deg2rad(camber)) * (
                frame["rot_vel"] * np.cos(np.deg2rad(camber)))
        frame["gyro_cor"] = (right["gyro_cor"] + left["gyro_cor"]) / 2
    elif n_sensors == 2:
        if side == 'right':
            frame["gyro_cor"] = right["gyro_cor"]
        else:
            frame["gyro_cor"] = left["gyro_cor"]

    # Calculation of velocity, acceleration and distance
    if side == 'right':
        right["vel_wheel"] = np.deg2rad(right["gyro_cor"]) * wsize  # angular velocity to linear velocity
        right["vel_wheel"] = lowpass_butter(right["vel_wheel"], sfreq=sfreq, cutoff=10)
        right["vel"] = right['vel_wheel']
        right["acc_wheel"] = lowpass_butter(np.gradient(right["vel"]) * sfreq, sfreq=sfreq,
                                            cutoff=10)  # mean acceleration from velocity
        right['acc'] = right['acc_wheel']
        right["dist"] = cumulative_trapezoid(right["vel"] / sfreq, initial=0.0)  # integral of velocity gives distance
    if side == 'left' or n_sensors == 3:
        left["vel_wheel"] = np.deg2rad(left["gyro_cor"]) * wsize  # angular velocity to linear velocity
        left["vel_wheel"] = lowpass_butter(left["vel_wheel"], sfreq=sfreq, cutoff=10)
        left["vel"] = left['vel_wheel']
        left["acc_wheel"] = lowpass_butter(np.gradient(left["vel"]) * sfreq, sfreq=sfreq,
                                            cutoff=10)  # mean acceleration from velocity
        left['acc'] = left['acc_wheel']
        left["dist"] = cumulative_trapezoid(left["vel"] / sfreq, initial=0.0)  # integral of velocity gives distance

    if n_sensors > 1:
        # Calculation of rotations, rotational velocity and rotational acceleration
        frame["rot"] = cumulative_trapezoid(abs(frame["rot_vel"]) / sfreq, initial=0.0)
        frame["rot_acc"] = np.gradient(frame["rot_vel"]) * sfreq

        if sensor_type == 'ngimu' or sensor_type == 'ximu3':  # Acceleration for NGIMU/XIMU3 is in g
            frame["accelerometer_x"] = frame["accelerometer_x"] * 9.81
        frame['acc'] = lowpass_butter(frame['accelerometer_x'], sfreq=sfreq, cutoff=10)
    else:
        if side == 'right':
            right["rot"] = cumulative_trapezoid(abs(right["rot_vel"]) / sfreq, initial=0.0)
            right["rot_acc"] = np.gradient(right["rot_vel"]) * sfreq
        else:
            left["rot"] = cumulative_trapezoid(abs(left["rot_vel"]) / sfreq, initial=0.0)
            left["rot_acc"] = np.gradient(left["rot_vel"]) * sfreq

    if n_sensors > 1:
        if side == 'right':
            frame["vel_right"] = right["vel_wheel"]  # Calculate frame centre distance
            frame['vel_wheel'] = frame['vel_right']
            right['vel'] -= np.tan(np.deg2rad(frame["rot_vel"] / sfreq)) * wbase / 2 * sfreq
            frame["vel"] = right["vel"]
        if side == 'left' or n_sensors == 3:
            frame["vel_left"] = left["vel_wheel"]  # Calculate frame centre distance
            frame['vel_wheel'] = frame['vel_left']
            left["vel"] -= np.tan(np.deg2rad(frame["rot_vel"] / sfreq)) * wbase / 2 * sfreq
            frame["vel"] = left["vel"]

        frame["dist"] = cumulative_trapezoid(frame["vel"], initial=0.0) / sfreq  # Combined distance

    """Perform skid correction from Rienk vd Slikke, please refer and reference to: Van der Slikke, R. M. A., et. al.
    Wheel skid correction is a prerequisite to reliably measure wheelchair sports kinematics based on inertial sensors.
    Procedia Engineering, 112, 207-212."""
    if n_sensors == 3:
        left["vel"] -= np.tan(np.deg2rad(frame["rot_vel"] / sfreq)) * wbase / 2 * sfreq
        r_ratio0 = np.abs(right["vel_wheel"]) / (
                    np.abs(right["vel_wheel"]) + np.abs(left["vel_wheel"]))  # Ratio left and right
        l_ratio0 = np.abs(left["vel_wheel"]) / (np.abs(right["vel_wheel"]) + np.abs(left["vel_wheel"]))
        r_ratio1 = np.abs(np.gradient(left["vel_wheel"])) / (np.abs(np.gradient(right["vel_wheel"]))
                                                             + np.abs(np.gradient(left["vel_wheel"])))
        l_ratio1 = np.abs(np.gradient(right["vel_wheel"])) / (np.abs(np.gradient(right["vel_wheel"]))
                                                              + np.abs(np.gradient(left["vel_wheel"])))

        comb_ratio = (r_ratio0 * r_ratio1) / ((r_ratio0 * r_ratio1) + (l_ratio0 * l_ratio1))  # Combine speed ratios
        comb_ratio.fillna(value=0., inplace=True)
        comb_ratio = lowpass_butter(comb_ratio, sfreq=sfreq, cutoff=20)  # Filter the signal
        comb_ratio = np.clip(comb_ratio, 0, 1)  # clamp Combine ratio values, not in df
        frame["skid_vel"] = (right['vel_wheel'] * comb_ratio) + (left["vel_wheel"] * (1 - comb_ratio))
        frame["vel"] = (right["vel"] + left["vel"]) / 2
        frame['vel_wheel'] = (frame["vel_right"] + frame["vel_left"]) / 2
        frame['dist'] = cumulative_trapezoid(frame["skid_vel"], initial=0.0) / sfreq
    if n_sensors > 1:
        frame["acc_wheel"] = lowpass_butter(np.gradient(frame["vel"]) * sfreq, sfreq=sfreq,
                                            cutoff=10)  # mean acceleration from velocity
        # distance in the x and y direction
        frame["dist_y"] = cumulative_trapezoid(
            frame['vel'] / sfreq * np.sin(np.deg2rad(cumulative_trapezoid(frame["rot_vel"] / sfreq, initial=0.0))),
            initial=0.0)
        frame["dist_x"] = cumulative_trapezoid(
            frame['vel'] / sfreq * np.cos(np.deg2rad(cumulative_trapezoid(frame["rot_vel"] / sfreq, initial=0.0))),
            initial=0.0)
    else:
        if side == 'right':
            right["dist_y"] = cumulative_trapezoid(
                right['vel'] / sfreq * np.sin(np.deg2rad(cumulative_trapezoid(right["rot_vel"] / sfreq, initial=0.0))),
                initial=0.0)
            right["dist_x"] = cumulative_trapezoid(
                right['vel'] / sfreq * np.cos(np.deg2rad(cumulative_trapezoid(right["rot_vel"] / sfreq, initial=0.0))),
                initial=0.0)
            right["acc_wheel"] = lowpass_butter(np.gradient(right["vel"]) * sfreq, sfreq=sfreq,
                                                cutoff=10)  # mean acceleration from velocity
            sessiondata['frame'] = sessiondata['right']
        else:
            left["dist_y"] = cumulative_trapezoid(
                left['vel'] / sfreq * np.sin(np.deg2rad(cumulative_trapezoid(left["rot_vel"] / sfreq, initial=0.0))),
                initial=0.0)
            left["dist_x"] = cumulative_trapezoid(
                left['vel'] / sfreq * np.cos(np.deg2rad(cumulative_trapezoid(left["rot_vel"] / sfreq, initial=0.0))),
                initial=0.0)
            left["acc_wheel"] = lowpass_butter(np.gradient(left["vel"]) * sfreq, sfreq=sfreq,
                                               cutoff=10)  # mean acceleration from velocity
            sessiondata['frame'] = sessiondata['left']

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


def offset(sessiondata, n_sensors=2, right_wheel=True, gyro_offset=False):
    """
    Remove potential offset in IMU data

    Parameters
    ----------
    sessiondata : dict
        resampled sessiondata structure
    right_wheel: bool
        if set to True, right wheel is used, if set to False, left wheel is used
    n_sensors: float
        number of sensors used, 2: right wheel and frame,
        3: right, left wheel and frame
    gyro_offset: bool
        if set to True, an additional gyroscope_z frame offset correction will be used

    Returns
    -------
    sessiondata : dict
        sessiondata with offset removed

    """
    if right_wheel:
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

        if right_wheel:
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
    if gyro_offset:
        sessiondata['frame']['gyroscope_z'] = np.sign(
            sessiondata['frame']['gyroscope_z']) * np.sqrt(sessiondata['frame']['gyroscope_x']**2
                                                           + sessiondata['frame']['gyroscope_y']**2
                                                           + sessiondata['frame']['gyroscope_z']**2)

    return sessiondata


def imu_synch(sessiondata, right_wheel=True, inplace=False):
    """
    Synchronise wheel and frame sensor IMUs

    Parameters
    ----------
    sessiondata : dict
        original sessiondata structure
    right_wheel: boolean
        if set to True, right wheel is used, if set to False, left wheel is used
    inplace : bool
        perform operation inplace

    Returns
    -------
    sessiondata : dict
        sessiondata with reoriented gyroscope data

    """
    if not inplace:
        sessiondata = copy.deepcopy(sessiondata)

    x = sessiondata['frame']['gyroscope_x']
    if right_wheel:
        y = sessiondata['right']['gyroscope_x']
    else:
        y = sessiondata['left']['gyroscope_x']

    correlation = signal.correlate(x - np.mean(x), y - np.mean(y), mode="full")
    lags = signal.correlation_lags(len(x), len(y), mode="full")
    lag = lags[np.argmax(abs(correlation))]
    if lag > 0:
        sessiondata['frame'] = sessiondata['frame'][lag:].reset_index(drop=True)
    else:
        if right_wheel:
            sessiondata['right'] = sessiondata['right'][abs(lag):].reset_index(drop=True)
        else:
            sessiondata['left'] = sessiondata['left'][abs(lag):].reset_index(drop=True)

    return sessiondata
