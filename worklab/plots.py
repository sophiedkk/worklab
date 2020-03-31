import matplotlib.pyplot as plt
import numpy as np
from .imu import push_imu


def plot_pushes(data, pushes, var="torque", start=True, stop=True, peak=True, ax=None):
    """
    Plot pushes from measurement wheel or ergometer data.

    Parameters
    ----------
    data : pd.DataFrame
    pushes : pd.DataFrame
    var : str
        variable to plot, default is torque
    start : bool
        plot push starts, default is True
    stop : bool
        plot push stops, default is True
    peak : bool
        plot push peaks, default is True
    ax : axis object
        Axis to plot on, you can add your own or it will make a new one.

    Returns
    -------
    ax : axis object

    """
    with plt.style.context("seaborn-white"):
        if not ax:
            _, ax = plt.subplots(1, 1)
        ax.plot(data["time"], data[var])
        if start:
            ax.plot(data["time"][pushes["start"]], data[var][pushes["start"]], "C1o")
        if stop:
            ax.plot(data["time"][pushes["stop"]], data[var][pushes["stop"]], "C1o")
        if peak:
            ax.plot(data["time"][pushes["peak"]], data[var][pushes["peak"]], "C2o")
        ax.set_xlabel("time")
        ax.set_ylabel(var)
        ax.set_title(f"{var} over time")
    return ax


def plot_pushes_ergo(data, pushes, var="torque", start=True, stop=True, peak=True):
    """
    Plot left and right side ergometer push data.

    Parameters
    ----------
    data : dict
    pushes : dict
    var : str
        variable to plot, default is torque
    start : bool
        plot push starts, default is True
    stop : bool
        plot push stops, default is True
    peak : bool
        plot push peaks, default is True

    Returns
    -------
    axes : np.array
        an array containing an axis for the left and right side

    """
    _, axes = plt.subplots(2, 1, sharex="all", sharey="all")
    for idx, side in enumerate(data):
        axes[idx] = plot_pushes(data[side], pushes[side], var=var, start=start, stop=stop, peak=peak, ax=axes[idx])
    plt.tight_layout()
    return axes


def bland_altman_plot(data1, data2, ax=None, condition=None):
    """
    Make a Bland-Altman plot.

    A Bland–Altman plot (Difference plot) is a method of data plotting used in analyzing the agreement between two
    different assays.

    Parameters
    ----------
    data1 : np.array, pd.Series
        First variable
    data2 : np.array, pd.Series
        Second variable
    ax : axis object, optional
        Axis to plot on, you can add your own or it will make a new one.
    condition : str, optional
        add labels to the plot

    Returns
    -------
    ax : axis object

    """
    if not ax:
        fig, ax = plt.subplots(1, 1)
    data1 = np.asarray(data1)
    data2 = np.asarray(data2)
    mean = np.mean([data1, data2], axis=0)
    diff = data1 - data2  # Difference between data1 and data2
    md = np.mean(diff)  # Mean of the difference
    sd = np.std(diff, axis=0)  # Standard deviation of the difference

    with plt.style.context("seaborn-white"):
        ax.scatter(mean, diff)
        ax.axhline(0, color='dimgray', linestyle='-')
        ax.axhline(md, color='darkgray', linestyle='--')
        ax.axhline(md + 1.96 * sd, color='lightcoral', linestyle='--')
        ax.axhline(md - 1.96 * sd, color='lightcoral', linestyle='--')
        ax.set_ylim([md - 3 * sd, md + 3 * sd])
        if condition:
            ax.set_xlabel(f"Mean of {condition}")
            ax.set_ylabel(f"Difference between {condition}")
            ax.set_title(f"Bland-Altman plot: {condition} ")
    return ax


def vel_plot(time, vel, name):
    """
    Plot velocity versus time

    Parameters
    ----------
    time : dict
        time structure
    vel : dict
        velocity structure
    name : str
        name of a session

    Returns
    -------
    ax: axis object

    """
    plt.style.use("seaborn-whitegrid")
    fig, ax = plt.subplots(1, 1, figsize=[14, 10])
    ax.plot(time, vel, 'r')
    ax.set_xlabel("Time [s]")
    ax.set_ylabel("Velocity [m/s]")
    ax.yaxis.label.set_color('r')
    ax.tick_params(axis='y', colors='r')
    ax.set_title(f"Velocity: {name}")
    ax.spines['left'].set_color('r')
    ax.spines['bottom'].set_color('k')
    ax.autoscale(tight=True)

    return ax

def vel_peak_plot(time, vel, name):
    """
    Plot velocity versus time, with vel_peak

    Parameters
    ----------
    time : dict
        time structure
    vel : dict
        velocity structure
    name : str
        name of a session

    Returns
    -------
    ax: axis object

    """
    # Calculate vel_peak and position of vel_peak
    y_max_vel = vel.idxmax()
    y_max_vel_value = np.max(vel)

    # Create time vs. velocity figure with vel_peak
    plt.style.use("seaborn-whitegrid")
    fig, ax = plt.subplots(1, 1, figsize=[14, 10])
    ax.plot(time, vel, 'r')
    ax.plot(time[y_max_vel], vel[y_max_vel], 'ko',
            label='Vel$_{peak}$: ' + str(round(y_max_vel_value, 2)) + ' m/s')
    ax.set_ylabel("Velocity [m/s]")
    ax.yaxis.label.set_color('r')
    ax.tick_params(axis='y', colors='r')
    ax.set_xlabel("Time [s]")
    ax.set_title(f"Velocity with vel_peak: {name}")
    ax.spines['left'].set_color('r')
    ax.spines['bottom'].set_color('k')
    ax.legend(loc='lower right', prop={'size': 12})
    ax.autoscale(tight=True)

    return ax

def vel_peak_dist_plot(time, vel, dist, name):
    """
    Plot velocity and distance against time

    Parameters
    ----------
    time : dict
        time structure
    vel : dict
        velocity structure
    dist : dict
        distance structure
    name : str
        name of a session

    Returns
    -------
    ax: axis object

    """
    # Calculate vel_peak and position of vel_peak
    y_max_vel = vel.idxmax()
    y_max_vel_value = np.max(vel)

    # Create time vs. velocity figure with vel_peak
    plt.style.use("seaborn-whitegrid")
    fig, ax1 = plt.subplots(1, 1, figsize=[14, 10])
    ax1.plot(time, vel, 'r')
    ax1.plot(time[y_max_vel], vel[y_max_vel], 'ko',
             label='Vel$_{peak}$: ' + str(round(y_max_vel_value, 2)) + ' m/s')
    ax1.set_ylabel("Velocity [m/s]")
    ax1.yaxis.label.set_color('r')
    ax1.tick_params(axis='y', colors='r')
    ax1.set_xlabel("Time [s]")
    ax1.set_title(f"Velocity and distance with vel_peak: {name}")
    ax1.legend(loc='lower right', prop={'size': 12})

    # Create time vs. distance figure
    ax2 = ax1.twinx()
    ax2.set_ylim(0, max(dist)+1)
    ax2.plot(time, dist)
    ax2.plot(time[y_max_vel], dist[y_max_vel], 'ko')
    ax2.yaxis.label.set_color('b')
    ax2.tick_params(axis='y', colors='b')
    ax2.spines['right'].set_color('b')
    ax2.spines['left'].set_color('r')
    ax2.spines['bottom'].set_color('k')
    ax2.set_ylabel("Distance [m]")
    ax1.autoscale(tight=True)

    return ax1, ax2


def acc_plot(time, acc, name):
    """
    Plot acceleration versus time

    Parameters
    ----------
    time : dict
        time structure
    acc : dict
        acceleration structure
    name : str
        name of a session

    Returns
    -------
    ax: axis object

    """
    # Create time vs. acceleration figure
    plt.style.use("seaborn-whitegrid")
    fig, ax = plt.subplots(1, 1, figsize=[14, 10])
    ax.plot(time, acc, 'g')
    ax.set_ylabel("Acceleration [m/$s^2$]")
    ax.yaxis.label.set_color('g')
    ax.tick_params(axis='y', colors='g')
    ax.set_xlabel("Time [s]")
    ax.set_title(f"Acceleration: {name}")
    ax.autoscale(tight=True)

    return ax


def acc_peak_plot(time, acc, name):
    """
    Plot acceleration versus time, with acc_peak

    Parameters
    ----------
    time : dict
        time structure
    acc : dict
        acceleration structure
    name : str
        name of a session

    Returns
    -------
    ax: axis object

    """
    # Calculate acc_peak and position of acc_peak
    y_max_acc_value = np.max(acc)
    y_max_acc = acc.idxmax()

    # Create time vs. acceleration figure with acc_peak
    plt.style.use("seaborn-whitegrid")
    fig, ax = plt.subplots(1, 1, figsize=[14, 10])
    ax.plot(time, acc, 'g')
    ax.plot(time[y_max_acc], acc[y_max_acc], 'k.',
            label='Acc$_{peak}$: ' + str(round(y_max_acc_value, 2)) + ' m/$s^2$')
    ax.set_ylabel("Acceleration [m/$s^2$]")
    ax.yaxis.label.set_color('g')
    ax.tick_params(axis='y', colors='g')
    ax.set_xlabel("Time [s]")
    ax.set_title(f"Acceleration with acc$_peak$: {name}")
    ax.legend(loc='lower center', prop={'size': 12})
    ax.autoscale(tight=True)

    return ax


def acc_peak_dist_plot(time, acc, dist, name):
    """
    Plot acceleration and distance versus time, with acc_peak

    Parameters
    ----------
    time : dict
        time structure
    vel : dict
        velocity structure
    dist : dict
        distance structure
    name : str
        name of a session

    Returns
    -------
    ax: axis object

    """
    # Calculate acc_peak and position of acc_peak
    y_max_acc_value = np.max(acc)
    y_max_acc = acc.idxmax()

    # Create time vs. acceleration figure with acc_peak
    plt.style.use("seaborn-whitegrid")
    fig, ax1 = plt.subplots(1, 1, figsize=[14, 10])
    ax1.plot(time, acc, 'g')
    ax1.plot(time[y_max_acc], acc[y_max_acc], 'k.',
             label='Acc$_{peak}$: ' + str(round(y_max_acc_value, 2)) + ' m/$s^2$')
    ax1.set_ylabel("Acceleration [m/$s^2$]")
    ax1.yaxis.label.set_color('g')
    ax1.tick_params(axis='y', colors='g')
    ax1.set_xlabel("Time [s]")
    ax1.legend(loc='lower center', prop={'size': 12})
    ax1.set_title(f"Acceleration and distance with acc_peak: {name}")

    # Create time vs. distance figure
    ax2 = ax1.twinx()
    ax2.plot(time, dist)
    ax2.plot(time[y_max_acc], dist[y_max_acc], 'k.')
    ax2.yaxis.label.set_color('b')
    ax2.tick_params(axis='y', colors='b')
    ax2.spines['right'].set_color('b')
    ax2.spines['left'].set_color('g')
    ax2.set_ylabel("Distance [m]")
    ax1.autoscale(tight=True)

    return ax1, ax2


def rot_vel_plot(time, rot_vel, name):
    """
    Plot rotational velocity versus time

    Parameters
    ----------
    time : dict
        time structure
    rot_vel : dict
        rotational velocity structure
    name : str
        name of a session

    Returns
    -------
    ax: axis object

    """
    # Create time vs. rotational velocity figure
    plt.style.use("seaborn-whitegrid")
    fig, ax = plt.subplots(1, 1, figsize=[20, 10])
    ax.plot(time, rot_vel, 'b')
    ax.set_xlabel("Time [s]")
    ax.set_ylabel("Rotational velocity [deg/s]")
    ax.yaxis.label.set_color('b')
    ax.tick_params(axis='y', colors='b')
    ax.set_title(f"Rotational Velociy: {name}")
    ax.autoscale(tight=True)

    return ax


def imu_push_plot(time, vel, acc_raw, name):
    """
    Plot push detection with IMUs

    Parameters
    ----------
    time : dict
        time structure
    vel : dict
        velocity structure
    acc_raw : dict
        raw acceleration structure
    name : str
        name of a session

    Returns
    -------
    ax: axis object

    """
    # Calculate push detection with function
    sfreq = 1 / time.diff().mean()
    push_idx, frame_acc_p, n_pushes, cycle_time, push_freq = push_imu(acc_raw, sfreq)

    # Create time vs. velocity with push detection figure
    plt.style.use("seaborn-whitegrid")
    fig, ax1 = plt.subplots(1, 1, figsize=[14, 10])
    ax1.set_ylim(-6, 6)
    ax1.plot(time, vel, 'r')
    ax1.plot(time[push_idx], vel[push_idx], 'k.')
    ax1.set_ylabel("Velocity [m/s]")
    ax1.yaxis.label.set_color('r')
    ax1.tick_params(axis='y', colors='r')
    ax1.set_xlabel("Time [s]")
    ax1.set_title(f"Push detection Sprint test: {name}")

    # Create time vs. acceleration with push detection figure
    ax2 = ax1.twinx()
    ax2.plot(time, frame_acc_p)
    ax2.plot(time[push_idx], frame_acc_p[push_idx], 'k.')
    ax2.yaxis.label.set_color('b')
    ax2.tick_params(axis='y', colors='b')
    ax2.spines['right'].set_color('b')
    ax2.spines['left'].set_color('r')
    ax2.set_ylabel("Acceleration [m/$s^2$]")
    ax2.set_ylim(-25, 25)

    return ax1, ax2


def straight_sprint_plot(vel, dist_x,
                         dist_y, name):
    """
    Plot straight sprint plot

    Parameters
    ----------
    vel : dict
        velocity structure
    dist_y : dict
        distance in Y direction structure
    dist_x : dict
        distance in X direction structure
    name : str
        name of a session

    Returns
    -------
    ax: axis object

    """
    # Calculate vel_peak and position of vel_peak
    y_max_vel = vel.idxmax()
    y_max_vel_value = np.max(vel)

    # Define vel zones
    vel_sin_2 = dist_y[vel > 2]
    vel_cos_2 = dist_x[vel > 2]
    vel_sin_3 = dist_y[vel > 3]
    vel_cos_3 = dist_x[vel > 3]
    vel_sin_4 = dist_y[vel > 4]
    vel_cos_4 = dist_x[vel > 4]

    # Create straight sprint figure
    plt.style.use("seaborn-whitegrid")
    fig, ax = plt.subplots(1, 1, figsize=[14, 10])
    ax.set_xlim(-6, 6)
    ax.set_ylim(0, max(dist_x)+1)
    ax.plot(-dist_y, dist_x)
    ax.plot(-vel_sin_2, vel_cos_2, 'y.', markersize=5,
            label='vel > 2 m/s')
    ax.plot(-vel_sin_3, vel_cos_3, 'g.', markersize=8,
            label='vel > 3 m/s')
    ax.plot(-vel_sin_4, vel_cos_4, 'r.', markersize=14,
            label='vel > 4 m/s')
    ax.plot(-dist_y[y_max_vel],
            dist_x[y_max_vel], 'ko', markersize=10,
            label='$vel_{peak}$: ' + str(round(y_max_vel_value, 2)) + ' m/s')
    ax.set_xlabel("Distance [m]")
    ax.set_ylabel("Distance [m]")
    ax.set_title(f"Sprint test: {name}")
    ax.legend()

    return ax


def overview_sprint_plot(time, vel, acc, dist, dist_x,
                         dist_y, acc_raw, name):
    """
    Plot overview straight sprint test

    Parameters
    ----------
    time : dict
        time structure
    vel : dict
        velocity structure
    dist : dict
        distance structure
    dist_y : dict
        distance in Y direction structure
    dist_x : dict
        distance in X direction structure
    acc_raw : dict
        raw acceleration structure
    name : str
        name of a session

    Returns
    -------
    ax: axis object

    """
    # Calculate push detection with function
    sfreq = 1 / time.diff().mean()
    push_idx, frame_acc_p, n_pushes, cycle_time, push_freq = push_imu(acc_raw, sfreq)

    # Calculate vel zones, vel_peak and acc_peak
    y_max_vel = vel.idxmax()
    y_max_acc = acc.idxmax()
    y_max_vel_value = np.max(vel)
    y_max_acc_value = np.max(acc)
    vel_sin_2 = dist_y[vel > 2]
    vel_cos_2 = dist_x[vel > 2]
    vel_sin_3 = dist_y[vel > 3]
    vel_cos_3 = dist_x[vel > 3]
    vel_sin_4 = dist_y[vel > 4]
    vel_cos_4 = dist_x[vel > 4]

    # Create time vs. velocity with push detection figure
    plt.style.use("seaborn-whitegrid")
    fig, [[ax1, ax2], [ax3, ax4]] = plt.subplots(2, 2, figsize=[16, 10])
    fig.suptitle(f"{name}")
    ax1.set_ylim(-6, 6)
    ax1.plot(time, vel, 'r')
    ax1.plot(time[push_idx], vel[push_idx], 'k.')
    ax1.set_ylabel("Velocity [m/s]")
    ax1.yaxis.label.set_color('r')
    ax1.tick_params(axis='y', colors='r')
    ax1.set_xlabel("Time [s]")
    # Create time vs. acceleration with push detection figure
    ax5 = ax1.twinx()
    ax5.plot(time, frame_acc_p)
    ax5.plot(time[push_idx], frame_acc_p[push_idx], 'k.')
    ax5.yaxis.label.set_color('b')
    ax5.tick_params(axis='y', colors='b')
    ax5.spines['right'].set_color('b')
    ax5.spines['left'].set_color('r')
    ax5.set_ylabel("Acceleration [m/$s^2$]")
    ax5.set_ylim(-30, 30)

    # Create time vs. velocity figure with vel_peak
    ax2.plot(time, vel, 'r')
    ax2.plot(time[y_max_vel], vel[y_max_vel], 'k.',
             label='Vel$_{peak}$: ' + str(round(y_max_vel_value, 2)) + ' m/s')
    ax2.set_ylabel("Velocity [m/s]")
    ax2.yaxis.label.set_color('r')
    ax2.tick_params(axis='y', colors='r')
    ax2.set_xlabel("Time [s]")
    ax2.legend(loc='lower right', prop={'size': 8})
    # Create time vs. distance figure
    ax6 = ax2.twinx()
    ax6.plot(time, dist)
    ax6.plot(time[y_max_vel], dist[y_max_vel], 'k.')
    ax6.yaxis.label.set_color('b')
    ax6.tick_params(axis='y', colors='b')
    ax6.spines['right'].set_color('b')
    ax6.spines['left'].set_color('r')
    ax6.set_ylabel("Distance [m]")

    # Create time vs. acceleration figure with acc_peak
    ax3.plot(time, acc, 'g')
    ax3.plot(time[y_max_acc], acc[y_max_acc], 'k.',
             label='Acc$_{peak}$: ' + str(round(y_max_acc_value, 2)) + ' m/$s^2$')
    ax3.set_ylabel("Acceleration [m/$s^2$]")
    ax3.yaxis.label.set_color('g')
    ax3.tick_params(axis='y', colors='g')
    ax3.set_xlabel("Time [s]")
    ax3.legend(loc='lower center', prop={'size': 8})
    # Create time vs. distance figure
    ax7 = ax3.twinx()
    ax7.plot(time, dist)
    ax7.plot(time[y_max_acc], dist[y_max_acc], 'k.')
    ax7.yaxis.label.set_color('b')
    ax7.tick_params(axis='y', colors='b')
    ax7.spines['right'].set_color('b')
    ax7.spines['left'].set_color('g')
    ax7.set_ylabel("Distance [m]")

    # Create Straight sprint figure with vel zones and vel_peak
    ax4.set_xlim(-6, 6)
    ax4.set_ylim(0, max(dist_x)+1)
    ax4.text(3, max(dist)-2, 'Endtime: ' + str(round(len(time) / sfreq, 2)) +'s',
             bbox=dict(facecolor='red', alpha=0.5))
    ax4.plot(-dist_y, dist_x)
    ax4.plot(-vel_sin_2, vel_cos_2, 'y.', markersize=5,
             label='Vel > 2 m/s')
    ax4.plot(-vel_sin_3, vel_cos_3, 'g.', markersize=8,
             label='Vel > 3 m/s')
    ax4.plot(-vel_sin_4, vel_cos_4, 'r.', markersize=14,
             label='Vel > 4 m/s')
    ax4.plot(-dist_y[y_max_vel],
             dist_x[y_max_vel], 'ko', markersize=10,
             label='Vel$_{peak}$: ' + str(round(y_max_vel_value, 2)) + ' m/s')
    ax4.set_xlabel("Distance [m]")
    ax4.set_ylabel("Distance [m]")
    ax4.legend(loc='upper left', prop={'size': 8})

    return ax1, ax2, ax3, ax4, ax5, ax6, ax7


def butterfly_plot(time, rot_vel, dist_x,
                   dist_y, name, mirror=False):
    """
    Plot butterfly sprint test

    Parameters
    ----------
    time : dict
        time structure
    rot_vel : dict
        rotational velocity structure
    dist_y : dict
        distance in Y direction structure
    dist_x : dict
        distance in X direction structure
    name : str
        name of a session
    mirror : bool
        make true if test is executed in other direction
    Returns
    -------
    ax: axis object

    """
    if mirror == True:
        dist_y = -dist_y
        dist_x = -dist_x
    # Caculate rotational vel zones and rot_vel_peak, rot_acc_peak
    sfreq = 1 / time.diff().mean()
    rot_vel.reset_index(inplace=True, drop=True)
    rot_vel_y_45 = dist_y[rot_vel.abs() > 45]
    rot_vel_x_45 = dist_x[rot_vel.abs() > 45]
    rot_vel_y_90 = dist_y[rot_vel.abs() > 90]
    rot_vel_x_90 = dist_x[rot_vel.abs() > 90]
    rot_vel_y_180 = dist_y[rot_vel.abs() > 180]
    rot_vel_x_180 = dist_x[rot_vel.abs() > 180]
    rotacc = np.gradient(rot_vel) * sfreq
    y_max_rot_vel = rot_vel.idxmax()
    y_max_rotacc = np.argmax(rotacc)
    y_max_rot_vel_value = np.max(rot_vel)
    y_max_rotacc_value = np.max(rotacc)

    # Create butterfly sprint figure
    plt.style.use("seaborn-whitegrid")
    fig, ax = plt.subplots(1, 1, figsize=[14, 10])
    ax.text(2, 7, 'Endtime: ' + str(round(len(time) / sfreq, 2)) + 's',
            bbox=dict(facecolor='green', alpha=0.5))
    ax.plot(dist_x, -dist_y)
    ax.plot(rot_vel_x_45, -rot_vel_y_45, 'y.', markersize=5,
            label='Rot vel > 45 deg/s')
    ax.plot(rot_vel_x_90, -rot_vel_y_90, 'g.', markersize=8,
            label='Rot vel > 90 deg/s')
    ax.plot(rot_vel_x_180, -rot_vel_y_180, 'r.', markersize=14,
            label='Rot vel > 180 deg/s')
    ax.plot(dist_x[y_max_rot_vel],
            -dist_y[y_max_rot_vel], 'ko', markersize=10,
            label='Rot vel$_{peak}$: ' + str(int(y_max_rot_vel_value)) + ' deg/s')
    ax.plot(dist_x[y_max_rotacc],
            -dist_y[y_max_rotacc], 'k*', markersize=10,
            label='Rot acc$_{peak}:$ ' + str(int(y_max_rotacc_value)) + ' deg/$s^2$')
    ax.set_xlabel("Distance [m]")
    ax.set_ylabel("Distance [m]")
    ax.set_title(f"Butterfly test: {name}")
    ax.legend(loc='upper left', prop={'size': 10})
    ax.autoscale()
    plt.tight_layout()
    plt.show()

    return ax


def overview_butterfly_plot(time, vel, acc, rot_vel, dist_x,
                            dist_y, name, mirror=False):
    """
    Plot butterfly sprint test overview

    Parameters
    ----------
    time : dict
        time structure
    vel : dict
        velocity structure
    acc : dict
        acceleration structure
    rot_vel : dict
        rotational velocity structure
    dist_y : dict
        distance in Y direction structure
    dist_x : dict
        distance in X direction structure
    name : str
        name of a session
    mirror : bool
        make true if test is executed in other direction
    Returns
    -------
    ax: axis object

    """
    if mirror == True:
        dist_y = -dist_y
        dist_x = -dist_x
    # Caculate rotational vel zones and rot_vel_peak, rot_acc_peak
    sfreq = 1 / time.diff().mean()
    rot_vel.reset_index(inplace=True, drop=True)
    rot_vel_y_45 = dist_y[rot_vel.abs() > 45]
    rot_vel_x_45 = dist_x[rot_vel.abs() > 45]
    rot_vel_y_90 = dist_y[rot_vel.abs() > 90]
    rot_vel_x_90 = dist_x[rot_vel.abs() > 90]
    rot_vel_y_180 = dist_y[rot_vel.abs() > 180]
    rot_vel_x_180 = dist_x[rot_vel.abs() > 180]
    rotacc = np.gradient(rot_vel) * sfreq
    y_max_rot_vel = rot_vel.idxmax()
    y_max_rotacc = np.argmax(rotacc)
    y_max_rot_vel_value = np.max(rot_vel)
    y_max_rotacc_value = np.max(rotacc)
    y_max_vel = vel.idxmax()
    y_max_acc = acc.idxmax()
    y_max_vel_value = np.max(vel)
    y_max_acc_value = np.max(acc)

    # Create time vs. rotational velocity figure
    plt.style.use("seaborn-whitegrid")
    fig, [[ax1, ax2], [ax3, ax4]] = plt.subplots(2, 2, figsize=[14, 10])
    fig.suptitle(f"{name}")
    ax1.plot(time, rot_vel, 'b')
    ax1.set_xlabel("Time [s]")
    ax1.set_ylabel("Rotational velocity [deg/s]")
    ax1.tick_params(axis='y', colors='b')
    ax1.yaxis.label.set_color('b')
    ax1.autoscale()

    # Create time vs. velocity figure with vel_peak
    ax2.plot(time, vel, 'r')
    ax2.plot(time[y_max_vel], vel[y_max_vel], 'k.',
             label='Vel$_{peak}$: ' + str(round(y_max_vel_value, 2)) + ' m/s')
    ax2.set_ylabel("Velocity [m/s]")
    ax2.yaxis.label.set_color('r')
    ax2.tick_params(axis='y', colors='r')
    ax2.set_xlabel("Time [s]")
    ax2.legend()
    ax2.set_title(" ")
    ax2.autoscale()

    # Create time vs. acceleration figure with acc_peak
    ax3.plot(time, acc, 'g')
    ax3.plot(time[y_max_acc], acc[y_max_acc], 'k.',
             label='Acc$_{peak}$: ' + str(round(y_max_acc_value, 2)) + ' m/$s^2$')
    ax3.set_ylabel("Acceleration [m/$s^2$]")
    ax3.yaxis.label.set_color('g')
    ax3.tick_params(axis='y', colors='g')
    ax3.set_xlabel("Time [s]")
    ax3.legend()
    ax3.autoscale()

    # Create butterfly sprint figure
    ax4.plot(dist_x, -dist_y)
    ax4.text(2, 7, 'Endtime: ' + str(round(len(time) / sfreq, 2)) + 's',
            bbox=dict(facecolor='green', alpha=0.5))
    ax4.plot(rot_vel_x_45, -rot_vel_y_45, 'y.', markersize=6,
             label='Rot vel > 45 deg/s')
    ax4.plot(rot_vel_x_90, -rot_vel_y_90, 'g.', markersize=8,
             label='Rot vel > 90 deg/s')
    ax4.plot(rot_vel_x_180, -rot_vel_y_180, 'r.', markersize=14,
             label='Rot vel > 180 deg/s')
    ax4.plot(dist_x[y_max_rot_vel],
             -dist_y[y_max_rot_vel], 'ko', markersize=10,
             label='Rot vel$_{peak}$: ' + str(int(y_max_rot_vel_value)) + ' deg/s')
    ax4.plot(dist_x[y_max_rotacc],
             -dist_y[y_max_rotacc], 'k*', markersize=10,
             label='Rot acc$_{peak}$: ' + str(int(y_max_rotacc_value)) + ' deg/$s^2$')
    ax4.set_xlabel("Distance [m]")
    ax4.set_ylabel("Distance [m]")
    ax4.legend(loc='upper left', prop={'size': 10})
    ax4.autoscale()
    plt.tight_layout()

    return ax1, ax2, ax3, ax4


def spider_plot(time, rot_vel, dist_x, dist_y,
                name, mirror=False):
    """
    Plot spider test

    Parameters
    ----------
    time : dict
        time structure
    rot_vel : dict
        rotational velocity structure
    dist_y : dict
        distance in Y direction structure
    dist_x : dict
        distance in X direction structure
    name : str
        name of a session
    mirror : bool
        make true if test is executed in other direction
    Returns
    -------
    ax: axis object

    """
    if mirror == True:
        dist_y = -dist_y
        dist_x = -dist_x
    # Caculate rotational vel zones and rot_vel_peak, rot_acc_peak
    sfreq = 1 / time.diff().mean()
    rot_vel.reset_index(inplace=True, drop=True)
    rot_vel_y_45 = dist_y[rot_vel.abs() > 45]
    rot_vel_x_45 = dist_x[rot_vel.abs() > 45]
    rot_vel_y_90 = dist_y[rot_vel.abs() > 90]
    rot_vel_x_90 = dist_x[rot_vel.abs() > 90]
    rot_vel_y_180 = dist_y[rot_vel.abs() > 180]
    rot_vel_x_180 = dist_x[rot_vel.abs() > 180]
    rotacc = np.gradient(rot_vel) * sfreq
    y_max_rot_vel = rot_vel.idxmax()
    y_max_rotacc = np.argmax(rotacc)
    y_max_rot_vel_value = np.max(rot_vel)
    y_max_rotacc_value = np.max(rotacc)

    # Create Spider figure
    plt.style.use("seaborn-whitegrid")
    fig, ax = plt.subplots(1, 1, figsize=[14, 10])
    if mirror == False:
        ax.text(1, 3, 'Endtime: ' + str(len(time) / sfreq) + 's',
                bbox=dict(facecolor='green', alpha=0.5))
    else:
        ax.text(3, 3, 'Endtime: ' + str(len(time) / sfreq) + 's',
                bbox=dict(facecolor='green', alpha=0.5))
    ax.plot(dist_x, dist_y)
    ax.plot(rot_vel_x_45, rot_vel_y_45, 'y.', markersize=5,
            label='Rot vel > 45 deg/s')
    ax.plot(rot_vel_x_90, rot_vel_y_90, 'g.', markersize=8,
            label='Rot vel > 90 deg/s')
    ax.plot(rot_vel_x_180, rot_vel_y_180, 'r.', markersize=14,
            label='Rot vel > 180 deg/s')
    ax.plot(dist_x[y_max_rot_vel],
            dist_y[y_max_rot_vel], 'ko', markersize=10,
            label='Rot vel$_{peak}:$ ' + str(int(y_max_rot_vel_value)) + ' deg/s')
    ax.plot(dist_x[y_max_rotacc],
            dist_y[y_max_rotacc], 'k*', markersize=10,
            label='Rot acc$_{peak}:$ ' + str(int(y_max_rotacc_value)) + ' deg/s$^{2}$')
    ax.set_xlabel("Distance [m]")
    ax.set_ylabel("Distance [m]")
    ax.set_title(f"Spider test: {name}")
    ax.legend(loc='upper left', prop={'size': 10})
    plt.tight_layout()

    return ax


def overview_spider_plot(time, vel, acc, rot_vel, dist_x,
                         dist_y, name, mirror=False):
    """
    Plot spider test

    Parameters
    ----------
    time : dict
        time structure
    vel : dict
        velocity structure
    acc : dict
        acceleration structure
    rot_vel : dict
        rotational velocity structure
    dist_y : dict
        distance in Y direction structure
    dist_x : dict
        distance in X direction structure
    name : str
        name of a session
    mirror : bool
        make true if test is executed in other direction
    Returns
    -------
    ax: axis object

    """
    if mirror == True:
        dist_y = -dist_y
        dist_x = -dist_x
    # Caculate rotational vel zones and rot_vel_peak, rot_acc_peak
    sfreq = 1 / time.diff().mean()
    rot_vel.reset_index(inplace=True, drop=True)
    rot_vel_y_45 = dist_y[rot_vel.abs() > 45]
    rot_vel_x_45 = dist_x[rot_vel.abs() > 45]
    rot_vel_y_90 = dist_y[rot_vel.abs() > 90]
    rot_vel_x_90 = dist_x[rot_vel.abs() > 90]
    rot_vel_y_180 = dist_y[rot_vel.abs() > 180]
    rot_vel_x_180 = dist_x[rot_vel.abs() > 180]
    rotacc = np.gradient(rot_vel) * sfreq
    y_max_rot_vel = rot_vel.idxmax()
    y_max_rotacc = np.argmax(rotacc)
    y_max_rot_vel_value = np.max(rot_vel)
    y_max_rotacc_value = np.max(rotacc)
    y_max_vel = vel.idxmax()
    y_max_acc = acc.idxmax()
    y_max_vel_value = np.max(vel)
    y_max_acc_value = np.max(acc)

    # Create time vs. rotational velocity figure
    plt.style.use("seaborn-whitegrid")
    fig, [[ax1, ax2], [ax3, ax4]] = plt.subplots(2, 2, figsize=[14, 10])
    fig.suptitle(f"{name}")
    ax1.plot(time, rot_vel, 'b')
    ax1.set_xlabel("Time [s]")
    ax1.set_ylabel("Rotational velocity [deg/s]")
    ax1.tick_params(axis='y', colors='b')
    ax1.yaxis.label.set_color('b')

    # Create time vs. velocity figure with vel_peak
    ax2.plot(time, vel, 'r')
    ax2.plot(time[y_max_vel], vel[y_max_vel], 'k.',
             label='Vel$_{peak}$: ' + str(round(y_max_vel_value, 2)) + ' m/s')
    ax2.set_ylabel("Velocity [m/s]")
    ax2.yaxis.label.set_color('r')
    ax2.tick_params(axis='y', colors='r')
    ax2.set_xlabel("Time [s]")
    ax2.legend(loc='lower right', prop={'size': 8})
    ax2.set_title(" ")

    # Create time vs. acceleration figure with acc_peak
    ax3.plot(time, acc, 'g')
    ax3.plot(time[y_max_acc], acc[y_max_acc], 'k.',
             label='Acc$_{peak}$: ' + str(round(y_max_acc_value, 2)) + ' m/$s^2$')
    ax3.set_ylabel("Acceleration [m/$s^2$]")
    ax3.yaxis.label.set_color('g')
    ax3.tick_params(axis='y', colors='g')
    ax3.set_xlabel("Time [s]")
    ax3.legend(loc='lower center', prop={'size': 8})

    # Create Spider figure
    ax4.set_ylim(-0.5, 4)
    ax4.set_xlim(-3.5, 2)
    ax4.text(0.9, 3.9, 'Endtime: ' + str(len(time) / sfreq) + 's',
             bbox=dict(facecolor='red', alpha=0.5))
    ax4.plot(dist_x, dist_y)
    ax4.plot(rot_vel_x_45, rot_vel_y_45, 'y.', markersize=5,
             label='rot_vel > 45 deg/s')
    ax4.plot(rot_vel_x_90, rot_vel_y_90, 'g.', markersize=8,
             label='rot_vel > 90 deg/s')
    ax4.plot(rot_vel_x_180, rot_vel_y_180, 'r.', markersize=14,
             label='rot_vel > 180 deg/s')
    ax4.plot(dist_x[y_max_rot_vel],
             dist_y[y_max_rot_vel], 'ko', markersize=10,
             label='rot_$vel_{peak}$: ' + str(int(y_max_rot_vel_value)) + ' deg/s')
    ax4.plot(dist_x[y_max_rotacc],
             dist_y[y_max_rotacc], 'k*', markersize=10,
             label='rot_$acc_{peak}$: ' + str(int(y_max_rotacc_value)) + ' deg/$s^2$')
    ax4.set_xlabel("Distance [m]")
    ax4.set_ylabel("Distance [m]")
    ax4.legend(loc='upper left', prop={'size': 6})

    return ax1, ax2, ax3, ax4
