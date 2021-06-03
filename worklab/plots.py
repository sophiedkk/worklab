import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.axisartist import ParasiteAxes
from mpl_toolkits.axisartist.parasite_axes import HostAxes

from .imu import push_imu
from .utils import lowpass_butter


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


def plot_pushes_ergo(data, pushes, title=None, var="power", start=True, stop=True, peak=True):
    """
    Plot left, right and mean side ergometer push data

    Parameters
    ----------
    data : dict
        processed ergometer data dictionary with dataframes
    pushes : dict
        processed push_by_push ergometer data dictionary with dataframes
    title : str
        title of the plot, optional
    var : str
        variable to plot, default is power
    start : bool
        plot push starts, default is True
    stop : bool
        plot push stops, default is True
    peak : bool
        plot push peaks, default is True

    Returns
    -------
    axes : np.array
        an array containing an axis for the left, right and mean side

    """
    _, axes = plt.subplots(3, 1, sharex="all", sharey="all")
    if title:
        plt.suptitle('Push detection: ' + str(title))
    if not title:
        plt.suptitle('Push detection')
    for idx, side in enumerate(data):
        axes[idx] = plot_pushes(data[side], pushes[side], var=var, start=start, stop=stop, peak=peak, ax=axes[idx])
        axes[idx].set_title(str(side) + ' ' + str(var))
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


def vel_plot(time, vel, name=''):
    """
    Plot velocity versus time

    Parameters
    ----------
    time : np.array, pd.Series
        time structure
    vel : np.array, pd.Series
        velocity structure
    name : str
        name of a session

    Returns
    -------
    ax: axis object

    """
    plt.style.use("seaborn-darkgrid")
    fig, ax = plt.subplots(1, 1, figsize=[10, 6])
    ax.plot(time, vel, 'r')
    ax.set_xlabel("Time [s]", fontsize=12)
    ax.set_ylabel("Velocity [m/s]", fontsize=12)
    ax.tick_params(axis='y', colors='r', labelsize=12)
    ax.tick_params(axis='x', labelsize=12)
    ax.yaxis.label.set_color('r')
    ax.set_title(f"{name} Velocity")
    ax.set_ylim(0, np.max(vel) + 0.5)
    ax.autoscale(axis='x', tight=True)

    return ax


def vel_peak_plot(time, vel, name=''):
    """
    Plot velocity versus time, with vel_peak

    Parameters
    ----------
    time : np.array, pd.Series
        time structure
    vel : np.array, pd.Series
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
    plt.style.use("seaborn-darkgrid")
    fig, ax = plt.subplots(1, 1, figsize=[10, 6])
    ax.plot(time, vel, 'r')
    ax.plot(time[y_max_vel], vel[y_max_vel], 'ko',
            label='Vel$_{peak}$: ' + str(round(y_max_vel_value, 2)) + ' m/s')
    ax.set_xlabel("Time [s]", fontsize=12)
    ax.set_ylabel("Velocity [m/s]", fontsize=12)
    ax.tick_params(axis='y', colors='r', labelsize=12)
    ax.tick_params(axis='x', labelsize=12)
    ax.yaxis.label.set_color('r')
    ax.set_title(f"{name} Velocity with vel_peak")
    ax.legend(loc='lower right', prop={'size': 12})
    ax.set_ylim(0, y_max_vel_value + 0.5)
    ax.autoscale(axis='x', tight=True)

    return ax


def vel_peak_dist_plot(time, vel, dist, name=''):
    """
    Plot velocity and distance against time

    Parameters
    ----------
    time : np.array, pd.Series
        time structure
    vel : np.array, pd.Series
        velocity structure
    dist : np.array, pd.Series
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
    plt.style.use("seaborn-darkgrid")
    fig, ax1 = plt.subplots(1, 1, figsize=[10, 6])
    ax1.set_ylim(0, y_max_vel_value + 0.5)
    ax1.plot(time, vel, 'r')
    ax1.plot(time[y_max_vel], vel[y_max_vel], 'ko',
             label='Vel$_{peak}$: ' + str(round(y_max_vel_value, 2)) + ' m/s')
    ax1.set_xlabel("Time [s]", fontsize=12)
    ax1.set_ylabel("Velocity [m/s]", fontsize=12)
    ax1.yaxis.label.set_color('r')
    ax1.tick_params(axis='y', colors='r', labelsize=12)
    ax1.tick_params(axis='x', labelsize=12)
    ax1.set_title(f"{name} Velocity and distance with vel_peak")
    ax1.legend(loc='lower right', prop={'size': 12})
    ax1.autoscale(axis='x', tight=True)

    # Create time vs. distance figure
    ax2 = ax1.twinx()
    ax2.set_ylim(0, max(dist)+1)
    ax2.plot(time, dist)
    ax2.plot(time[y_max_vel], dist[y_max_vel], 'ko')
    ax2.set_ylabel("Distance [m]", fontsize=12)
    ax2.tick_params(axis='y', colors='b', labelsize=12)
    ax2.yaxis.label.set_color('b')
    ax2.autoscale(axis='x', tight=True)

    return ax1, ax2


def acc_plot(time, acc, name=''):
    """
    Plot acceleration versus time

    Parameters
    ----------
    time : np.array, pd.Series
        time structure
    acc : np.array, pd.Series
        acceleration structure
    name : str
        name of a session

    Returns
    -------
    ax: axis object

    """
    # Create time vs. acceleration figure
    plt.style.use("seaborn-darkgrid")
    fig, ax = plt.subplots(1, 1, figsize=[10, 6])
    ax.plot(time, acc, 'g')
    ax.set_xlabel("Time [s]", fontsize=12)
    ax.set_ylabel("Acceleration [m/$s^2$]", fontsize=12)
    ax.tick_params(axis='y', colors='g', labelsize=12)
    ax.tick_params(axis='x', labelsize=12)
    ax.yaxis.label.set_color('g')
    ax.set_title(f"{name} Acceleration")
    ax.set_ylim(np.min(acc) - 1, np.max(acc) + 1)
    ax.autoscale(axis='x', tight=True)

    return ax


def acc_peak_plot(time, acc, name=''):
    """
    Plot acceleration versus time, with acc_peak

    Parameters
    ----------
    time : np.array, pd.Series
        time structure
    acc : np.array, pd.Series
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
    plt.style.use("seaborn-darkgrid")
    fig, ax = plt.subplots(1, 1, figsize=[10, 6])
    ax.plot(time, acc, 'g')
    ax.plot(time[y_max_acc], acc[y_max_acc], 'k.',
            label='Acc$_{peak}$: ' + str(round(y_max_acc_value, 2)) + ' m/$s^2$')
    ax.set_xlabel("Time [s]", fontsize=12)
    ax.set_ylabel("Acceleration [m/$s^2$]", fontsize=12)
    ax.tick_params(axis='y', colors='g', labelsize=12)
    ax.tick_params(axis='x', labelsize=12)
    ax.yaxis.label.set_color('g')
    ax.set_title(f"{name} Acceleration with acc_peak")
    ax.legend(loc='lower center', prop={'size': 12})
    ax.set_ylim(np.min(acc) - 1, y_max_acc_value + 1)
    ax.autoscale(axis='x', tight=True)

    return ax


def acc_peak_dist_plot(time, acc, dist, name=''):
    """
    Plot acceleration and distance versus time, with acc_peak

    Parameters
    ----------
    time : np.array, pd.Series
        time structure
    acc : np.array, pd.Series
        acceleration structure
    dist : np.array, pd.Series
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
    plt.style.use("seaborn-darkgrid")
    fig, ax1 = plt.subplots(1, 1, figsize=[10, 6])
    ax1.set_ylim(np.min(acc) - 1, y_max_acc_value + 1)
    ax1.plot(time, acc, 'g')
    ax1.plot(time[y_max_acc], acc[y_max_acc], 'k.',
             label='Acc$_{peak}$: ' + str(round(y_max_acc_value, 2)) + ' m/$s^2$')
    ax1.set_xlabel("Time [s]", fontsize=12)
    ax1.set_ylabel("Acceleration [m/$s^2$]", fontsize=12)
    ax1.tick_params(axis='y', colors='g', labelsize=12)
    ax1.tick_params(axis='x', labelsize=12)
    ax1.yaxis.label.set_color('g')
    ax1.legend(loc='lower center', prop={'size': 12})
    ax1.set_title(f"{name} Acceleration and distance with acc_peak")
    ax1.autoscale(axis='x', tight=True)

    # Create time vs. distance figure
    ax2 = ax1.twinx()
    ax2.set_ylim(0, max(dist) + 1)
    ax2.plot(time, dist)
    ax2.plot(time[y_max_acc], dist[y_max_acc], 'k.')
    ax2.set_ylabel("Distance [m]", fontsize=12)
    ax2.tick_params(axis='y', colors='b', labelsize=12)
    ax2.yaxis.label.set_color('b')
    ax2.autoscale(axis='x', tight=True)

    return ax1, ax2


def rot_vel_plot(time, rot_vel, name=''):
    """
    Plot rotational velocity versus time

    Parameters
    ----------
    time : np.array, pd.Series
        time structure
    rot_vel : np.array, pd.Series
        rotational velocity structure
    name : str
        name of a session

    Returns
    -------
    ax: axis object

    """
    # Create time vs. rotational velocity figure
    plt.style.use("seaborn-darkgrid")
    fig, ax = plt.subplots(1, 1, figsize=[10, 6])
    ax.plot(time, rot_vel, 'b')
    ax.set_xlabel("Time [s]", fontsize=12)
    ax.set_ylabel("Rotational velocity [deg/s]", fontsize=12)
    ax.tick_params(axis='y', colors='b', labelsize=12)
    ax.tick_params(axis='x', labelsize=12)
    ax.yaxis.label.set_color('b')
    ax.set_title(f"{name} Rotational Velociy")
    ax.set_ylim(np.min(rot_vel) - 10, np.max(rot_vel) + 10)
    ax.autoscale(axis='x', tight=True)

    return ax


def set_axes_equal_3d(axes):
    """
    Set 3D plot axes to equal scale and size

    Parameters
    ----------
    axes : matplotlib.axes._subplots.Axes3DSubplot
        axes containing 3D plotted data
    """
    axes.set_box_aspect([1, 1, 1])
    limits = np.array([axes.get_xlim3d(),
                       axes.get_ylim3d(),
                       axes.get_zlim3d()])
    origin = np.mean(limits, axis=1)
    radius = 0.5 * np.max(np.abs(limits[:, 1] - limits[:, 0]))
    x, y, z = origin
    axes.set_xlim3d([x - radius, x + radius])
    axes.set_ylim3d([y - radius, y + radius])
    axes.set_zlim3d([z - radius, z + radius])


def imu_push_plot(time, vel, acc_raw, name=''):
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
    push_idx, acc_filt, n_pushes, cycle_time, push_freq = push_imu(acc_raw, sfreq)

    # Calculate processed acceleration from velocity
    acc = lowpass_butter(np.gradient(vel) * sfreq, sfreq=sfreq, cutoff=20)

    # Create time vs. velocity with push detection figure
    plt.style.use("seaborn-darkgrid")
    fig, ax1 = plt.subplots(1, 1, figsize=[10, 6])
    ax1.set_ylim(-6, 6)
    ax1.plot(time, vel, 'r')
    ax1.plot(time[push_idx], vel[push_idx], 'k.', markersize=10)
    ax1.set_xlabel("Time [s]", fontsize=12)
    ax1.set_ylabel("Velocity [m/s]", fontsize=12)
    ax1.tick_params(axis='y', colors='r', labelsize=12)
    ax1.tick_params(axis='x', labelsize=12)
    ax1.yaxis.label.set_color('r')
    ax1.set_title(f"{name} Push detection Sprint test")
    ax1.autoscale(axis='x', tight=True)

    # Create time vs. acceleration with push detection figure
    ax2 = ax1.twinx()
    ax2.set_ylim(-25, 25)
    ax2.plot(time, acc, 'C7', alpha=0.5)
    ax2.plot(time, acc_filt, 'b')
    ax2.plot(time[push_idx], acc_filt[push_idx], 'k.', markersize=10, label="Detected push")
    ax2.set_ylabel("Acceleration [m/$s^2$]", fontsize=12)
    ax2.tick_params(axis='y', colors='b', labelsize=12)
    ax2.yaxis.label.set_color('b')
    ax2.legend(frameon=True)
    ax2.autoscale(axis='x', tight=True)

    return ax1, ax2


def plot_power_speed_dist(data, title="", ylim_power=None, ylim_speed=None, ylim_distance=None):
    """
    Plot power, speed and distance versus time for left (solid line) and
    right (dotted line) seperately

    Figure scales automatically, unless you specify it manually with the ylim_* arguments

    Parameters
    ----------
    data: dict
        processed ergometer data dictionary with dataframes
    title: str
        a title for the plot
    ylim_power: list [min, max] of float or int, optional
        list of the minimal and maximal ylim for power in W
    ylim_speed: list [min, max] of floats or int, optional
        list of the minimal and maximal ylim for speed in km/h
    ylim_distance: list [min, max] of floats or int, optional
        list of the minimal and maximal ylim for distance in m

    Returns
    -------
    fig: matplotlib.figure.Figure
    axes: tuple
        the three axes objects

    """
    plt.style.use("seaborn-ticks")
    fig = plt.figure()

    # Generate three axes
    host = HostAxes(fig, [0.15, 0.1, 0.65, 0.8])
    par1 = ParasiteAxes(host, sharex=host)
    par2 = ParasiteAxes(host, sharex=host)
    host.parasites.append(par1)
    host.parasites.append(par2)

    # Edit axis visibility
    host.axis["right"].set_visible(False)
    par1.axis["right"].set_visible(True)
    par1.axis["right"].major_ticklabels.set_visible(True)
    par1.axis["right"].label.set_visible(True)

    # Define third ax
    new_axisline = par2.get_grid_helper().new_fixed_axis
    par2.axis["right2"] = new_axisline(loc="right", axes=par2, offset=(60, 0))
    fig.add_axes(host)

    # Plot data
    host.plot(data["left"]["time"], data["left"]["power"], "forestgreen", label="Power left")
    host.plot(data["right"]["time"], data["right"]["power"], "forestgreen", linestyle="dotted", label="Power right")
    par1.plot(data["left"]["time"], data["left"]["speed"], "firebrick", label="Speed left", alpha=0.7)
    par1.plot(data["right"]["time"], data["right"]["speed"], "firebrick", linestyle="dotted", label="Speed right", alpha=0.7)
    par2.plot(data["left"]["time"], data["left"]["dist"], "y", label="Distance left", alpha=0.5)
    par2.plot(data["right"]["time"], data["right"]["dist"], "y", linestyle="dotted", label="Distance right", alpha=0.5)

    host.autoscale(tight=True, axis="x")

    # Scale figure (manually or automatically)
    if ylim_power:
        host.set_ylim(ylim_power[0], ylim_power[1])
    if not ylim_power:
        host.set_ylim(0., max(max(data["left"]["power"]), max(data["right"]["power"])) * 1.5)

    if ylim_speed:
        par1.set_ylim(ylim_speed[0], ylim_speed[1])
    if not ylim_speed:
        par1.set_ylim(0., max(max(data["left"]["speed"]), max(data["right"]["speed"])) * 1.1)

    if ylim_distance:
        par2.set_ylim(ylim_distance[0], ylim_distance[1])
    if not ylim_distance:
        par2.set_ylim(0., max(max(data["left"]["dist"]), max(data["right"]["dist"])) * 1.1)

    host.set_title(title)
    host.set_xlabel("Time [s]")
    host.set_ylabel("Power [W]")
    par1.set_ylabel("Speed [km/h]")
    par2.set_ylabel("Distance [m]")

    host.legend(loc="upper left", frameon=True)  # make the legend
    host.axis["left"].label.set_color("forestgreen")
    par1.axis["right"].label.set_color("firebrick")
    par2.axis["right2"].label.set_color("y")

    return fig, (host, par1, par2)
