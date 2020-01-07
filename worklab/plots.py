import matplotlib.pyplot as plt
import numpy as np


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
