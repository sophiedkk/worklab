"""
-Plotting functionalities-
Description: Most variables can easily be plotted with matplotlib or pandas as most data in this package is contained
in dataframes. Some plotting is tedious however these are functions for those plots.
Author:     Rick de Klerk
Contact:    r.de.klerk@umcg.nl
Company:    University Medical Center Groningen
License:    GNU GPLv3.0
Date:       27/06/2019
"""
import matplotlib.pyplot as plt
import numpy as np


def plot_pushes(data, pushes, var="torque", start=True, stop=True, peak=True, ax=None):
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


def plot_ergometer_pushes(data, pushes, var="torque", start=True, stop=True, peak=True):
    _, axes = plt.subplots(2, 1, sharex="all", sharey="all")
    for idx, side in enumerate(data):
        axes[idx] = plot_pushes(data[side], pushes[side], var=var, start=start, stop=stop, peak=peak, ax=axes[idx])
    plt.tight_layout()
    return axes


def bland_altman_plot(data1, data2, ax=None, condition=None):
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
