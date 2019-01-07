"""
-Default plotting functions for ergometer/measurement wheel data-
Description: Some default plots for regular data; pyqtgraph and matplotlib!
Author:     Rick de Klerk
Contact:    r.de.klerk@umcg.nl
Company:    University Medical Center Groningen
License:    GNU GPLv3.0
Date:       26/03/2018
"""

import matplotlib.pyplot as plt
from . import formats


# def auto_plot(data):
#     if isinstance(data, formats.Kinetics):
#
#
#
#     return fig, ax


def plot_multiple_vars(data, dvars=None):
    with plt.style.context("seaborn"):
        if not dvars:
            dvars = ["force", "speed"]
        ncols = len(dvars)
        # noinspection PyTypeChecker
        fig, ax = plt.subplots(1, ncols, sharex=True)
        for n, var in enumerate(dvars):
            ax[n].plot(data["left"]["time"], data["left"][var])
        plt.autoscale(enable=True, axis='x', tight=True)
        labels = formats.get_names_units()
        for n, var in enumerate(dvars):
            ax[n].set_xlabel(labels["time"][0] + " (" + labels["time"][1] + ")")
            ax[n].set_ylabel(labels[var][0] + " (" + labels[var][1] + ")")
            ax[n].set_title(labels[var][0] + " over time")
    return fig, ax


def plot_single(data, var=None):
    with plt.style.context("seaborn"):
        if not var:
            var = "force"
        fig, ax = plt.subplots()
        ax.plot(data["time"], data[var])
        plt.autoscale(enable=True, axis='both', tight=True)
        labels = formats.get_names_units()
        ax.set_xlabel(labels["time"][0] + " (" + labels["time"][1] + ")")
        ax.set_ylabel(labels[var][0] + " (" + labels[var][1] + ")")
        ax.set_title(labels[var][0] + " over time")
    return fig, ax


def plot_pushes(data, pushes, var="torque"):
    with plt.style.context("seaborn"):
        fig, ax = plt.subplots()
        ax.plot(data["time"], data[var])
        ax.plot(data["time"][pushes["start"]], data[var][pushes["start"]], "C1o")
        ax.plot(data["time"][pushes["end"]], data[var][pushes["end"]], "C1o")
        ax.plot(data["time"][pushes["peak"]], data[var][pushes["peak"]], "C2o")
        labels = formats.get_names_units()
        ax.set_xlabel(labels["time"][0] + " (" + labels["time"][1] + ")")
        ax.set_ylabel(labels[var][0] + " (" + labels[var][1] + ")")
        ax.set_title(labels[var][0] + " over time with individual pushes")
    return fig, ax
