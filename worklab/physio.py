import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import tkinter as tk
from tkinter import simpledialog

from .utils import find_nearest


def cut_spiro(data_spiro, start, end):
    """
    Cuts data to time of interest

    Parameters
    ----------
    data_spiro : pd.dataframe
        spirometer data
    start : float
        start time [s]
    end : float
        end time [s]

    Returns
    -------
    data_spiro : dataframe
        data cutted to time of interest

    """
    index_start = abs(data_spiro["time"] - start).idxmin() + 1
    index_end = abs(data_spiro["time"] - end).idxmin()
    data_spiro["time"] = data_spiro["time"] - start

    data_spiro = data_spiro.iloc[index_start:index_end]
    return data_spiro


def calc_weighted_average(dataframe, weights):
    """
    Calculate the weighted average of all columns in a DataFrame.

    Parameters
    ----------
    dataframe : pd.DataFrame
        input dataframe
    weights : pd.Series, np.array
        can be any iterable of equal length

    Returns
    -------
    averages : pd.Series
        the weighted averages of each column

    """
    return dataframe.apply(lambda col: np.average(col[~np.isnan(col)], weights=weights[~np.isnan(col)]), axis=0)


def wasserman(data_spiro, power, title=None):
    """
    Makes the 9 wasserman plot from a graded exercise test
    Plot 1: Ventilation & power vs time
    Plot 2: Heart rate & VO2/HR vs time
    Plot 3: VO2 & VCO2 vs time
    Plot 4: VE vs VCO2
    Plot 5: HR & VCO2 vs VO2
    Plot 6: VE/VO2 & VE/VCO2 against time
    Plot 7: VT (teugvolume) vs VE
    Plot 8: RER vs time
    Plot 9: PetO2 & PetCO2 vs time

    Parameters
    ----------
    data_spiro : pd.DataFrame
        spirometer data
    power : pd.DataFrame
         a dataframe containing the mean power output per step, showed as a continuous signal
         (use function: power_per_min)
    title : str, optional
        title of the plot. The default is None.

    Returns
    -------
    fig : matplotlib.figure.Figure
        9 wasserman plots
    result_gxt : pd.DataFrame
        most important outcomes of the maximal exercise test

    """
    fig, ax = plt.subplots(3, 3, figsize=[10, 7])
    if title:
        fig.suptitle("9 Wasserman plots " + str(title), size=20)
    if not title:
        fig.suptitle("9 Wasserman plots", size=20)

    # Plot 1: Ventilation & power vs time

    ax[0, 0].set_title("Plot - 1", fontweight="bold")
    ax[0, 0].plot(data_spiro["time"], data_spiro["VE"].rolling(window=15).mean(), color="darkblue")
    ax[0, 0].set_xlabel("time [s]")
    ax[0, 0].set_ylabel("VE [l/min]", {"color": "darkblue"})
    vemax = data_spiro["VE"].rolling(window=15).mean().max()
    ax[0, 0].text(0.05, 0.9, transform=ax[0, 0].transAxes, s="VEmax: " + str(round(vemax, 1)), color="darkblue")
    ax1 = ax[0, 0].twinx()
    ax1.plot(power.index, power["power"], color="forestgreen", linestyle="dashed")
    ax1.fill_between(power.index, power["power"], color="forestgreen", alpha=0.15)
    ax1.set_ylabel("Power [W]", {"color": "forestgreen"})
    popeak = power.max()
    ax1.text(0.05, 0.8, transform=ax1.transAxes, s="POpeak: " + str(round(popeak[0], 1)), color="forestgreen")

    # Plot 2: Heart rate & VO2/HR vs time

    ax[0, 1].set_title("Plot - 2", fontweight="bold")
    ax[0, 1].plot(data_spiro["time"], data_spiro["HR"], color="firebrick")
    ax[0, 1].set_xlabel("time [s]")
    ax[0, 1].set_ylabel("HR [b/min]", {"color": "firebrick"})
    ax[0, 1].set_ylim(0, 220)
    hrmax = data_spiro["HR"].max()
    ax[0, 1].text(0.05, 0.9, transform=ax[0, 1].transAxes, s="HR-max: " + str(hrmax), color="firebrick")
    ax1 = ax[0, 1].twinx()
    ax1.plot(data_spiro["time"], data_spiro["O2pulse"].rolling(window=15).mean(), color="darkblue", linestyle="dashed")
    ax1.set_ylim(0, 0.024)
    ax1.set_ylabel("VO2/HR [ml/min]", {"color": "darkblue"})

    # Plot 3: VO2 & VCO2 vs time

    ax[0, 2].set_title("Plot - 3", fontweight="bold")
    ax[0, 2].plot(data_spiro["time"], data_spiro["VO2"].rolling(window=15).mean(), color="darkblue")
    ax[0, 2].set_xlabel("time [s]")
    ax[0, 2].set_ylabel("VO2 [l/min]", {"color": "darkblue"})
    ax[0, 2].set_ylim(0, 1.05 * data_spiro["VCO2"].max())
    vo2max = data_spiro["VO2"].rolling(window=10).mean().max()
    ax[0, 2].text(0.05, 0.9, transform=ax[0, 2].transAxes, s="VO2_max: " + str(round(vo2max, 1)), color="darkblue")
    ax1 = ax[0, 2].twinx()
    ax1.plot(data_spiro["time"], data_spiro["VCO2"].rolling(window=15).mean(), color="firebrick")
    ax1.set_ylabel("VCO2 [l/min]", {"color": "firebrick"})
    ax1.set_ylim(0, 1.05 * data_spiro["VCO2"].max())
    ax2 = ax[0, 2].twinx()
    ax2.spines["right"].set_position(("axes", 1.2))
    ax2.plot(power.index, power["power"], color="forestgreen", linestyle="dashed", alpha=0.1)
    ax2.fill_between(power.index, power["power"], color="forestgreen", alpha=0.1)
    ax2.set_ylabel("Power [W]", {"color": "forestgreen"})

    # Plot 4: VE vs VCO2

    ax[1, 0].set_title("Plot - 4", fontweight="bold")
    ax[1, 0].scatter(data_spiro["VCO2"], data_spiro["VE"], color="darkblue", s=1)
    ax[1, 0].set_xlabel("VCO2 [l/min]", {"color": "red"})
    ax[1, 0].set_ylabel("VE [l/min]", {"color": "darkblue"})

    # Plot 5: HR & VCO2 vs VO2

    ax[1, 1].set_title("Plot - 5", fontweight="bold")
    ax[1, 1].scatter(data_spiro["VO2"], data_spiro["HR"], color="red", s=1)
    ax[1, 1].set_xlabel("VO2 [l/min]")
    ax[1, 1].set_ylabel("HR [b/min]", {"color": "red"})
    ax[1, 1].set_ylim(0, 220)
    ax1 = ax[1, 1].twinx()
    ax1.scatter(data_spiro["VO2"], data_spiro["VCO2"], color="darkblue", s=1)
    ax1.set_ylabel("VCO2 [l/min]", {"color": "darkblue"})

    # Plot 6: VE/VO2 & VE/VCO2 against time

    ax[1, 2].set_title("Plot - 6", fontweight="bold")
    ax[1, 2].plot(data_spiro["time"], data_spiro["VE/VO2"].rolling(window=15).mean(), color="darkblue")
    ax[1, 2].set_xlabel("time [s]")
    ax[1, 2].set_ylabel("VE/VO2", {"color": "darkblue"})
    ax[1, 2].set_ylim(0, 60)
    ax1 = ax[1, 2].twinx()
    ax1.plot(data_spiro["time"], data_spiro["VE/VCO2"].rolling(window=15).mean(), color="red")
    ax1.set_ylim(0, 60)
    ax1.set_ylabel("VE/VCO2", {"color": "red"})
    ax2 = ax[1, 2].twinx()
    ax2.spines["right"].set_position(("axes", 1.2))
    ax2.plot(power.index, power["power"], color="forestgreen", linestyle="dashed", alpha=0.1)
    ax2.fill_between(power.index, power["power"], color="forestgreen", alpha=0.1)
    ax2.set_ylabel("Power [W]", {"color": "forestgreen"})

    # Plot 7: VT (teugvolume) vs VE

    ax[2, 0].set_title("Plot - 7", fontweight="bold")
    ax[2, 0].scatter(data_spiro["VE"], data_spiro["VT"], color="darkblue", s=1)
    ax[2, 0].set_xlabel("VE [l/min]", {"color": "darkblue"})
    ax[2, 0].set_ylabel("TV [L]")
    ax[2, 0].set_ylim(0, 3.5)

    # Plot 8: RER vs time

    ax[2, 1].set_title("Plot - 8", fontweight="bold")
    ax[2, 1].plot(data_spiro["time"], data_spiro["RER"], color="lightblue")
    ax[2, 1].set_xlabel("time")
    ax[2, 1].set_ylabel("RER", {"color": "lightblue"})
    ax[2, 1].set_ylim(0, 1.6)
    rermax = data_spiro["RER"].rolling(window=15).mean().max()
    ax[2, 1].text(0.05, 0.9, transform=ax[2, 1].transAxes, s="RER_max: " + str(round(rermax, 3)), color="lightblue")
    ax[2, 1].hlines(1.1, data_spiro["time"][1], data_spiro["time"].iloc[-1], alpha=0.5, colors="k")

    # Plot 9: PetO2 & PetCO2 vs time

    if "PetO2" in list(data_spiro.columns):
        ax[2, 2].set_title("Plot - 9", fontweight="bold")
        ax[2, 2].plot(data_spiro["time"], data_spiro["PetO2"].rolling(window=15).mean(), color="darkblue")
        ax[2, 2].set_xlabel("time")
        ax[2, 2].set_ylabel("PetO2 [mmHg]", {"color": "darkblue"})
        ax[2, 2].set_ylim(50, 130)
        ax1 = ax[2, 2].twinx()
        ax1.plot(data_spiro["time"], data_spiro["PetCO2"].rolling(window=15).mean(), color="red")
        ax1.set_ylim(10, 90)
        ax1.set_ylabel("PetCO2 [mmHg]", {"color": "red"})
        ax2 = ax[2, 2].twinx()
        ax2.spines["right"].set_position(("axes", 1.2))
        ax2.plot(power.index, power["power"], color="forestgreen", linestyle="dashed", alpha=0.1)
        ax2.fill_between(power.index, power["power"], color="forestgreen", alpha=0.1)
        ax2.set_ylabel("Power [W]", {"color": "forestgreen"})

    plt.tight_layout()
    plt.subplots_adjust(top=0.90, hspace=0.5)

    # Save relevant outcomes from ergometer and spirometer
    result_gxt = [{"POpeak": popeak[0], "VO2peak": vo2max, "HRmax": hrmax, "RERmax": rermax, "VEmax": vemax}]
    result_gxt = pd.DataFrame(result_gxt)

    return fig, result_gxt


def aerobic_threshold(data_spiro, power, start_spiro, muser):
    """
    Shows four plots to determine the aerobic ventilatory threshold from the maximal exercise test
        Plot 1: HR vs VO2
        Plot 2: time vs VE/VO2 and time vs VE/VCO2
        Plot 3: time vs RER
        Plot 4: time vs PETO2 and time vs PETCO2

    Parameters
    ----------
    data_spiro : pd.DataFrame
        dataframe containing spirometer data
    power : pd.DataFrame
        dataframe containing the mean power output per step, showed as a continuous signal (see power_per_min)
    start_spiro : float
        start of maximal exercise test on spirometer
    muser : float
        mass user (kg)

    Returns
    -------
    fig : matplotlib.figure.Figure
        plots to determine vt1
    vt1 : pd.DataFrame
        main outcomes at vt1

    """

    fig, ax = plt.subplots(2, 2, figsize=[10, 7])
    fig.suptitle("Determination VT1", size=20)

    # Plot 5: HR versus VO2
    # VT1 if the inclination angle changes

    ax[0, 0].set_title("Plot - 5", fontweight="bold")
    ax[0, 0].scatter(
        data_spiro["VO2"].rolling(window=15).mean(), data_spiro["HR"].rolling(window=15).mean(), color="red", s=1
    )
    ax[0, 0].set_ylabel("HR [b/min]", {"color": "red"})
    ax[0, 0].set_ylim(0, 220)
    ax[0, 0].set_xlabel("VO2 [l/min]")
    ax1 = ax[0, 0].twinx()
    ax1.plot(
        data_spiro["VO2"].rolling(window=15).mean(), data_spiro["VCO2"].rolling(window=15).mean(), color="darkblue"
    )
    ax1.set_ylabel("VCO2 [l/min]", {"color": "darkblue"})

    # Plot 6: VE/VO2 & VE/VCO2 vs time
    # VT1 if VE/VO2 (blue line) switches from decrease to increase

    ax[0, 1].set_title("Plot - 6", fontweight="bold")
    ax[0, 1].plot(data_spiro["time"], data_spiro["VE/VO2"].rolling(window=15).mean(), color="darkblue")
    ax[0, 1].set_ylabel("VE/VO2", {"color": "darkblue"})
    ax[0, 1].set_ylim(0, 60)
    ax[0, 1].set_xlabel("time [s]")
    ax1 = ax[0, 1].twinx()
    ax1.plot(data_spiro["time"], data_spiro["VE/VCO2"].rolling(window=15).mean(), color="red")
    ax1.set_ylim(0, 60)
    ax1.set_ylabel("VE/VCO2", {"color": "red"})
    ax2 = ax[0, 1].twinx()
    ax2.spines["right"].set_position(("axes", 1.2))
    ax2.plot(power.index, power["power"], color="forestgreen", linestyle="dashed", alpha=0.1)
    ax2.fill_between(power.index, power["power"], color="forestgreen", alpha=0.1)
    ax2.set_ylabel("Power [W]", {"color": "forestgreen"})

    # Plot 8: RER vs time
    # VT1 should be RER < 1.0

    ax[1, 0].set_title("Plot - 8", fontweight="bold")
    ax[1, 0].plot(data_spiro["time"], data_spiro["RER"], color="lightblue")
    ax[1, 0].set_xlabel("time")
    ax[1, 0].set_ylabel("RER", {"color": "lightblue"})
    ax[1, 0].set_ylim(0, 1.6)
    ax[1, 0].hlines(1.0, data_spiro["time"][0], data_spiro["time"].iloc[-1], alpha=0.5, colors="k")

    # Plot 9: PetO2 & PetCO2 vs time
    # VT1 if PetO2 starts to increase

    if "PetO2" in list(data_spiro.columns):
        ax[1, 1].set_title("Plot - 9", fontweight="bold")
        ax[1, 1].plot(data_spiro["time"], data_spiro["PetO2"].rolling(window=15).mean(), color="darkblue")
        ax[1, 1].set_ylim(50, 130)
        ax[1, 1].set_ylabel("PetO2 [mmHg]", {"color": "darkblue"})
        ax[1, 1].set_xlabel("time")
        ax1 = ax[1, 1].twinx()
        ax1.plot(data_spiro["time"], data_spiro["PetCO2"].rolling(window=15).mean(), color="red")
        ax1.set_ylim(10, 90)
        ax1.set_ylabel("PetCO2 [mmHg]", {"color": "red"})
        ax2 = ax[1, 1].twinx()
        ax2.spines["right"].set_position(("axes", 1.2))
        ax2.plot(power.index, power["power"], color="forestgreen", linestyle="dashed", alpha=0.1)
        ax2.fill_between(power.index, power["power"], color="forestgreen", alpha=0.1)
        ax2.set_ylabel("Power [W]", {"color": "forestgreen"})

    plt.tight_layout()
    plt.subplots_adjust(top=0.90, hspace=0.5)

    # Click on the place where VT1 is, python plots a line at the same time point in all four plots.
    # If you are satisfied with the result --> y, and the outcomes are printed
    # If you are not satisfied with the result --> n, and you can place a new line

    while True:
        pts = plt.ginput(1)
        time = pts[0][0]
        if time > 10:
            time = find_nearest(data_spiro["time"], time)
            line1 = ax[0, 0].axvline(x=data_spiro["VO2"][data_spiro["time"] == time].values, color="k", linestyle="--")

        else:
            time = find_nearest(data_spiro["VO2"], time)
            time = data_spiro["time"][data_spiro["VO2"] == time].values[0]
            line1 = ax[0, 0].axvline(x=data_spiro["VO2"][data_spiro["time"] == time].values, color="k", linestyle="--")

        line2 = ax[0, 1].axvline(x=time, color="k", linestyle="--")
        line3 = ax[1, 0].axvline(x=time, color="k", linestyle="--")
        line4 = ax[1, 1].axvline(x=time, color="k", linestyle="--")
        plt.pause(0.05)

        root = tk.Tk()
        root.withdraw()
        agree = simpledialog.askstring(title="Evaluation", prompt="Are you satisfied (y/n)?:")
        if agree == "y":
            break
        if agree == "n":
            for line in [line1, line2, line3, line4]:
                line.set_color(color="grey")
                line.set_linewidth(0.9)

        else:
            print("Wrong input, expected y or n")
            break

    print(
        "VT1-time: " + str(time - int(start_spiro)) + " s"
        "\nVT1-power: " + str(round(power.iloc[time].values[0], 2)) + " W"
        "\nVT1-HR: " + str(data_spiro["HR"][data_spiro["time"] == time].values[0]) + " bpm"
        "\nVT1-VO2: "
        + str(round(data_spiro["VO2"].rolling(window=30).mean()[data_spiro["time"] == time].values[0], 3))
        + " l/min"
    )

    vt1 = pd.DataFrame(
        [
            {
                "VT1-time": time - int(start_spiro),
                "VT1-power": power.iloc[time].values[0],
                "VT1-HR": data_spiro["HR"][data_spiro["time"] == time].values[0],
                "VT1-VO2": data_spiro["VO2"].rolling(window=30).mean()[data_spiro["time"] == time].values[0],
                "VT1-VO2/kg": (
                    (data_spiro["VO2"].rolling(window=30).mean()[data_spiro["time"] == time].values[0]) * 1000
                )
                / muser,
            }
        ]
    )

    return fig, vt1


def anaerobic_threshold(data_spiro, power, start_spiro, muser):
    """
    Shows four plots to determine the anaerobic ventilatory threshold from the maximal exercise test
        Plot 1: time vs VE
        Plot 2: time vs VE/VO2 and time vs VE/VCO2
        Plot 3: time vs RER
        Plot 4: time vs PETO2 and time vs PETCO2

    Parameters
    ----------
    data_spiro : pd.DataFrame
        dataframe containing spirometer data
    power : pd.DataFrame
        dataframe containing the mean power output per step, showed as a continuous signal (see power_per_min)
    start_spiro : float
        start of maximal exercise test on spirometer
    muser : float
        mass user (kg)

    Returns
    -------
    fig : matplotlib.figure.Figure
        plots to determine vt2
    vt1 : pd.DataFrame
        main outcomes at vt2

    """

    fig, ax = plt.subplots(2, 2, figsize=[10, 7])
    fig.suptitle("Determination VT2", size=20)

    # Plot 1: VE vs time
    # VT2 occurs is there is a steeper increase in the VE

    ax[0, 0].set_title("Plot - 1", fontweight="bold")
    ax[0, 0].plot(data_spiro["time"], data_spiro["VE"].rolling(window=15).mean(), color="darkblue")
    ax[0, 0].set_xlabel("time [s]")
    ax[0, 0].set_ylabel("VE [l/min]", {"color": "darkblue"})
    vemax = data_spiro["VE"].rolling(window=30).mean().max()
    ax[0, 0].text(0.1, 0.9, transform=ax[0, 0].transAxes, s="VEmax: " + str(round(vemax, 1)), color="darkblue")
    ax1 = ax[0, 0].twinx()
    ax1.plot(power.index, power["power"], color="forestgreen", linestyle="dashed")
    ax1.fill_between(power.index, power["power"], color="forestgreen", alpha=0.15)
    ax1.set_ylabel("Power [W]", {"color": "forestgreen"})
    popeak = power.max()
    ax1.text(0.1, 0.8, transform=ax[0, 0].transAxes, s="POpeak: " + str(round(popeak[0], 1)), color="forestgreen")

    # Plot 6: VE/VO2 & VE/VCO2 vs time
    # VT2 if VE/VCO2 (red line) switches from decrease to increase

    ax[0, 1].set_title("Plot - 6", fontweight="bold")
    ax[0, 1].plot(data_spiro["time"], data_spiro["VE/VO2"].rolling(window=15).mean(), color="darkblue")
    ax[0, 1].set_ylabel("VE/VO2", {"color": "darkblue"})
    ax[0, 1].set_ylim(0, 60)
    ax[0, 1].set_xlabel("time [s]")
    ax1 = ax[0, 1].twinx()
    ax1.plot(data_spiro["time"], data_spiro["VE/VCO2"].rolling(window=15).mean(), color="red")
    ax1.set_ylim(0, 60)
    ax1.set_ylabel("VE/VCO2", {"color": "red"})
    ax2 = ax[0, 1].twinx()
    ax2.spines["right"].set_position(("axes", 1.2))
    ax2.plot(power.index, power["power"], color="forestgreen", linestyle="dashed", alpha=0.1)
    ax2.fill_between(power.index, power["power"], color="forestgreen", alpha=0.1)
    ax2.set_ylabel("Power [W]", {"color": "forestgreen"})

    # Plot 8: RER vs time
    # VT2 should have a higher RER than VT1

    ax[1, 0].set_title("Plot - 8", fontweight="bold")
    ax[1, 0].plot(data_spiro["time"], data_spiro["RER"], color="lightblue")
    ax[1, 0].set_xlabel("time")
    ax[1, 0].set_ylabel("RER", {"color": "lightblue"})
    ax[1, 0].set_ylim(0, 1.6)
    ax[1, 0].hlines(1.0, data_spiro["time"][0], data_spiro["time"].iloc[-1], alpha=0.5, colors="k")

    # Plot 9: PetO2 & PetCO2 vs time
    # VT2 if PetCO2 starts to decrease

    if "PetO2" in list(data_spiro.columns):
        ax[1, 1].set_title("Plot - 9", fontweight="bold")
        ax[1, 1].plot(data_spiro["time"], data_spiro["PetO2"].rolling(window=15).mean(), color="darkblue")
        ax[1, 1].set_ylim(50, 130)
        ax[1, 1].set_ylabel("PetO2 [mmHg]", {"color": "darkblue"})
        ax[1, 1].set_xlabel("time")
        ax1 = ax[1, 1].twinx()
        ax1.plot(data_spiro["time"], data_spiro["PetCO2"].rolling(window=15).mean(), color="red")
        ax1.set_ylim(10, 90)
        ax1.set_ylabel("PetCO2 [mmHg]", {"color": "red"})
        ax2 = ax[1, 1].twinx()
        ax2.spines["right"].set_position(("axes", 1.2))
        ax2.plot(power.index, power["power"], color="forestgreen", linestyle="dashed", alpha=0.1)
        ax2.fill_between(power.index, power["power"], color="forestgreen", alpha=0.1)
        ax2.set_ylabel("Power [W]", {"color": "forestgreen"})

    plt.tight_layout()
    plt.subplots_adjust(top=0.90, hspace=0.5)

    # Click on the place where VT2 is, python plots a line at the same time point in all four plots.
    # If you are satisfied with the result --> y, and the outcomes are printed
    # If you are not satisfied with the result --> n, and you can place a new line

    while True:
        pts = plt.ginput(1)
        time = pts[0][0]
        time = find_nearest(data_spiro["time"], time)
        line1 = ax[0, 0].axvline(x=time, color="k", linestyle="--")
        line2 = ax[0, 1].axvline(x=time, color="k", linestyle="--")
        line3 = ax[1, 0].axvline(x=time, color="k", linestyle="--")
        line4 = ax[1, 1].axvline(x=time, color="k", linestyle="--")
        plt.pause(0.05)

        root = tk.Tk()
        root.withdraw()
        agree = simpledialog.askstring(title="Evaluation", prompt="Are you satisfied (y/n)?:")
        if agree == "y":
            break
        if agree == "n":
            for line in [line1, line2, line3, line4]:
                line.set_color(color="grey")
                line.set_linewidth(0.9)
        else:
            print("Wrong input, expected y or n")
            break

    print(
        "VT2-time: " + str(time - int(start_spiro)) + " s"
        "\nVT2-power: " + str(round(power.iloc[time].values[0], 2)) + " W"
        "\nVT2-HR: " + str(data_spiro["HR"][data_spiro["time"] == time].values[0]) + " bpm"
        "\nVT2-VO2: "
        + str(round(data_spiro["VO2"].rolling(window=30).mean()[data_spiro["time"] == time].values[0], 3))
        + " l/min"
    )

    vt2 = pd.DataFrame(
        [
            {
                "VT2-time": time - int(start_spiro),
                "VT2-power": power.iloc[time].values[0],
                "VT2-HR": data_spiro["HR"][data_spiro["time"] == time].values[0],
                "VT2-VO2": data_spiro["VO2"].rolling(window=30).mean()[data_spiro["time"] == time].values[0],
                "VT2-VO2/kg": (
                    (data_spiro["VO2"].rolling(window=30).mean()[data_spiro["time"] == time].values[0]) * 1000
                )
                / muser,
            }
        ]
    )

    return fig, vt2
