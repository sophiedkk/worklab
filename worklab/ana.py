from scipy.signal import find_peaks
import itertools as it
import copy
import math
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import sklearn
import seaborn as sns

from .plots import plot_power_speed_dist
from .physio import calc_weighted_average


def mean_data(data):
    """
    Combined data of left and right module
    Time, speed, aspeed, acc and dist are averaged
    Force, torque, power, work and uforce are averaged and multiplied with two

    Parameters
    ----------
    data : dict
        processed ergometer data dictionary with dataframes for left and right

    Returns
    -------
    data : dict
        with left, right and mean module

    """
    data["mean"] = (data["left"] + data["right"]) / 2
    target_cols = ["force", "torque", "power", "work", "uforce"]
    data["mean"][target_cols] = data["mean"][target_cols] * 2

    return data


def cut_data(data, start, end, distance=True):
    """
    Cuts data to time of interest

    Parameters
    ----------
    data : dict
        processed ergometer data dictionary with dataframes
    start : float
        start time [s]
    end : float
        end time [s]
    distance : bool, optional
        resets distance to 0 at start, default is True.

    Returns
    -------
    data : dict
        data cutted to time of interest

    """
    for side in data:
        data[side] = data[side][(data[side]["time"] > start) & (data[side]["time"] < end)]
        data[side]["time"] = data[side]["time"] - start
        if distance:
            data[side]["dist"] -= data[side]["dist"].iloc[0]
        data[side].reset_index(drop=True, inplace=True)

    return data


def isometricforce(data, title=None, height=40, distance=500, ylim=None):
    """
    Calculates the three seconds maximal user force and plots it against time (darkblue).
    Peaks are annotated with a dot and with the height of the peak, max value is shown in the corner.
    Possibility to scale manually

    Parameters
    ----------
    data : dict
        processed ergometer data dictionary with dataframes
    title : str, optional
        plotted on top of graph, default is None
    height : float, optional
        minimal height of peak, default is 40 N
    distance : float, optional
        minimal distance between peaks, default is 500 samples
    ylim : list [min, max] of floats or int, optional
        list of the minimal and maximal ylim for user force in N

    Returns
    -------
    fig : matplotlib.figure.Figure
    peaks : series
        peaks of maximal user force (averaged over left and right)
    """
    # calculate 3s rolling average
    for side in data:
        data[side]["timed"] = pd.to_datetime(data[side]["time"], unit="s")
        data[side].set_index("timed", inplace=True)
        data[side]["3s"] = data[side]["uforce"].rolling(window="3s").mean()

    # plot force in blue and annotate peaks
    fig, ax = plt.subplots(3, figsize=[20, 9], sharex="all", sharey="all")
    idx = [0, 1, 2]

    peaks_y_all = []
    for side, x in zip(data, idx):
        peaks = find_peaks(data[side]["3s"], height=height, distance=distance)
        peaks_x = pd.Series(list(peaks[0]))
        peaks_y = pd.Series(list(peaks[1]["peak_heights"]))

        ax[x].plot(data[side]["time"], data[side]["3s"], label=side, color="mediumblue")
        ax[x].scatter(data[side]["time"][peaks_x], peaks_y, color="mediumblue")
        if ylim:
            ax[x].set_ylim(ylim[0], ylim[1])
        else:
            ax[x].set_ylim(-50, 1.3 * data["mean"]["uforce"].max())
        ax[x].xaxis.set_tick_params(labelsize=16)
        ax[x].yaxis.set_tick_params(labelsize=16)

        # annotate peaks in plot
        n = pd.Series(peaks_y)
        x_peaks = data[side]["time"][peaks_x].reset_index(drop=True)
        for i, txt in enumerate(round(n, 1)):
            ax[x].annotate(txt, (x_peaks[i], peaks_y[i]), fontsize=16)

        ax[x].text(
            0.85,
            0.07,
            transform=ax[x].transAxes,
            s="max = " + str(round(n.max(), 1)) + "N",
            bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.8),
            fontsize=16,
        )

        ax[x].legend(loc="upper left", fontsize=16, frameon=True)

        # style plot
        if x == 0:
            if title:
                ax[x].set_title(
                    "Isometric force production (3 seconds) for left, right and the average \n" + str(title),
                    fontsize=20)
            else:
                ax[x].set_title("Isometric force production (3 seconds) for left, right and the average", fontsize=20)
        if x == 1:
            ax[x].set_ylabel("Force [N]", fontsize=16)
            ax[x].yaxis.label.set_color("mediumblue")

        if x == 2:
            ax[x].set_xlabel("Time [s]", fontsize=16)

        peaks_y = pd.DataFrame(peaks_y, columns=[side])
        peaks_y_all.append(peaks_y)

    peaks = pd.DataFrame()
    peaks["left"] = peaks_y_all[0]
    peaks["right"] = peaks_y_all[1]
    peaks["mean"] = peaks_y_all[2]
    return fig, peaks


def protocol_wingate(fiso, muser, mwc, v=2):
    """
    Calculates the protocol for the Wingate test on a wheelchair ergometer,
    based on the regression equations between the isometric force, anaerobic
    and aerobic power.

    (Janssen T.W.J., Van Oers C.A.J.M., Hollander A.P., Veeger H.E.J., Van der Woude L.H.V.
    Isometric strength sprint power and anaerobic power in individuals with a
    spinal cord injury. Med Sci Sports Exercise 1993;25(7):863-870.
    doi:10.1249/00005768-199307000-00016)

    Parameters
    ----------
    fiso : float
        maximal 3 seconds force in N, average of left and right
    muser : float/int
        mass user
    mwc : float/int
        mass wheelchair
    v : float/int, optional
        mean velocity wingate, default is 2 m/s

    Returns
    -------
    Print the maximal three seconds force, the predicted p30, the aimed mean velocity
    and the calculated resistance.

    """
    mtotal = muser + mwc

    fisokg = fiso / muser
    p30 = 0.51 * fisokg - 0.18
    ptotal = p30 * muser
    f = ptotal / v
    mu = f / (mtotal * 9.81)

    protocol = (
        "-" * 60
        + "\n The maximal three second force is: "
        + str(round(fiso, 2))
        + " N"
        + "\n The predicted P30 = "
        + str(round(ptotal, 2))
        + "\n Aimed is for an average velocity of "
        + str(v)
        + " m/s"
        + "\n The calculated resistance for the Wingate test is: "
        + str(round(mu, 3))
        + " mu"
        + "\n"
        + 60 * "-"
    )

    print(protocol)


def wingate(data, title=None, box=False, ylim=5):
    """
    Wingate test analyse. Gives a plot with the power (green) and velocity (red),
    also prints the important performance indicators

    Parameters
    ----------
    data : dict
        processed and cutted ergometer data dictionary with dataframes
    title : str
        title of figure
    box : bool
        prints important performance indicators on figure, default is False
    ylim : float, optional
        sets the ylim of the graph, default is 5 ms

    Returns
    -------
    fig : matplotlib.figure.Figure
    outcomes : dataframe

    """
    # rolling average over 5 seconds
    for side in data:
        data[side]["p5"] = data[side]["power"].rolling(window=500).mean()

    # plot figure with power and velocity
    fig, ax = plt.subplots(figsize=[15, 15])
    ax.plot(data["mean"]["time"], data["mean"]["power"], label="power", color="forestgreen")
    ax.set_ylabel("Power [W]", fontsize=18)
    ax.set_ylim(0, 1.1 * data["mean"]["power"].max())
    ax.set_xlabel("time [s]", fontsize=18)
    ax.legend(loc="upper left", fontsize=14)
    ax.yaxis.label.set_color("forestgreen")
    ax.tick_params(axis="x", labelsize=14)
    ax.tick_params(axis="y", labelsize=14)

    ax1 = ax.twinx()
    ax1.plot(data["mean"]["time"], data["mean"]["speed"], color="firebrick", label="speed", alpha=0.5)
    ax1.set_ylabel("speed [m/s]", fontsize=18)
    ax1.yaxis.label.set_color("firebrick")
    ax1.set_ylim(0, ylim)
    ax1.legend(fontsize=14)
    ax1.tick_params(axis="y", labelsize=14)

    # calculate outcomes
    p30 = data["mean"]["power"].mean()
    pmax = data["mean"]["power"].max()
    p5max = data["mean"]["p5"].max()
    p5start = data["mean"]["p5"].iloc[499]
    p5min = data["mean"]["p5"].min()
    p5end = data["mean"]["p5"].iloc[-1]
    rfmaxmin = ((p5max - p5min) / p5max) * 100
    rfstartend = ((p5start - p5end) / p5start) * 100
    vmean = data["mean"]["speed"].mean()
    vmax = data["mean"]["speed"].max()

    outcomes = [
        {
            "P30": p30,
            "Pmax": pmax,
            "P5max": p5max,
            "P5min": p5min,
            "rfmaxmin": rfmaxmin,
            "P5start": p5start,
            "P5end": p5end,
            "rfstartend": rfstartend,
            "vmean": vmean,
            "vmax": vmax,
        }
    ]

    outcomes = pd.DataFrame(outcomes)
    if box:
        # annotate important outcomes in figure
        ax.text(
            0.80,
            0.1,
            transform=ax.transAxes,
            s="P30 = "
            + str(round(p30, 0))
            + " W"
            + "\nPmax = "
            + str(round(pmax, 0))
            + "W"
            + "\nP5max = "
            + str(round(p5max, 2))
            + " W"
            + "\nvmax = "
            + str(round(vmax, 1))
            + " m/s"
            + "\nRF = "
            + str(round(rfstartend, 1))
            + " %",
            bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.9),
            fontsize=14,
        )

    if title:
        ax.set_title("Wingate test \n" + str(title), fontsize=24)
    else:
        ax.set_title("Wingate test", fontsize=24)

    return fig, outcomes


def protocol_max(p30, muser, mwc, v=1.39):
    """
    Calculates the protocol for the Maximal exercise test on a wheelchair ergometer,
    based on the regression equations between the isometric force, anaerobic
    and aerobic power.

    (Janssen T.W.J., Van Oers C.A.J.M., Hollander A.P., Veeger H.E.J., Van der Woude L.H.V.
    Isometric strength sprint power and anaerobic power in individuals with a
    spinal cord injury. Med Sci Sports Exercise 1993;25(7):863-870.
    doi:10.1249/00005768-199307000-00016)

    Parameters
    ----------
    p30 : float
        average power over a 30-sec Wingate test
    muser : float/int
        mass user
    mwc : float/int
        mass wheelchair
    v : float, optional
        constant comfortable velocity for the test, default is 1.39 m/s

    Returns
    -------
    Print the p30, the popeak, the aimed mean velocity and the resistance for each step.

    """
    mtotal = muser + mwc

    p30kg = p30 / muser

    poaer = 0.67 * p30kg + 0.11  # (Janssen)
    pototal = poaer * muser
    postart = 0.20 * pototal  # start at 20% of PO-peak
    posubmax2 = 0.40 * pototal  # 40% of PO-peak

    # calculate resistance start, submax2 & end
    f0 = postart / v
    mu0 = f0 / (mtotal * 9.81)  # resistance start
    f1 = posubmax2 / v
    mu1 = f1 / (mtotal * 9.81)  # resistance submax2
    fend = pototal / v
    muend = fend / (mtotal * 9.81)  # resistance end
    diff = muend - mu0

    protocol = (
        "-" * 60
        + "\n P30 = "
        + str(round(p30, 3))
        + "\n Predicted PO-peak = "
        + str(round(pototal, 3))
        + "\n Sub-maximal 20%PO-peak = "
        + str(round(mu0, 4))
        + " mu"
        + " and velocity is "
        + str(round(v, 2))
        + " m/s"
        "\n Sub-maximal 40%PO-peak = " + str(round(mu1, 4)) + " mu" + " and velocity is " + str(round(v, 2)) + " m/s"
        "\n-\n The start resistance for the maximal exercise test should be " + str(round(mu0, 4)) + " mu"
        "\n After 10 minutes the resistance should be " + str(round(muend, 4)) + " mu"
        "\n The step length is thus " + str(round((1 / 9) * diff, 4)) + " mu"
        "\n Velocity is always " + str(round(v, 2)) + " m/s"
        "\n Make sure to prolong the protocol for 25 minutes." + "\n" + 60 * "-"
    )

    print(protocol)


def maximal1min(data, dur, title=None):
    """
    Maximal exercise test analyse. Gives a plot with the power (green) and velocity (red)
    for each step, also prints the important performance indicators per step:
        Work [J]
        Mean power [W]
        Maximal power [W]
        Mean velocity [ms]

    Parameters
    ----------
    data : dict
        processed and cutted ergometer data dictionary with dataframes
    dur : int
        duration of max test in seconds
    title : str, optional
        title of figure

    Returns
    -------
    fig : matplotlib.figure.Figure
    outcomes : dataframe

    """
    # plot figure with 4 columns and x rows (depending on duration test)
    n = [*range(math.ceil(dur / 60))]
    ncolumns = 4  # columns in the figure
    nrows = math.ceil(((max(n) + 1) / ncolumns))  # rows in the figure

    fig, ax = plt.subplots(nrows, ncolumns, sharey="all", figsize=(20, 16))
    if title:
        plt.suptitle(
            "Analysis of maximal exercise test for: "
            + str(title)
            + "\nIncrements = 1 min, last 20sec of each minute shown"
        )
    else:
        plt.suptitle("Analysis of maximal exercise test"
                     + "\nIncrements = 1 min, last 20sec of each minute shown")
    rows = []
    for i in list(range(0, 5)):
        x = list(it.repeat(i, 4))
        rows = rows + x  # rows in the figure

    columns = list(range(0, 4)) * 7  # columns in the figure

    # variables of interest for each step
    mean_power = []
    max_power = []
    mean_vel = []
    work = []

    for i, r, c in zip(n, rows, columns):
        # slice mean data for each step
        x = copy.deepcopy(data["mean"])
        s = x[(x["time"] > ((i + 1) * 60) - 60) & (x["time"] < ((i + 1) * 60))]
        s = s[s["time"] > (s["time"].max() - 20)]

        # calculate variables for each step (last 20 seconds)
        mean_p = s["power"].mean()
        mean_power.append(mean_p)
        max_p = s["power"].max()
        max_power.append(max_p)
        mean_v = s["speed"].mean()
        mean_vel.append(mean_v)
        w = s["work"].sum()
        work.append(w)

        # plot power versus time and velocity on the second y-axis
        ax[r, c].plot(s["time"], s["power"], color="forestgreen")
        ax1 = ax[r, c].twinx()
        ax1.plot(s["time"], s["speed"], color="firebrick", alpha=0.5)
        ax1.set_ylim(-1, 1.2 * data["mean"]["speed"].max())

        # set title and box with mean power per step
        ax[r, c].set_title("Step " + str(i + 1), fontweight="bold")
        ax[r, c].text(
            0.50,
            0.05,
            transform=ax[r, c].transAxes,
            s="mean_power = " + str(round(mean_p, 1)),
            bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.8),
        )

    plt.subplots_adjust(top=0.90, hspace=0.4)

    step = pd.DataFrame(n, columns=["step"]).T + 1
    work = pd.DataFrame(work, columns=["work"]).T
    mean_power = pd.DataFrame(mean_power, columns=["mean_power"]).T
    max_power = pd.DataFrame(max_power, columns=["max_power"]).T
    mean_vel = pd.DataFrame(mean_vel, columns=["mean_vel"]).T

    outcomes = pd.concat([step, work, mean_power, max_power, mean_vel])
    outcomes = outcomes.T
    return fig, outcomes


def ana_sprint(data, data_pbp, half=5, title=None):
    """
    Sprint test analyse. Plot a figure with the power, speed and distance for
    left and right separate. Also saves important outcomes

    Parameters
    ----------
    data : dict
        processed and cutted ergometer data dictionary with dataframes
    data_pbp : dict
        processed and cutted push_by_push ergometer data dictionary with dataframes
    half : float, optional
        half-time of the sprint, default is 5 s
    title : str, optional
        title of figure

    Returns
    -------
    fig : matplotlib.figure.Figure
    outcomes : dataframe

    """
    if title:
        fig = plot_power_speed_dist(data, title)
    else:
        fig = plot_power_speed_dist(data, title=" ")

    half = half * 100  # to get right index
    outcomes = [
        {
            "distance_half": round(data["mean"]["dist"].iloc[half], 1),
            "distance_half_l": round(data["left"]["dist"].iloc[half], 1),
            "distance_half_r": round(data["right"]["dist"].iloc[half], 1),
            "distance": round(data["mean"]["dist"].max(), 1),
            "distance_l": round(data["left"]["dist"].max(), 1),
            "distance_r": round(data["right"]["dist"].max(), 1),
            "max_vel": round(data["mean"]["speed"].max(), 1),
            "max_vel_l": round(data["left"]["speed"].max(), 1),
            "max_vel_r": round(data["right"]["speed"].max(), 1),
            "mean_vel": round(data["mean"]["speed"].mean(), 1),
            "mean_vel_l": round(data["left"]["speed"].mean(), 1),
            "mean_vel_r": round(data["right"]["speed"].mean(), 1),
            "max_power": round(data["mean"]["power"].max(), 0),
            "max_power_l": round(data["left"]["power"].max(), 0),
            "max_power_r": round(data["right"]["power"].max(), 0),
            "mean_power": round(data["mean"]["power"].mean(), 0),
            "mean_power_l": round(data["left"]["power"].mean(), 0),
            "mean_power_r": round(data["right"]["power"].mean(), 0),
            "maxpowerafter3": round(data_pbp["mean"]["maxpower"][0:3].max(), 0),
            "maxpowerafter3_l": round(data_pbp["left"]["maxpower"][0:3].max(), 0),
            "maxpowerafter3_r": round(data_pbp["right"]["maxpower"][0:3].max(), 0),
            "maxvelafter3": round(data_pbp["mean"]["maxspeed"][0:3].max(), 1),
            "maxvelafter3_l": round(data_pbp["left"]["maxspeed"][0:3].max(), 1),
            "maxvelafter3_r": round(data_pbp["right"]["maxspeed"][0:3].max(), 1),
            "ctime": round(data_pbp["mean"]["ctime"].mean(), 2),
            "p_time": round(data_pbp["mean"]["ptime"].mean(), 2),
        }
    ]

    outcomes = pd.DataFrame(outcomes)

    return fig, outcomes


def ana_submax(data_ergo, data_pbp, data_spiro):
    """
    Sub maximal test analyse. Saves important outcomes

    Parameters
    ----------
    data_ergo : pd.DataFrame
        processed and cutted ergometer data
    data_pbp : pd.DataFrame
        processed and cutted ergometer data
    data_spiro : pd.DataFrame
        processed and cutted spirometer data

    Returns
    -------
    outcomes : pd.DataFrame

    """
    mean_spiro = calc_weighted_average(data_spiro[["RER", "EE", "HR", "VO2"]], data_spiro["weights"])
    mean_spiro = pd.DataFrame(mean_spiro).T

    mean_ergo = [
        {
            "mean_vel": data_ergo["mean"]["speed"].mean(),
            "meanpower": data_ergo["mean"]["power"].mean(),
            "ptime_l": data_pbp["left"]["ptime"].mean(),
            "ptime_r": data_pbp["right"]["ptime"].mean(),
            "ctime_l": data_pbp["left"]["ctime"].mean(),
            "ctime_r": data_pbp["right"]["ctime"].mean(),
            "ca_l": data_pbp["left"]["cangle_deg"].mean(),
            "ca_r": data_pbp["right"]["cangle_deg"].mean(),
            "meanpowerperpush": data_pbp["mean"]["meanpower"].mean(),
            "maxpowerperpush": data_pbp["mean"]["maxpower"].mean(),
            "angle_deg": data_pbp["mean"]["cangle_deg"].mean(),
            "slope": data_pbp["mean"]["slope"].mean(),
            "smoothness": data_pbp["mean"]["smoothness"].mean(),
            "freq": 1 / data_pbp["mean"]["ctime"].mean(),
        }
    ]

    mean_ergo = pd.DataFrame(mean_ergo)

    outcomes = pd.concat([mean_ergo, mean_spiro], axis=1)
    outcomes["me"] = (outcomes["meanpower"] / outcomes["EE"]) * 100
    return outcomes


def force_velocity_curve(data_pbp, upper_lim=800, var='max'):
    """
    Creates force-velocity curves for wheelchair sports

    Parameters
    ----------
    data_pbp : pd.DataFrame
        processed push-by-push ergometer dataframe with output for all 6 sprints
    upper_lim : int
        upper limit recommendations for LP (800) and HP (1400)
    var : str
        'max' force and velocity or 'mean' force and velocity

    Returns
    -------
    fig : figure
        force-velocity plot
    variables : pd.DataFrame
        r2, optimal velocity/power, x/y coordinates and coefficient

    """
    if var == 'max':
        speed = 'maxspeed'
        force = 'maxuforce'
    else:
        speed = 'meanspeed'
        force = 'meanuforce'

    data_pbp = data_pbp[data_pbp.index > 0]
    x = np.array(data_pbp[speed]).reshape((-1, 1))
    y = np.array(data_pbp[force])
    data_pbp['x'] = np.array(data_pbp[speed]).reshape((-1, 1))
    data_pbp['y'] = np.array(data_pbp[force])
    model = sklearn.LinearRegression()
    model.fit(x, y)
    model = sklearn.LinearRegression().fit(x, y)
    r_sq = model.score(x, y)
    x1 = np.linspace(0, float(abs(model.intercept_ / model.coef_)), 100)
    xx = np.linspace(x.min(), x.max(), 100)

    pred_y = model.intercept_ + model.coef_ * x
    pred_y1 = model.intercept_ + model.coef_ * x1
    pred_y2 = model.intercept_ + model.coef_ * xx
    power = xx * pred_y2
    power1 = x1 * pred_y1
    parabola = pd.DataFrame({'POmax': power1, 'vmax': x1})
    pomax_pos = parabola['POmax'].idxmax()
    pomax_opt = parabola['POmax'].max()
    vmax_opt = parabola['vmax'][pomax_pos]

    variables = pd.DataFrame([])
    variables['R2'] = [r_sq]
    variables['opt_vel'] = vmax_opt
    variables['opt_pow'] = pomax_opt
    variables['y_cor'] = model.intercept_
    variables['x_cor'] = x1[-1]
    variables['coef'] = model.coef_
    variables = round(variables, 2)

    sns.set_style('darkgrid')
    col_pal = sns.color_palette("dark:#5A9_r")
    sns.set_palette(col_pal)
    fig, ax = plt.subplots(1, 1, figsize=(14, 8))
    ax.set_ylabel('Force [N]', fontsize=14)
    ax.set_ylim(0, upper_lim)
    ax.set_xlabel('Velocity [ms]', fontsize=14)
    ax.set_xlim(0, 6)
    ax.tick_params(axis='both', labelsize=12)
    ax = sns.scatterplot(data=data_pbp, x='x', y='y', hue='Resistance')
    ax.plot(x1, pred_y1, color='k', linestyle='--')
    ax.plot(x, pred_y, color='k')
    ax.annotate('R2 = ' + str(round(r_sq, 2)), xy=(0.75, 0.90), xycoords='axes fraction')
    ax.annotate('y = ' + str(round(model.intercept_, 1)) + ' ' + str(round(model.coef_[0], 1)) + ' * x',
                xy=(0.75, 0.85), xycoords='axes fraction')

    ax1 = ax.twinx()
    ax1.grid(False)
    ax1.plot(x1, power1, color='grey', linestyle='--')
    ax1.plot(xx, power, color='grey')
    ax1.set_ylim(0, upper_lim)
    ax1.set_ylabel('Power [W]', color='grey', fontsize=14)
    ax1.yaxis.label.set_color('grey')
    ax1.spines['right'].set_color('grey')
    ax1.tick_params(axis='y', colors='grey')
    ax1.tick_params(axis='both', labelsize=12)
    ax1.annotate('Optimal velocity (' + str(round(vmax_opt, 1)) + ' ms)', xy=(0.75, 0.95), xycoords='axes fraction')

    return fig, variables
