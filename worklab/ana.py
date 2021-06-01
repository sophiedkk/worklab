from scipy.signal import find_peaks
import itertools as it
import copy
import math
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import sys

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
    data['mean'] = (data['left'] + data['right'])/2
    target_cols = ['force', 'torque', 'power', 'work', 'uforce']
    data['mean'][target_cols] = data['mean'][target_cols] * 2

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
        data[side] = data[side][(data[side]['time'] > start) & (data[side]['time'] < end)]
        data[side]['time'] = data[side]['time'] - start
        if distance:
            data[side]['dist'] = data[side]['dist'] - data[side]['dist'].iloc[0]
        data[side].reset_index(drop=True, inplace=True)

    return data


def isometricforce(data, title=None, height=40, distance=100, vel=True, ylim=None):
    """
    Calculates the three seconds maximal user force and plots it against time (darkblue).
    Peaks are annotated with a dot and with the height of the peak, max value is shown in the corner.
    Plot velocity optional & possibility to scale manually

    Parameters
    ----------
    data : dict
        processed ergometer data dictionary with dataframes
    title : str, optional
        plotted on top of graph, default is None
    height : float, optional
        minimal height of peak, default is 40 N
    distance : float, optional
        minimal distance between peaks, default is 100 samples
    vel : bool
        plot velocity, default is True
    ylim : list [min, max] of floats or int, optional
        list of the minimal and maximal ylim for user force in N

    Returns
    -------
    fig : matplotlib.figure.Figure
    peaks_y : series
        peaks of maximal user force (averaged over left and right)
    """
    # calculate 3s rolling average
    for side in data:
        data[side]['3s'] = data[side]['uforce'].rolling(window=300).mean()

    # plot figure, force in blue en velocity (optional) in red
    fig, ax = plt.subplots(3, figsize=[20, 9], sharex=True, sharey=True)
    idx = [0, 1, 2]

    for side, x in zip(data, idx):
        peaks = find_peaks(data[side]['3s'], height=height, distance=distance)
        peaks_x = pd.Series(list(peaks[0]))
        peaks_y = pd.Series(list(peaks[1]['peak_heights']))

        ax[x].plot(data[side]['time'], data[side]['3s'], label=side, color='mediumblue')
        ax[x].scatter(data[side]['time'][peaks_x], peaks_y, color='mediumblue')
        if ylim:
            ax[x].set_ylim(ylim[0], ylim[1])
        if not ylim:
            ax[x].set_ylim(-50, 1.3 * data['mean']['uforce'].max())
        ax[x].xaxis.set_tick_params(labelsize=16)
        ax[x].yaxis.set_tick_params(labelsize=16)

        if vel:
            ax1 = ax[x].twinx()
            ax1.plot(data[side]['time'], data[side]['speed'], label=side, color='firebrick', alpha=0.4)
            ax1.set_ylim(-0.1, 0.5)
            ax1.yaxis.set_tick_params(labelsize=16)

        # annotate peaks in plot
        n = pd.Series(peaks_y)
        x_peaks = data[side]['time'][peaks_x].reset_index(drop=True)
        for i, txt in enumerate(round(n, 1)):
            ax[x].annotate(txt, (x_peaks[i], peaks_y[i]), fontsize=16)

        ax[x].text(0.85, 0.07, transform=ax[x].transAxes, s='max = '+str(round(n.max(), 1))+'N',
                   bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8), fontsize=16)

        ax[x].legend(loc='upper left', fontsize=16, frameon=True)

        # style plot
        if x == 0:
            if title:
                ax[x].set_title('Isometric force production (3 seconds) for left, right and the average \n'+str(title), fontsize=20)
            if not title:
                ax[x].set_title('Isometric force production (3 seconds) for left, right and the average', fontsize=20)
        if x == 1:
            ax[x].set_ylabel('Force [N]', fontsize=16)
            ax[x].yaxis.label.set_color('mediumblue')
            if vel:
                ax1.set_ylabel('Velocity [m/s]', fontsize=16)
                ax1.yaxis.label.set_color('firebrick')

        if x == 2:
            ax[x].set_xlabel('Time [s]', fontsize=16)

        peaks_y = pd.DataFrame(peaks_y, columns=['max_user_force'])
    return fig, peaks_y


def protocol_wingate(fiso, muser, mwc, folder=None, v=2):
    """
    Calculates the protocol for the Wingate test on a wheelchair ergometer,
    based on the regression equations between the isometric force, anaerobic
    and aerobic power.

    (Janssen TWJ, Van Oers CAJM, Hollander AP, Veeger HEJ, Van der Woude LHV.
    Isometric strength sprint power and anaerobic power in individuals with a
    spinal cord injury. Med Sci Sports Exerc. 1993;25(7):863-870.
    doi:10.1249/00005768-199307000-00016)

    Parameters
    ----------
    fiso : float
        maximal 3 seconds force in N, average of left and right
    muser : float/int
        mass user
    mwc : float/int
        mass wheelchair
    folder : str, optional
        file path, protocol will be saved here
    v : float/int, optional
        mean velocity wingate, default is 2 m/s

    Returns
    -------
    Print the maximal three seconds force, the predicted P30, the aimed mean velocity
    and the calculated resistance. Option to save in folder

    """
    mtotal = muser + mwc

    fisokg = fiso / muser
    p30 = 0.51 * fisokg - 0.18
    ptotal = p30 * muser
    f = ptotal / v
    mu = f / (mtotal * 9.81)

    protocol = '-' * 60 + \
        '\n The maximal three second force is: ' + str(round(fiso, 2)) + ' N' + \
        '\n The predicted P30 = ' + str(round(ptotal, 2)) + \
        '\n Aimed is for an average velocity of ' + str(v) + ' m/s' + \
        '\n The calculated resistance for the Wingate test is: ' + str(round(mu, 3)) + ' mu' + \
        '\n' + 60 * '-'

    print(protocol)

    # save print in folder (optional)
    if folder:
        original_stdout = sys.stdout  # original standard output
        with open(folder+'//'+'isometricwingate.txt', 'a') as f:
            sys.stdout = f  # change the standard output to the file we created
            print(protocol)
            sys.stdout = original_stdout  # reset the standard output


def wingate(data, title=None):
    """
    Wingate test analyse. Gives a plot with the power (green) and velocity (red),
    also prints the important performance indicators

    Parameters
    ----------
    data : dict
        processed and cutted ergometer data dictionary with dataframes
    title : str
        title of figure

    Returns
    -------
    fig : matplotlib.figure.Figure
    outcomes : dataframe

    """
    # rolling average over 5 seconds
    for side in data:
        data[side]['p5'] = data[side]['power'].rolling(window=500).mean()

    # plot figure with power and velocity
    fig, ax = plt.subplots(figsize=[15, 15])
    ax.plot(data['mean']['time'], data['mean']['power'], label='power', color='forestgreen')
    ax.set_ylabel('Power [W]', fontsize=18)
    ax.set_ylim(0, 1.1 * data['mean']['power'].max())
    ax.set_xlabel('time [s]', fontsize=18)
    ax.legend(loc='upper left', fontsize=14)
    ax.yaxis.label.set_color('forestgreen')
    ax.tick_params(axis="x", labelsize=14)
    ax.tick_params(axis="y", labelsize=14)

    ax1 = ax.twinx()
    ax1.plot(data['mean']['time'], data['mean']['speed'], color='firebrick', label='speed', alpha=.5)
    ax1.set_ylabel('speed [m/s]', fontsize=18)
    ax1.yaxis.label.set_color('firebrick')
    ax1.set_ylim(0, 5)
    ax1.legend(fontsize=14)
    ax1.tick_params(axis="y", labelsize=14)

    # calculate outcomes
    P30 = data['mean']['power'].mean()
    Pmax = data['mean']['power'].max()
    P5max = data['mean']['p5'].max()
    P5start = data['mean']['p5'].iloc[499]
    P5min = data['mean']['p5'].min()
    P5end = data['mean']['p5'].iloc[-1]
    rfmaxmin = ((P5max - P5min)/P5max) * 100
    rfstartend = ((P5start - P5end)/P5start) * 100
    vmean = data['mean']['speed'].mean()
    vmax = data['mean']['speed'].max()

    outcomes = [{"P30": P30, "Pmax": Pmax, "P5max": P5max, "P5min": P5min,
                 "rfmaxmin": rfmaxmin, "P5start": P5start, "P5end": P5end,
                 "rfstartend": rfstartend, "vmean": vmean, "vmax": vmax}]

    outcomes = pd.DataFrame(outcomes)

    # annotate important outcomes in figure
    ax.text(0.80, 0.1, transform=ax.transAxes, s='P30 = ' + str(round(P30, 0)) + ' W' +
            '\nPmax = ' + str(round(Pmax, 0)) + 'W' +
            '\nP5max = ' + str(round(P5max, 2)) + ' W' +
            '\nvmax = ' + str(round(vmax, 1)) + ' m/s' +
            '\nRF = ' + str(round(rfstartend, 1)) + ' %'
            , bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.9), fontsize=14)

    if title:
        ax.set_title('Wingate test \n' + str(title), fontsize=24)
    if not title:
        ax.set_title('Wingate test', fontsize=24)

    return fig, outcomes


def protocol_max(P30, muser, mwc, folder=None, v=1.39, protocol="J"):
    """
    Calculates the protocol for the Maximal exercise test on a wheelchair ergometer,
    based on the regression equations between the isometric force, anaerobic
    and aerobic power.

    (Janssen TWJ, Van Oers CAJM, Hollander AP, Veeger HEJ, Van der Woude LHV.
    Isometric strength sprint power and anaerobic power in individuals with a
    spinal cord injury. Med Sci Sports Exerc. 1993;25(7):863-870.
    doi:10.1249/00005768-199307000-00016)

    Parameters
    ----------
    P30 : float
        average power over a 30-sec Wingate test
    muser : float/int
        mass user
    mwc : float/int
        mass wheelchair
    folder : str, optional
        file path, protocol will be saved here
    v : float, optional
        constant comfortable velocity for the test, default is 1.39 m/s
    protocol : str, optional
        "J" is Janssen protocol, "W" is van der Woude protocol, default is "J"

    Returns
    -------
    Print the P30, the POpeak, the aimed mean velocity and the resistance for each step,
    option to save in folder

    """
    mtotal = muser + mwc

    p30kg = P30/muser
    if protocol == "J":
        poaer = 0.67 * p30kg + 0.11  # (Janssen)
    if protocol == "W":
        poaer = 0.75 * p30kg + 0.03  # (Woude)
    pototal = poaer * muser
    postart = 0.20 * pototal  # start at 20% of POpeak
    posubmax2 = 0.40 * pototal  # 40% of POpeak

    # calculate resistance start, submax2 & end
    f0 = postart / v
    mu0 = f0 / (mtotal * 9.81)  # resistance start
    f1 = posubmax2 / v
    mu1 = f1 / (mtotal * 9.81)  # resistance submax2
    fend = pototal / v
    muend = fend / (mtotal * 9.81)  # resistance end
    diff = muend - mu0

    protocol = '-' * 60 + '\n P30 = ' + str(round(P30, 3)) + \
        '\n Predicted POpeak = ' + str(round(pototal, 3)) + \
        '\n Submaximal 20%POpeak = ' + str(round(mu0, 4)) + ' mu' + ' and velocity is ' + str(round(v, 2)) + ' m/s' \
        '\n Submaximal 40%POpeak = ' + str(round(mu1, 4)) + ' mu' + ' and velocity is ' + str(round(v, 2)) + ' m/s' \
        '\n-\n The start resistance for the maximal exercise test should be ' + str(round(mu0, 4)) + ' mu'\
        '\n After 10 minutes the resistance should be ' + str(round(muend, 4)) + ' mu'\
        '\n The step length is thus ' + str(round((1/9) * diff, 4)) + ' mu'\
        '\n Velocity is always ' + str(round(v, 2)) + ' m/s'\
        '\n Make sure to prolong the protocol for 25 minutes.' +\
        '\n' + 60 * '-'

    print(protocol)

    # option to save print in folder
    if folder:
        original_stdout = sys.stdout  # original standard output
        with open(folder + '//' + 'wingatemax.txt', 'a') as f:
            sys.stdout = f  # change the standard output to the file we created.
            print(protocol)
            sys.stdout = original_stdout  # reset the standard output


def maximal1min(data, data_pbp, dur, title=None):
    """
    Maximal exercise test analyse. Gives a plot with the power (green) and velocity (red)
    for each step, also prints the important performance indicators per step

    Parameters
    ----------
    data : dict
        processed and cutted ergometer data dictionary with dataframes
    data_pbp : dict
        processed and cutted push_by_push ergometer data dictionary with dataframes
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
    n = [*range(math.ceil((dur) / 60))]
    ncolumns = 4  # columns in the figure
    nrows = math.ceil(((max(n)+1)/ncolumns))  # rows in the figure

    fig, ax = plt.subplots(nrows, ncolumns, sharey=True, figsize=(20, 16))
    if title:
        plt.suptitle('Analysis of maximal exercise test for: ' + str(title) +
                     '\nIncrements = 1 min, last 20sec of each minute shown')
    if not title:
        plt.suptitle('Analysis of maximal exercise test' +
                     '\nIncrements = 1 min, last 20sec of each minute shown')

    for i in list(range(0, 5)):
        x = list(it.repeat(i, 4))
        if i == 0: rows = []
        rows = rows + x  # rows in the figure

    columns = list(range(0, 4)) * 7  # columns in the figure

    # variables of interest for each step
    mean_power = []; max_power = []; mean_vel = []; work = []; push_time = []; cycle_time = []

    for i, r, c in zip(n, rows, columns):
        # slice mean data for each step
        x = copy.deepcopy(data['mean'])
        s = x[(x['time'] > ((i+1)*60)-60) & (x['time'] < (((i+1)*60)))]
        s = s[s['time'] > (s['time'].max()-20)]

        # slice push by push data for each step
        z = copy.deepcopy(data_pbp['mean'])
        p = z[(z['tstart'] > ((i+1)*60)-60) & (z['tstop'] < (((i+1)*60)))]
        p = p[p['tstart'] > (s['time'].max()-20)]

        # calculate variables for each step (last 20 seconds)
        mean_p = s['power'].mean();    mean_power.append(mean_p)
        max_p = s['power'].max();    max_power.append(max_p)
        mean_v = s['speed'].mean();    mean_vel.append(mean_v)
        w = s['work'].sum();    work.append(w)
        push = p['ptime'].mean();    push_time.append(push)
        cycle = p['ctime'].mean();    cycle_time.append(cycle)

        # plot power versus time and the start and stop of each step
        ax[r, c].plot(s['time'], s['power'], color='forestgreen')
        ax[r, c].plot(s["time"][p["start"]], s['power'][p["start"]], 'ok', alpha=0.7, markersize=4)
        ax[r, c].plot(s["time"][p["stop"]], s['power'][p["stop"]], "ok", alpha=0.7, markersize=4)

        # plot velocity on the second y axis
        ax1 = ax[r, c].twinx()
        ax1.plot(s['time'], s['speed'], color='firebrick', alpha=0.5)
        ax1.set_ylim(-1, 2)

        # set title and box with mean power per step
        ax[r, c].set_title("Step " + str(i + 1), fontweight="bold")
        ax[r, c].text(0.50, 0.05, transform=ax[r, c].transAxes,
                      s='mean_power = ' + str(round(mean_p, 1)),
                      bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

    plt.subplots_adjust(top=0.90, hspace=0.4)

    step = (pd.DataFrame(n, columns=['step']).T) + 1
    work = pd.DataFrame(work, columns=['work']).T
    mean_power = pd.DataFrame(mean_power, columns=['mean_power']).T
    max_power = pd.DataFrame(max_power, columns=['max_power']).T
    mean_vel = pd.DataFrame(mean_vel, columns=['mean_vel']).T
    push_time = pd.DataFrame(push_time, columns=['push_time']).T
    cycle_time = pd.DataFrame(cycle_time, columns=['cycle_time']).T

    outcomes = pd.concat([step, work, mean_power, max_power, mean_vel, push_time, cycle_time])
    outcomes = outcomes.T
    return fig, outcomes


def sprint(data, data_pbp, title):
    """
    Sprint test analyse. Plot a figure with the power, speed and distance for
    left and right seperate. Also saves important outcomes

    Parameters
    ----------
    data : dict
        processed and cutted ergometer data dictionary with dataframes
    data_pbp : dict
        processed and cutted push_by_push ergometer data dictionary with dataframes
    title : str, optional
        title of figure

    Returns
    -------
    fig : matplotlib.figure.Figure
    outcomes : dataframe

    """
    fig = plot_power_speed_dist(data, title)

    outcomes = [{'distance': data['mean']['dist'].max(),
                 'max_vel': data['mean']['speed'].max(), 'mean_vel': data['mean']['speed'].mean(),
                 'max_vel_l': data['left']['speed'].max(), 'mean_vel_l': data['left']['speed'].mean(),
                 'max_vel_r': data['right']['speed'].max(), 'mean_vel_r': data['right']['speed'].mean(),
                 'max_power': data['mean']['power'].max(), 'mean_power': data['mean']['power'].mean(),
                 'mean_power_l': data['left']['power'].mean(), 'max_power_l': data['left']['power'].max(),
                 'max_power_r': data['right']['power'].max(), 'mean_power_r': data['right']['power'].mean(),
                 'ptime': data_pbp['mean']['ptime'].mean(), 'maxpowerafter3': data_pbp['mean']['maxpower'][0:3].max(),
                 'ctime': data_pbp['mean']['ctime'].mean()}]

    outcomes = pd.DataFrame(outcomes)

    return fig, outcomes
