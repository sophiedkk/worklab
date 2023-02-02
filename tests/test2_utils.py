from worklab.utils import lowpass_butter
import matplotlib.pyplot as plt
from scipy import signal
import numpy as np


def signal_lag(y1, y2, sfreq=100, cutoff=6, order=2, plot=True, verbose=True):
    """
    Data alignment function, that can align two devices.

    Aligns 2 datasets based on given input variables, after low bandpass filtering.
    It is advised to use speed data for alignment.

    Parameters
    ----------
    y1: np.array, pd.Series
        Variable from first device which lag of y2 is calculated on, preferably velocity array
    y2: np.array, pd.Series
        Variable from second device which is compared to y1, preferably velocity array
    sfreq : float
        sample frequency of the signal, default is 100
    cutoff : float
        cutoff frequency for the filter, default is 6
    order : int
        order of the filter, default is 2
    plot : boolean
        can be used to plot the correlation line, default = True
    verbose : boolean
        can be used to print out the samples that y2 lags compared to y1, default = True

    Returns
    -------
    delay : int
        the number of samples that y2 should be shifted to correlate maximally to y1
    maxcorr : int
        the correlation between y2 and y1, if shifted according to the delay

    See Also
    --------
    lowpass_butter

    """

    y1 = lowpass_butter(y1, sfreq=sfreq, cutoff=cutoff, order=order)
    y2 = lowpass_butter(y2, sfreq=sfreq, cutoff=cutoff, order=order)
    n = len(y1)

    plot = True
    verbose = True

    corr = signal.correlate(y2, y1, mode='same') / np.sqrt(
        signal.correlate(y1, y1, mode='same')[int(n/2)] * signal.correlate(y2, y2, mode='same')[int(n/2)])
    delay_arr = np.linspace(-0.5*n, 0.5*n, n)
    maxcorr = np.argmax(corr)
    delay = int(round(delay_arr[maxcorr]))

    if plot:
        plt.figure()
        plt.plot(delay_arr, corr)
        plt.title('Lag: ' + str(delay) + ' s')
        plt.xlabel('Lag (samples)')
        plt.ylabel('Correlation coefficient')
        plt.show()

    if verbose:
        print("\n" + "=" * 80 + f"\nSignal y2 is {delay} samples off from y1!\n" + "=" * 80 + "\n")

    return delay, maxcorr
