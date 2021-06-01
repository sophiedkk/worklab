import numpy as np

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
    index_start = abs(data_spiro['time'] - start).idxmin() + 1
    index_end = abs(data_spiro['time'] - end).idxmin()
    data_spiro['time'] = data_spiro['time'] - start

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
