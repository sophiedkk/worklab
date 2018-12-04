"""
-Ergometer/measurement wheel data importer-
Description: Contains all functions specific for acceleration tests conducted
on an Esseda wheelchair ergometer. In an acceleration test a constant force
is applied and the acceleration of the rollers is measured.
Author:     Rick de Klerk
Contact:    r.de.klerk@umcg.nl
Company:    University Medical Center Groningen
License:    GNU GPLv3.0
Date:       26/03/2018
"""

import numpy as np
from pandas import DataFrame, to_numeric
from pandas.io import excel
import csv
import datetime
from . import formats
from struct import unpack


def pick_file():
    from tkinter import Tk
    from tkinter.filedialog import askopenfilename
    Tk().withdraw()  # no root window
    filename = askopenfilename(title="Open data file or files",
                               filetypes=[("Data files", "*.csv;*.xlsx;*.data")])  # return path to selected file
    print("You selected: ", filename)
    return filename


def load(filename="", obj=True, sfreq=200):
    """ Input: full file path or file in existing path from any source
        Output: will try to detect file type and choose the right load function"""
    if not filename:
        filename = pick_file()
    print("\n", "-" * 50)
    print(f"Loading data from {filename} ...")
    if "HSB" in filename:  # HSBlogger for Esseda
        data = load_HSB(filename)
        report_missing(data)
    elif "LEM" in filename:  # LEM datafile for Esseda
        data = load_LEM(filename)
        report_missing(data)
    elif ".xls" in filename:  # COSMED
        data = load_spiro(filename)
    elif ".data" in filename:  # Optipush
        data = load_opti(filename)
    elif ".txt" in filename:  # SMARTwheel
        data = load_sw(filename, sfreq=sfreq)
    elif "FIETS" in filename:  # Lode fiets
        data, _, _ = load_LEM_bike(filename)
    elif ".n3d" in filename:  # Optotrak
        data, _ = load_n3d(filename)
        return data
    else:
        raise Exception("Could not identify data source with load")
    if "right" in data and obj:
        data = formats.Kinetics(filename=filename, data=data)
    print("Data loaded!")
    print("-" * 50)
    return data


def load_HSB(filename):
    """ Input: full file path or file in existing path from HSBtool
        Output: dictionary with ergometer data in numpy arrays"""
    print("Loading from HSB data file")
    data = formats.get_erg_format()
    with open(filename, 'r') as csvfile:
        reader = csv.reader(csvfile, delimiter=';')
        next(reader, None)  # skip header
        for row in reader:
            if row[0] == "0":
                data["left"]["time"].append(float(row[1].replace(",", ".")))
                data["left"]["force"].append(float(row[2].replace(",", ".")))
                data["left"]["speed"].append(float(row[3].replace(",", ".")))
            else:
                data["right"]["time"].append(float(row[1].replace(",", ".")))
                data["right"]["force"].append(float(row[2].replace(",", ".")))
                data["right"]["speed"].append(float(row[3].replace(",", ".")))
    for side in data:
        data[side] = {dkey: np.asarray(data[side][dkey]) for dkey in data[side]}  # Convert to arrays
        if data[side]["time"].size:  # Remove time offset
            data[side]["time"] -= data[side]["time"][0]
            if data[side]["time"][1] - data[side]["time"][0] > 0.1:
                data[side]["time"] *= 0.01
        else:
            print("No right module detected!")
            data["right"] = {key: np.zeros(data["left"]["time"].size) for key in data["right"].keys()}
    for side in data:
        if np.mean(data[side]["force"]) < 0:
            data[side]["force"] *= -1  # Flip force direction
        if np.mean(data[side]["speed"]) < 0:
            data[side]["speed"] *= -1  # Flip speed direction
    return data


def load_LEM(filename):
    """ Input: full file path or file in existing path from LEM excel sheet
        Output: dictionary with ergometer data in numpy arrays"""
    print("Loading from LEM data file")
    df = excel.read_excel(filename, sheet_name="HSB")
    df = df.dropna(axis=1, how='all')  # remove empty columns
    df = df.apply(lambda col: to_numeric(col.str.replace(',', '.')) if isinstance(col[0], str) else col, axis=0)

    cols = len(df.columns) / 5  # LEM does this annoying thing where it starts in new columns
    mats = np.split(df.as_matrix(), int(cols), axis=1)
    dmat = np.concatenate(tuple(mats), axis=0)

    data = formats.get_lem_format()
    data["left"]["time"] = dmat[:, 0]
    data["left"]["uforce"] = dmat[:, 1]
    data["left"]["speed"] = dmat[:, 3]
    data["right"]["time"] = dmat[:, 0]
    data["right"]["uforce"] = dmat[:, 2]
    data["right"]["speed"] = dmat[:, 4]

    data = {idx: {dkey: data[idx][dkey][~np.isnan(data[idx]["uforce"])] for dkey in side}
            for (idx, side) in data.items()}  # remove NaNs created by empty rows in excel
    return data


def load_LEM_bike(filename):
    pp_info = excel.read_excel(filename, sheet_name=0)  # subject information
    pp_data = excel.read_excel(filename, sheet_name=2)  # 5 Hz data
    pp_data = pp_data.rename(columns=lambda x: x.strip())  # remove spaces from header
    pp_prot = excel.read_excel(filename, sheet_name=3, skiprows=2)  # skip to header
    return pp_data, pp_info, pp_prot


def load_opti(filename):
    """ Input: full file path or file in existing path from Optipush system
        Output: dictionary with measurement wheel data in numpy arrays"""
    print("Loading from Optipush data file")
    data = formats.get_mw_format()
    with open(filename, 'r') as csvfile:
        reader = csv.reader(csvfile, delimiter='\t')
        [next(reader, None) for _ in range(12)]  # skip header
        for row in reader:
            data["time"].append(float(row[0]))
            data["fx"].append(float(row[3]))
            data["fy"].append(float(row[4]))
            data["fz"].append(float(row[5]))
            data["mx"].append(float(row[6]))
            data["my"].append(float(row[7]))
            data["torque"].append(float(row[8]) * - 1)
            data["angle"].append(float(row[9]))
    data = {dkey: np.asarray(data[dkey]) for dkey in data}
    data["angle"] *= (np.pi / 180)
    data["angle"] -= data["angle"][0]  # remove angle offset
    return data


def load_sw(filename, sfreq=200):
    """ Input: full file path or file in existing path from SMARTwheel system
        Output: dictionary with measurement wheel data in numpy arrays"""
    print("Loading from SMARTwheel data file")
    data = formats.get_mw_format()
    with open(filename, 'r') as csvfile:
        reader = csv.reader(csvfile, delimiter=',')
        [next(reader, None) for _ in range(12)]  # skip header
        for row in reader:
            data["time"].append(float(row[1]))
            data["fx"].append(float(row[18]))
            data["fy"].append(float(row[19]))
            data["fz"].append(float(row[20]))
            data["mx"].append(float(row[21]))
            data["my"].append(float(row[22]))
            data["torque"].append(float(row[23]))
            data["angle"].append(float(row[3]))
    data = {dkey: np.asarray(data[dkey]) for dkey in data}
    data["time"] *= (1/sfreq)
    data["angle"] = np.unwrap(data["angle"] * (np.pi/180)) * - 1  # in radians
    return data


def load_LEM_spline(filename):
    """ Input: full file path or file in existing path from LEM excel file
        Output: spline value dictionary"""
    gear_ratio = 4  # Roller to loadcell
    data = {"left": [], "right": []}
    df = excel.read_excel(filename, sheet_name="Devices", header=5, skiprows=0)
    df = df.iloc[:, [1, 2]]  # Remove random columns
    df = df[8:88:8]  # Remove random rows
    data["left"] = df.values[:, 0]
    data["right"] = df.values[:, 1]
    for side in data:
        for idx, value in enumerate(data[side]):
            data[side][idx] = value.replace(",", ".")
        data[side] = data[side].astype(float) * gear_ratio
    return data


def dt_to_s(dt):
    h, m, s = dt.split(":")
    time = int(h) * 3600 + int(m) * 60 + int(s)
    return time


def load_spiro(filename):
    """ Input: full file path or file in existing path from COSMED spirometer
        Output: pandas dataframe with breath by breath data"""
    data = excel.read_excel(filename, skiprows=[1, 2], usecols="J:DX")
    data["time"], data["power"] = np.zeros(data.shape[0]), np.zeros(data.shape[0])
    data["time"] = data.apply(lambda row: dt_to_s(row["t"]), axis=1)  # hh:mm:ss to s
    data["power"] = data["EEm"] * 4184 / 60  # added power (kcal/min to J/s)
    return data


def load_n3d(filename, verbose=True):
    """Reads NDI-optotrak data files

    Input
        filename: *.NDF datafile
        verbose: default is True, disable to disable printouts
    Output
        data: list with markers with 3D arrays of measurement data in milimeters
        sfrq: sample frequency in Hz"""

    with open(filename, "rb") as f:
        content = f.read()

    # filetype = unpack('c', content[0:1])[0]
    items = unpack('h', content[1:3])[0]  # int16, number of markers
    subitems = unpack('h', content[3:5])[0]  # int16, number of dimensions (usually 3)
    numframes = unpack('i', content[5:9])[0]  # int32, number of frames
    sfrq = unpack('f', content[9:13])[0]  # float32, sample frequency
    # usercomment = merge_chars(unpack('c' * 60, content[13:73]))  # char
    # sys_comment = merge_chars(unpack('c' * 60, content[73:133]))
    # descrp_file = merge_chars(unpack('c' * 30, content[133:163]))
    # cuttoff_frq = unpack('h', content[163:165])
    coll_time = merge_chars(unpack('c' * 10, content[165:175]))
    coll_date = merge_chars(unpack('c' * 10, content[175:185]))
    # rest = merge_chars(unpack('c' * 71, content[185:256]))  # padding

    if verbose:
        print("-" * 50)
        print(f'Reading data from {filename}, recorded on {coll_date} at {coll_time} with {sfrq} Hz.')
        print("-" * 50)

    num_total = items * subitems * numframes  # total number of 'samples'
    optodata = np.array(unpack('f' * num_total, content[256:]))

    optodata[optodata <= -10e20] = np.nan  # replace NDF nan with nan
    optodata = np.reshape(optodata, (numframes, items, subitems)).T  # row = xyz, column = marker, 3rd = samples
    return optodata, sfrq


def report_missing(data):
    if "right" in data:
        for side in data:
            boolarray = np.diff(data[side]["time"]) > 0.011  # 0.001 higher because floats
            if np.sum(boolarray) or np.isnan(np.min(data[side]["speed"])):
                print("\n", "-" * 50)
                print(f"Number of missing datapoints for {side} module = {np.sum(boolarray)}")


def export_pushes(pbp):
    now = datetime.datetime.now()
    now = now.strftime("%Y%m%d_%H%M")
    df = DataFrame.from_dict(pbp, orient="index")
    df = df.transpose()
    df.to_csv(f"{now}_pbp.tsv", sep="\t", index=False)


def merge_chars(chars):
    """Merges list of binary characters to single string"""
    return ''.join([x.decode("utf-8") for x in chars])


if __name__ == "__main__":  # Some test functions
    # pick_file()
    # testdata = load("C:/Users/rick_/Data/20180312_Coast_down_tests/20180312_155033_HSB.csv")
    # report_missing(testdata)
    # load_opti("opti.data")
    # testdata = load_sw("SW.txt")
    # testdata = load_opti("C:\\Users\\rick_\\Development\\analysis\\example_data\\opti.data")
    testdata = load("C:\\Users\\rick_\\Development\\analysis\\example_data\\COSMED_example.xls")
    print(testdata.head())
