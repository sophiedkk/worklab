"""
-Communication module-
Description: Contains functions for reading data from any worklab device. If you abide by regular naming conventions
you will only need the load function which will infer the correct function for you. You can also use device-specific
load functions if needed.
Author:     Rick de Klerk
Contact:    r.de.klerk@umcg.nl
Company:    University Medical Center Groningen
License:    GNU GPLv3.0
Date:       27/06/2019
"""
import csv
import re
from collections import defaultdict
from glob import glob
from os import listdir, path
from struct import unpack

import numpy as np
import pandas as pd

from .utils import pick_file, pd_dt_to_s, merge_chars


def load(filename: str = ""):
    """Most important function in the module. Provides high level loading function to load common data formats.
    If no filename is given will try to load filename using a file dialog. Will try to infer data source from filename.
    Try to use a specific load function if load cannot infer the datatype.

    :param filename: name or path to file of interest
    :return: raw data, format depends on source, but is usually a dict or pandas dataframe
    """
    filename = pick_file() if not filename else filename
    if not filename:
        raise Exception("Please provide a filename")
    print("\n" + "=" * 80)
    print(f"Initializing loading for {filename} ...")
    if ".xlsx" in filename or "spiro" in filename.lower():  # COSMED
        print("File identified as COSMED datafile. Attempting to load ...")
        data = load_spiro(filename)
    elif "lem" in filename.lower():
        print("File identified as Esseda datafile. Attempting to load ...")
        data = load_esseda(filename)
    elif "HSB.csv" in filename:
        print("File identified as HSB-logger datafile. Attempting to load ...")
        data = load_hsb(filename)
    elif ".txt" in filename and "drag" not in filename.lower():
        print("File identified as SMARTwheel datafile. Attempting to load ...")
        data = load_sw(filename)
    elif ".dat" in filename:
        print("File identified as Optipush datafile. Attempting to load ...")
        data = load_opti(filename)
    elif "fiets" in filename.lower():
        print("File identified as Bicycle ergometer datafile. Attempting to load ...")
        data = load_bike(filename)
    elif ".n3d" in filename:
        print("File identified as Optotrak datafile. Attempting to load ...")
        data, _ = load_n3d(filename)
    elif ".xml" in filename:
        print("Folder identified as NGIMU folder. Attempting to load ...")
        data = load_session(path.split(filename)[0], filenames=["sensors"])
    elif "drag" in filename.lower():
        data = load_drag_test(filename)
    elif ".csv" in filename.lower():
        data = load_optitrack(filename)
    else:
        raise Exception("No file name given or could not identify data source with load!!")
    print("Data loaded!")
    print("=" * 80 + "\n")
    return data


def load_spiro(filename: str) -> pd.DataFrame:
    """Loads COSMED spirometer data from excel file

    :param filename: full file path or file in existing path from COSMED spirometer
    :return: Spiro data in pandas dataframe
    """
    data = pd.read_excel(filename, skiprows=[1, 2], usecols="J:XX")
    data["time"] = data.apply(lambda row: pd_dt_to_s(row["t"]), axis=1)  # hh:mm:ss to s
    data["power"] = data["EEm"] * 4184 / 60  # added power (kcal/min to J/s)
    data["weights"] = np.insert(np.diff(data["time"]), 0, 0)  # used for calculating weighted average
    data["HR"] = np.zeros(len(data)) if "HR" not in data else data["HR"]  # can be missing when sensor is not detected
    data = data[data["time"] > 0]  # remove "missing" data
    return data[["time", "Rf", "HR", "power", "VO2", "VCO2", "weights"]]


def load_opti(filename: str) -> pd.DataFrame:
    """Loads Optipush data from .data file

    :param filename: filename or path to Optipush file
    :return: dataframe with 3D kinetics data
    """
    names = ["time", "fx", "fy", "fz", "mx", "my", "torque", "angle"]
    dtypes = {name: np.float64 for name in names}
    usecols = [0, 3, 4, 5, 6, 7, 8, 9]
    opti_df = pd.read_csv(filename, names=names, delimiter="\t", usecols=usecols, dtype=dtypes, skiprows=12)
    opti_df["angle"] *= (np.pi / 180)
    opti_df["torque"] *= -1
    return opti_df


def load_sw(filename: str, sfreq: int = 200) -> pd.DataFrame:
    """Loads SMARTwheel data from .csv file

    :param filename: filename or path to SMARTwheel data
    :param sfreq: samplefreq, this can be changed for a test, default is 200
    :return: dataframe with 3D kinetics data
    """
    names = ["time", "fx", "fy", "fz", "mx", "my", "torque", "angle"]
    dtypes = {name: np.float64 for name in names}
    usecols = [1, 18, 19, 20, 21, 22, 23, 3]
    sw_df = pd.read_csv(filename, names=names, usecols=usecols, dtype=dtypes)
    sw_df["time"] *= (1 / sfreq)
    sw_df["angle"] = np.unwrap(sw_df["angle"] * (np.pi / 180)) * - 1  # in radians
    return sw_df


def load_hsb(filename: str) -> dict:
    """Loads HSB ergometer data from HSB datafile

    :param filename: full file path or file in existing path from HSB csv file
    :return: dictionary with ergometer data in dataframes
    """
    # noinspection PyTypeChecker
    data = {"left": defaultdict(list), "right": defaultdict(list)}
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
            data["right"] = {key: np.zeros(data["left"]["time"].size) for key in data["right"]}
    for side in data:
        if np.mean(data[side]["force"]) < 0:
            data[side]["force"] *= -1  # Flip force direction
        if np.mean(data[side]["speed"]) < 0:
            data[side]["speed"] *= -1  # Flip speed direction
    return {"left": pd.DataFrame(data["left"]), "right": pd.DataFrame(data["right"])}


def load_esseda(filename: str) -> dict:
    """Loads HSB ergometer data from LEM datafile

    :param filename: full file path or file in existing path from LEM excel sheet
    :return: dictionary with ergometer data in dataframes
    """
    df = pd.read_excel(filename, sheet_name="HSB")
    df = df.dropna(axis=1, how='all')  # remove empty columns
    df = df.apply(lambda col: pd.to_numeric(col.str.replace(',', '.')) if isinstance(col[0], str) else col, axis=0)

    cols = len(df.columns) / 5  # LEM does this annoying thing where it starts in new columns
    mats = np.split(df.values, int(cols), axis=1)
    dmat = np.concatenate(tuple(mats), axis=0)

    data = {"left": pd.DataFrame(), "right": pd.DataFrame()}
    data["left"]["time"] = dmat[:, 0]
    data["left"]["force"] = dmat[:, 1]
    data["left"]["speed"] = dmat[:, 3]
    data["right"]["time"] = dmat[:, 0]
    data["right"]["force"] = dmat[:, 2]
    data["right"]["speed"] = dmat[:, 4]

    for side in data:
        data[side].dropna(inplace=True)
    return data


def load_bike(filename: str) -> pd.DataFrame:
    """Load bicycle ergometer data from LEM datafile

    :param filename: path to file for pandas to parse
    :return: DataFrame with time, load, rpm, and HR data
    """
    return pd.read_excel(filename, sheet_name=2, names=["time", "load", "rpm", "HR"])  # 5 Hz data


def load_spline(filename) -> dict:
    """Load wheelchair ergometer calibration spline from LEM datafile

    :param filename: full file path or file in existing path from LEM excel file
    :return: spline value dictionary
    """
    gear_ratio = 4  # Roller to loadcell
    data = {"left": None, "right": None}
    df = pd.read_excel(filename, sheet_name="Devices", header=5, skiprows=0)
    df = df.iloc[:, [1, 2]]  # Remove random columns
    df = df[8:88:8]  # Remove random rows
    data["left"] = df.values[:, 0]
    data["right"] = df.values[:, 1]
    for side in data:
        for idx, value in enumerate(data[side]):
            data[side][idx] = value.replace(",", ".")
        data[side] = data[side].astype(float) * gear_ratio
    return data


def load_n3d(filename: str, verbose: bool = True) -> np.array:
    """Reads NDI-optotrak data files

    :param filename: *.NDF datafile
    :param verbose: default is True, disable to disable printouts
    :return: list with markers with 3D arrays of measurement data in milimeters
    """
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
    return optodata


def load_session(root_dir: str, filenames: list = None) -> dict:
    """Imports NGIMU session in nested dictionary with all devices and sensors. Translated from xio-Technologies.
    https://github.com/xioTechnologies/NGIMU-MATLAB-Import-Logged-Data-Example

    :param root_dir: directory where session is located
    :param filenames: Optional - list of sensor names or single sensor name that you would like to include
    :return: returns nested object sensordata[device][sensor][dataframe]
    """
    directory_contents = listdir(root_dir)  # all content in directory
    if not directory_contents:
        raise Exception("No contents in directory")
    if "Session.xml" not in directory_contents:
        raise Exception("Session.xml not found.")
    directories = glob(f"{root_dir}/*/")  # folders of all devices
    session_data = dict()

    for sensordir in directories:  # loop through all sensor directories
        sensor_files = glob(f"{sensordir}/*.csv")
        device_name = path.split(path.split(sensordir)[0])[-1]
        device_name = "Left" if "links" in device_name.lower() or "left" in device_name.lower() else device_name
        device_name = "Right" if "rechts" in device_name.lower() or "right" in device_name.lower() else device_name
        device_name = "Frame" if "frame" in device_name.lower() else device_name
        session_data[device_name] = dict()

        for sensor_file in sensor_files:  # loop through all csv files
            sensor_name = path.split(sensor_file)[-1].split(".csv")[0]  # sensor without path or extension

            if filenames and sensor_name not in filenames:
                continue  # skip if filenames is given and sensor not in filenames

            session_data[device_name][sensor_name] = pd.read_csv(sensor_file).drop_duplicates()
            session_data[device_name][sensor_name].rename(columns=lambda x: re.sub("[(\[].*?[)\]]", "", x)
                                                          .replace(" ", ""), inplace=True)  # remove units from name

        if not session_data[device_name]:
            raise Exception("No data was imported")
    return session_data


def load_drag_test(filename):
    dragtest = defaultdict(list)
    with open(filename, "r") as f:
        [next(f) for _ in range(33)]  # Skip header and junk
        for line in f:
            data = line.split()
            dragtest["angle"].append(float(data[0]))
            dragtest["force"].append(float(data[1]))
    return pd.DataFrame(dragtest)


def load_optitrack(filename: str, include_header: bool = False):
    """Loads Optitrack marker data

    :param filename: Full path to filename or filename in current path
    :param include_header: Whether or not to include the header in the output
    :return: Marker data in dictionary, metadata in dictionary
    """
    # First get all the metadata
    header = {}
    with open(filename, 'r') as f:
        header["metadata"] = f.readline().replace("\n", "").split(",")
        next(f)
        header["marker_type"] = f.readline().replace("\n", "").replace(",,", "frame,time,").split(",")
        header["marker_label"] = f.readline().replace("\n", "").replace(",,", "frame,time,").split(",")
        header["marker_id"] = f.readline().replace("\n", "").replace(",,", "frame,time,").split(",")
        header["header_label1"] = f.readline().replace("\n", "").replace(",,", "frame,time,").split(",")
        header["header_label2"] = f.readline().replace("\n", "").split(",")

    # Split the metadata string into key value pairs
    metadata_dict = {}
    for key, value in zip(header["metadata"][0::2], header["metadata"][1::2]):
        if "Frame" in key:
            value = float(value)
        metadata_dict[key] = value
    header["metadata"] = metadata_dict

    # Get the markers
    first_marker = header["marker_type"].index("Marker")
    n_markers = (len(header["marker_type"]) - first_marker) // 3

    marker_data = {}
    for i in range(n_markers):
        marker_label = header["marker_label"][first_marker + i * 3]
        if "Unlabeled" in marker_label:
            marker_label = "marker_" + str(i)
        marker_columns = [first_marker + i * 3, first_marker + 1 + i * 3, first_marker + 2 + i * 3]
        marker_data[marker_label] = pd.read_csv(filename, skiprows=list(range(7)), usecols=marker_columns,
                                                names=["X", "Y", "Z"])
        marker_data[marker_label] = marker_data[marker_label][:-1]  # remove last (empty) row
    return marker_data, header if include_header else marker_data
