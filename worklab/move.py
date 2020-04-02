import numpy as np
import pandas as pd


def get_perp_vector(vector2d, clockwise=True):
    """
    Get the vector perpendicular to the input vector. Only works in 2D as 3D has infinite solutions.

    Parameters
    ----------
    vector2d : np.array
        [n, 3] vector data
    clockwise : bool
        clockwise or counterclockwise rotation

    Returns
    -------
    perp_vector2d : np.array
        rotated vector

    """
    vector2d = np.asarray(vector2d)  # make sure it's an array
    perp_vector2d = np.zeros(vector2d.shape)
    if clockwise:
        """Gets 2D vector perpendicular to input vector, rotated clockwise"""
        perp_vector2d[:, 0] = vector2d[:, 1]
        perp_vector2d[:, 1] = vector2d[:, 0] * -1
    else:
        """Gets 2D vector perpendicular to input vector, rotated counterclockwise"""
        perp_vector2d[:, 0] = vector2d[:, 1] * -1
        perp_vector2d[:, 1] = vector2d[:, 0]
    return perp_vector2d


def get_rotation_matrix(new_frame, local_to_world=True):
    """
    Get the rotation matrix between a new reference frame and the global reference frame or the other way around.

    Parameters
    ----------
    new_frame : np.array
        3x3 array specifying the new reference frame
    local_to_world : bool
        global to local or local to global

    Returns
    -------
    rotation_matrix : np.array
        rotation matrix that can be used to rotate marker data, e.g.: rotation_matrix @ marker

    """
    x1 = np.array([1, 0, 0])  # X axis of the world
    x2 = np.array([0, 1, 0])  # Y axis of the world
    x3 = np.array([0, 0, 1])  # Z axis of the world

    x1_prime = new_frame[:, 0]  # X axis of the local frame
    x2_prime = new_frame[:, 1]  # Y axis of the local frame
    x3_prime = new_frame[:, 2]  # Z axis of the local frame

    if local_to_world:
        rotation_matrix = np.array([[np.dot(x1, x1_prime), np.dot(x1, x2_prime), np.dot(x1, x3_prime)],
                                    [np.dot(x2, x1_prime), np.dot(x2, x2_prime), np.dot(x2, x3_prime)],
                                    [np.dot(x3, x1_prime), np.dot(x3, x2_prime), np.dot(x3, x3_prime)]])
    else:
        rotation_matrix = np.array([[np.dot(x1_prime, x1), np.dot(x1_prime, x2), np.dot(x1_prime, x3)],
                                    [np.dot(x2_prime, x1), np.dot(x2_prime, x2), np.dot(x2_prime, x3)],
                                    [np.dot(x3_prime, x1), np.dot(x3_prime, x2), np.dot(x3_prime, x3)]])
    return rotation_matrix


def mirror(vector3d, axis="xyz"):
    """
    Simply mirror one or multiple axes.

    Parameters
    ----------
    vector3d: np.array
        vector to be mirrored, also works on dataframes
    axis: str
        string with axes to be mirrored

    Returns
    -------
    vector3d: np.array
        mirrored vector

    """
    x, y, z = 1, 1, 1
    if "x" in axis.lower():
        x = -1
    if "y" in axis.lower():
        y = -1
    if "z" in axis.lower():
        z = -1
    vector3d = vector3d * np.array([x, y, z])
    return vector3d


def scale(vector3d, x=1., y=1., z=1.):
    """
    Scale a vector in different directions.

    Parameters
    ----------
    vector3d: np.array
        array to be scaled, also works on dataframes
    x: float
        x-axis scaling
    y: float
        y-axis scaling
    z: floag
        z-axis scaling

    Returns
    -------
    vector3d: np.array
        scaled array

    """
    return vector3d * np.array([x, y, z])


def rotate(vector3d, angle, deg=False, axis="z"):
    """
    Rotate a vector around a given axis, specify rotation angle in radians or degrees.

    Parameters
    ----------
    vector3d: np.array
        vector to be rotated, also works on dataframes
    angle: float
        angle to rotate over
    deg: bool
        True if angle is specified in degrees, False for radians
    axis: str
        axis to rotate over, default = "z"

    Returns
    -------
    vector3d: np.array
        rotated vector

    """
    df = None  # we have to separately handle dataframes here
    if isinstance(vector3d, pd.DataFrame):
        df = vector3d.copy()
        vector3d = vector3d.values

    if deg:
        angle = np.deg2rad(angle)

    if axis.lower() == "x":
        vector3d[:, 1] = vector3d[:, 1] * np.cos(angle) + vector3d[:, 2] * np.sin(angle)
        vector3d[:, 2] = vector3d[:, 1] * np.sin(angle) * -1 + vector3d[:, 2] * np.cos(angle)
    elif axis.lower() == "y":
        vector3d[:, 0] = vector3d[:, 0] * np.cos(angle) + vector3d[:, 2] * np.sin(angle)
        vector3d[:, 2] = vector3d[:, 0] * np.sin(angle) * -1 + vector3d[:, 2] * np.cos(angle)
    else:
        vector3d[:, 0] = vector3d[:, 0] * np.cos(angle) + vector3d[:, 1] * np.sin(angle)
        vector3d[:, 1] = vector3d[:, 0] * np.sin(angle) * -1 + vector3d[:, 1] * np.cos(angle)

    if df is not None:
        df[["X", "Y", "Z"]] = vector3d
        vector3d = df
    return vector3d


def normalize(x):
    """
    Normalizes [n, 3] marker data using an l2 norm.

    Parameters
    ----------
    x : np.array
        marker data to be normalized

    Returns
    -------
    x : np.array
        normalized marker data

    """
    if isinstance(x, pd.DataFrame):
        return x.div(np.linalg.norm(x), axis=0)
    else:
        return x / np.linalg.norm(x)


def calc_marker_angles(v_1, v_2, deg=False):
    """
    Calculates n angles between two [n, 3] markers.

    Parameters
    ----------
    v_1 : np.array
        [n, 3] array or DataFrame for marker 1
    v_2 : np.array
        [n, 3] array or DataFrame for marker 2
    deg : bool
        return radians or degrees, default is radians

    Returns
    -------
    x : np.array
        returns [n, 1] array with the angle for each sample

    """
    p1 = np.einsum('ij,ij->i', v_1, v_2)
    p2 = np.cross(v_1, v_2, axis=1)
    p3 = np.linalg.norm(p2, axis=1)
    angles = np.arctan2(p3, p1)
    return np.rad2deg(angles) if deg else angles
