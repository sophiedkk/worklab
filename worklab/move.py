import numpy as np
import pandas as pd


def get_perp_vector(vector2d: np.array, clockwise=True) -> np.array:
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
    if clockwise:
        """Gets 2D vector perpendicular to input vector, rotated clockwise"""
        perp_vector2d = np.empty(vector2d.shape)
        perp_vector2d[:, 0] = vector2d[:, 1] * -1
        perp_vector2d[:, 1] = vector2d[:, 0]
        perp_vector2d[:, 2] = vector2d[:, 2]
    else:
        """Gets 2D vector perpendicular to input vector, rotated counterclockwise"""
        perp_vector2d = np.empty(vector2d.shape)
        perp_vector2d[:, 0] = vector2d[:, 1]
        perp_vector2d[:, 1] = vector2d[:, 0] * -1
        perp_vector2d[:, 2] = vector2d[:, 2]
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


def calc_marker_angles(v_1: np.array, v_2: np.array, deg=False):
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
