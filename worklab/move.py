import numpy as np
import pandas as pd


def get_perp_vector(vector2d, clockwise=True, normalized=True):
    """
    Get the vector perpendicular to the input vector. Only works in 2D as 3D has infinite solutions.

    Parameters
    ----------
    vector2d : np.array
        [n, 3] vector data, only uses x and y
    clockwise : bool
        clockwise or counterclockwise rotation
    normalized : bool
        whether or not to normalize the result, default is True

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
    return normalize(perp_vector2d) if normalized else perp_vector2d


def get_rotation_matrix(new_frame, local_to_world=True):
    """
    Get the rotation matrix between a new reference frame and the global reference frame or the other way around.

    Parameters
    ----------
    new_frame : np.array
        [3, 3] array specifying the new reference frame
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


def get_orthonormal_frame(point1, point2, point3, mean=False):
    """Returns an orthonormal frame from three reference points. For example, a local coordinate system from three
    marker points.

    Parameters
    ----------
    point1: np.array
        first marker point, used as origin if mean=False
    point2:  np.array
        second marker point, used as x-axis
    point3: np.array
        third marker point
    mean: bool
        whether or not the mean should be used as origin, default is False

    Returns
    -------
    origin: np.array
        xyz column vector with coordinates of the origin which is point1 or the mean of all points
    orthonormal: np.array
        3x3 array with orthonormal coordinates [x, y, z] of the new axis system

    """
    if mean:
        origin = np.mean(np.vstack([point1, point2, point3]), axis=0)
    else:
        origin = np.array(point1)

    x_axis = normalize(point2 - origin)
    y_axis = normalize(np.cross(x_axis, point3 - origin))
    z_axis = normalize(np.cross(x_axis, y_axis))

    orthonormal = np.vstack([x_axis, y_axis, z_axis]).T
    return origin[:, None], orthonormal


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
        array to be scaled, also works on dataframes, assumes [n, xyz] data
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
    Rotate a vector around a single given axis, specify rotation angle in radians or degrees.

    Parameters
    ----------
    vector3d: np.array
        vector to be rotated, also works on dataframes, assumes [n, xyz] data
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

    output_vector = vector3d.copy()

    if axis.lower() == "x":
        output_vector[:, 1] = vector3d[:, 1] * np.cos(angle) + vector3d[:, 2] * np.sin(angle)
        output_vector[:, 2] = vector3d[:, 1] * np.sin(angle) * -1 + vector3d[:, 2] * np.cos(angle)
    elif axis.lower() == "y":
        output_vector[:, 0] = vector3d[:, 0] * np.cos(angle) + vector3d[:, 2] * np.sin(angle)
        output_vector[:, 2] = vector3d[:, 0] * np.sin(angle) * -1 + vector3d[:, 2] * np.cos(angle)
    else:
        output_vector[:, 0] = vector3d[:, 0] * np.cos(angle) + vector3d[:, 1] * np.sin(angle)
        output_vector[:, 1] = vector3d[:, 0] * np.sin(angle) * -1 + vector3d[:, 1] * np.cos(angle)

    if df is not None:
        df[["X", "Y", "Z"]] = output_vector
        output_vector = df
    return output_vector


def magnitude(vector3d):
    """
    Calculates the vector magnitude using an l2 norm. Works with [1, 3] or [n, 3] vectors.

    Parameters
    ----------
    vector3d: np.array
        a [1, 3] or [n, 3] vector

    Returns
    -------
    vector3d: np.array
        scalar value or column vector

    """
    if vector3d.ndim == 1:
        return np.linalg.norm(vector3d)
    else:
        return np.linalg.norm(vector3d, axis=1)[:, None]  # make column vector


def normalize(vector3d):
    """
    Normalizes [n, 3] marker data using an l2 norm. Works with [1, 3] and [n, 3] vectors, both arrays and dataframes.

    Parameters
    ----------
    vector3d : np.array
        marker data to be normalized

    Returns
    -------
    vector3d : np.array
        normalized marker data

    """
    if vector3d.ndim == 1:
        return vector3d / magnitude(vector3d)
    else:
        if isinstance(vector3d, pd.DataFrame):
            return vector3d.div(magnitude(vector3d), axis=1)
        else:
            return vector3d / magnitude(vector3d)


def distance(point1, point2):
    """
    Compute Euclidean distance between two points, this is the distance if you were to draw a straight line.

    Parameters
    ----------
    point1 : np.array
        a [1, 3] or [n, 3] array with point coordinates
    point2 : np.array
        a [1, 3] or [n, 3] array with point coordinates
    Returns
    -------
    distance : np.array
        distance from point1 to point2 in a [1, 3] or [n, 3] array

    """
    if point1.ndim == 1 and point2.ndim == 1:
        return np.sqrt(np.sum(np.square(point2 - point1)))
    else:
        return np.sqrt(np.sum(np.square(point2 - point1), axis=1))


def marker_angles(v_1, v_2, deg=False):
    """
    Calculates n angles between two [n, 3] markers, two [1, 3] markers, or one [n, 3] and one [1, 3] marker.

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
        returns [n, 1] array with the angle for each sample or scalar value

    """
    if v_1.ndim == 1 and v_2.ndim == 1:
        angle = np.arctan2(np.linalg.norm(np.cross(v_1, v_2)), np.dot(v_1, v_2))
        return np.rad2deg(angle) if deg else angle
    else:
        if v_1.ndim == 1:
            v_1 = np.tile(v_1, (len(v_2), 1))  # extend array if necessary
        elif v_2.ndim == 1:
            v_2 = np.tile(v_2, (len(v_1), 1))
        p1 = np.einsum('ij,ij->i', v_1, v_2)  # vectorized dot operation
        p2 = np.cross(v_1, v_2, axis=1)
        p3 = np.linalg.norm(p2, axis=1)
        angles = np.arctan2(p3, p1)
        return np.rad2deg(angles) if deg else angles


def is_unit_length(vector3d, atol=1.e-8):
    """Checks whether an array ([1, 3] or [n, 3]) is equal to unit length given a tolerance"""
    return np.allclose(magnitude(vector3d), 1.0, rtol=0, atol=atol)
