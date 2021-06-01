import numpy as np
import pandas as pd
from numpy.linalg import solve
from scipy.spatial.transform import Rotation as R

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


def acs_to_car_ang(acs, order=[0, 1, 2]):
    """Anatomical coordinate system to cardanic angles

    Note: Only works if used in the DSEM coordinate system
    Y pointing upward, X pointing laterally to the right and Z point backwards

    Parameters
    ----------
    acs: np.array
        anatomical coordinate system

    order: list[int]
        list of integers 0='x', 1='y', 2='z'

    Returns
    -------
    angles: np.array
        cardanic angles
    """

    i = order[0]
    j = order[1]
    k = order[2]

    if np.remainder(j - i + 3, 3) == 1:
        a = 1
    else:
        a = -1

    if i != k:  # Cardan angles
        a1 = np.arctan2(-a * acs[:, j, k], acs[:, k, k])
        a2 = np.arcsin(a * acs[:, i, k])
        a3 = np.arctan2(-a * acs[:, i, j], acs[:, i, i])

    else:  # Euler angles
        l = 3 - i - j
        a1 = np.arctan2(acs[:, j, i], -a * acs[:, l, i])
        a2 = np.arccos(acs[:, i, i])
        a3 = np.arctan2(acs[:, i, j], a * acs[:, i, l])

    angles = np.stack([a1, a2, a3], axis=1)

    return angles


def make_marker_dict(markers, marker_names=None):
    """Create a dictionary of nx3 arrays with name samples

    Parameters
    ----------
    marker : np.array
        marker points

    marker_names : list[strings]
        list of marker names

    Returns
    -------
    marker_dict: dict[np.array]
        dictionary of nx3 markers with names
    """

    if marker_names is None:
        marker_names = ['Hand1', 'Hand2', 'Hand3', 'Low_arm1', 'Low_arm2', 'Low_arm3', 'Up_arm1', 'Up_arm2', 'Up_arm3',

                        'Acro1', 'Acro2', 'Acro3', 'Sternum1', 'Sternum2', 'Sternum3', 'Wheel1', 'Wheel2', 'Wheel3',

                        'Racket1', 'Racket2', 'Racket3', 'M2', 'M5', 'RS', 'US', 'EM', 'EL', 'TS', 'AI', 'AA',

                        'AC', 'PC', 'C7', 'T8', 'PX', 'IJ', 'SC', 'Centre', '12 clock', '4 clock', '8 clock', 'TopBlade',

                        'LeftBlade', 'BottomGrip']

    if len(marker_names) != markers.shape[2]:
        raise IndexError("Number of names and markers are not identical.")
    marker_dict = {}

    for idx, name in enumerate(marker_names):
        marker_dict[name] = markers[..., idx]
    return marker_dict


def rotate_matrix(ang, axis='z'):
    """Create a rotation matrix to rotate around x, y or z-axis

    Parameters
    ----------
    ang : int
        angle of rotation

    axis : string (default = 'z')
        axis of rotation, 'x', 'y' or 'z'

    Returns
    -------
    rotate: np.array
        rotation matrix
    """

    if axis.lower() == 'x':
        rotate = [[1, 0, 0], [0, np.cos(ang), -np.sin(ang)], [0, np.sin(ang), np.cos(ang)]]
    elif axis.lower() == 'y':
        rotate = [[np.cos(ang), 0, np.sin(ang)], [0, 1, 0], [-np.sin(ang), 0, np.cos(ang)]]
    else:
        rotate = [[np.cos(ang), -np.sin(ang), 0], [np.sin(ang), np.cos(ang), 0], [0, 0, 1]]

    return rotate


def get_local_coordinate(marker, acs, origin):
    """Make the local coordinate system from the anatomical coordinate system

    Parameters
    ----------
    marker : np.array
        marker points

    acs : np.array
        anatomical coordinate system

    origin : np.array
        origin for the local coordinate system

    Returns
    -------
    local_marker : dict[np.array]
        local coordinate system
    """

    marker = marker.copy()
    marker -= origin
    local_marker = solve(acs, marker)

    return local_marker


def make_acs_sc(AA, TS, AI, DSEM=False):
    """Make the anatomical coordinate system of the scapula based on ISB recommendations

    Y is pointing upwards
    X is pointing to the front
    Z is pointing laterally to the right

    Parameters
    ----------
    AA : np.array
        data points of the angulus acromialis

    TS : np.array
        data points of the trigonum spinae

    AI : np.array
        data points of the angulus inferior

    DSEM : boolean
        set to True to use coordinate system guidelines DSEM
        Y pointing upward, X pointing laterally to the right and Z point backwards

    Returns
    -------
    local : dict[np.array]
        local coordinate system of the scapula

    acs : np.array
        anatomical coordinate system of the scapula

    origin: np.array
        origin of the local coordinate system of the scapula
    """
    origin = AA

    if DSEM is False:
        z_axis = AA - TS
        support_vector = AI - AA
        x_axis = np.cross(z_axis, support_vector, axis=1)
        y_axis = np.cross(z_axis, x_axis, axis=1)
    else:
        x_axis = AA - TS
        support_vector = AA - AI
        z_axis = np.cross(x_axis, support_vector, axis=1)
        y_axis = np.cross(z_axis, x_axis, axis=1)

    z_axis = normalize(z_axis)
    x_axis = normalize(x_axis)
    y_axis = normalize(y_axis)

    acs = np.stack([x_axis, y_axis, z_axis], axis=2)
    local = {}
    points = [AA, TS, AI]
    names = ['AA', 'TS', 'AI']
    for points, names in zip(points, names):
        local[names] = get_local_coordinate(points, acs, origin)

    return local, acs, origin


def make_acs_th(IJ, PX, C7, T8, DSEM=False):
    """Make the anatomical coordinate system of the thorax based on ISB recommendations

    Y is pointing upwards
    X is pointing to the front
    Z is pointing laterally to the right

    Parameters
    ----------
    IJ : np.array
        data points of the incisura jugularis

    PX : np.array
        data points of the processus xiphoideus

    C7: np.array
        data points of C7

    T8: np.array
        data points of T8

    DSEM: boolean (default = False)
        set to True to use coordinate system guidelines DSEM
        Y pointing upward, X pointing laterally to the right and Z point backwards

    Returns
    -------
    local : dict[np.array]
        local coordinate system of the thorax

    acs : np.array
        anatomical coordinate system of the thorax

    origin: np.array
        origin of the local coordinate system of the thorax
    """
    origin = IJ
    y_axis = (C7 + IJ) / 2 - (T8 + PX) / 2

    if DSEM is False:
        support_vector = normalize(IJ - (C7 + IJ) / 2)
        z_axis = np.cross(support_vector, y_axis, axis=1)
        x_axis = np.cross(y_axis, z_axis, axis=1)
    else:
        support_vector = normalize((C7 + IJ) / 2 - IJ)
        x_axis = np.cross(y_axis, support_vector, axis=1)
        z_axis = np.cross(x_axis, y_axis, axis=1)

    z_axis = normalize(z_axis)
    x_axis = normalize(x_axis)
    y_axis = normalize(y_axis)

    acs = np.stack([x_axis, y_axis, z_axis], axis=2)
    local = {}
    points = [IJ, PX, C7, T8]
    names = ['IJ', 'PX', 'C7', 'T8']
    for points, names in zip(points, names):
        local[names] = get_local_coordinate(points, acs, origin)

    return local, acs, origin


def make_acs_cl(SC, AC, IJ, PX, C7, T8, AA=None, DSEM=False):
    """Make the anatomical coordinate system of the clavicule based on ISB recommendations

    Y is pointing upwards
    X is pointing to the front
    Z is pointing laterally to the right

    Parameters
    ----------
    SC: np.array
        data points of the sternoclaviculare joint

    AC: np.array
        data points of the dorsal acromioclaviculare joint

    IJ : np.array
        data points of the incisura jugularis

    PX : np.array
        data points of the processus xiphoideus

    C7: np.array
        data points of C7

    T8: np.array
        data points of T8

    AA: np.array
        data points of AA (only necessary for DSEM)

    DSEM: boolean (default = False)
        set to True to use coordinate system guidelines DSEM
        Y pointing upward, X pointing laterally to the right and Z point backwards

    Returns
    -------
    local : dict[np.array]
        local coordinate system of the clavicule

    acs : np.array
        anatomical coordinate system of the clavicule

    origin: np.array
        origin of the local coordinate system of the clavicule
    """

    origin = SC

    if DSEM is False:
        z_axis = AC - SC
        support_vector = ((C7 + IJ) / 2 - (T8 + PX) / 2)
        x_axis = np.cross(support_vector, z_axis, axis=1)
        y_axis = np.cross(z_axis, x_axis, axis=1)
        points = [SC, AC, IJ, PX, C7, T8]
        names = ['SC', 'AC', 'IJ', 'PX', 'C7', 'T8']
    else:
        x_axis = AC - SC
        support_vector = AA - AC
        y_axis = np.cross(support_vector, x_axis, axis=1)
        z_axis = np.cross(x_axis, y_axis, axis=1)
        points = [SC, AC, AA]
        names = ['SC', 'AC', 'AA']

    z_axis = normalize(z_axis)
    x_axis = normalize(x_axis)
    y_axis = normalize(y_axis)

    acs = np.stack([x_axis, y_axis, z_axis], axis=2)
    local = {}
    for points, names in zip(points, names):
        local[names] = get_local_coordinate(points, acs, origin)

    return local, acs, origin


def make_acs_hu(GH, EL, EM, DSEM=False):
    """Make the anatomical coordinate system of the humerus based on ISB recommendations

    Y is pointing upwards
    X is pointing to the front
    Z is pointing laterally to the right

    Parameters
    ----------
    GH: np.array
        data points of the glenohumeral rotation centre

    EL: np.array
        data points of the epicondylus lateral

    EM: np.array
        data points of the epicondylus medial

    DSEM: boolean (default = False)
        set to True to use coordinate system guidelines DSEM
        Y pointing upward, X pointing laterally to the right and Z point backwards

    Returns
    -------
    local : dict[np.array]
        local coordinate system of the humerus

    acs : np.array
        anatomical coordinate system of the humerus

    origin: np.array
        origin of the local coordinate system of the humerus
    """
    if DSEM is False:
        origin = GH
        y_axis = GH - (EL + EM) / 2
        support_vector = EM - EL
        x_axis = np.cross(support_vector, y_axis, axis=1)
        z_axis = np.cross(x_axis, y_axis, axis=1)
    else:
        origin = (EM + EL) / 2
        y_axis = GH - origin
        support_vector = EL - origin
        z_axis = np.cross(support_vector, y_axis, axis=1)
        x_axis = np.cross(y_axis, z_axis)

    z_axis = normalize(z_axis)
    x_axis = normalize(x_axis)
    y_axis = normalize(y_axis)

    acs = np.stack([x_axis, y_axis, z_axis], axis=2)
    local = {}
    points = [GH, EL, EM]
    names = ['GH', 'EL', 'EM']
    for points, names in zip(points, names):
        local[names] = get_local_coordinate(points, acs, origin)

    return local, acs, origin


def make_acs_fa(US, RS, EL, EM, DSEM=False):
    """Make the anatomical coordinate system of the forearm based on ISB recommendations

    Y is pointing upwards
    X is pointing to the front
    Z is pointing laterally to the right

    Parameters
    ----------
    US: np.array
        data points of the ulna styloid

    RS: np.array
        data points of the radial styloid

    EL: np.array
        data points of the epicondylus lateral

    EM: np.array
        data points of the epicondylus medial

    DSEM: boolean (default = False)
        set to True to use coordinate system guidelines DSEM
        Y pointing upward, X pointing laterally to the right and Z point backwards

    Returns
    -------
    local : dict[np.array]
        local coordinate system of the forearm

    acs : np.array
        anatomical coordinate system of the forearm

    origin: np.array
        origin of the local coordinate system of the forearm
    """
    if DSEM is False:
        origin = US
        y_axis = (EL + EM) / 2 - US
        support_vector = RS - US
        x_axis = np.cross(y_axis, support_vector, axis=1)
        z_axis = np.cross(x_axis, y_axis, axis=1)
    else:
        origin = (US + RS) / 2
        y_axis = origin - (EM + EL) / 2
        x_axis = origin - RS
        z_axis = np.cross(x_axis, y_axis)

    z_axis = normalize(z_axis)
    x_axis = normalize(x_axis)
    y_axis = normalize(y_axis)

    acs = np.stack([x_axis, y_axis, z_axis], axis=2)
    local = {}
    points = [US, RS, EL, EM]
    names = ['US', 'RS', 'EL', 'EM']
    for points, names in zip(points, names):
        local[names] = get_local_coordinate(points, acs, origin)

    return local, acs, origin


def make_acs_hand(M2, M5, US, RS):
    """Make the anatomical coordinate system of the hand in DSEM

    Y is pointing upwards
    X is pointing laterally to the right
    Z is pointing backwards

    Parameters
    ----------
    M2: np.array
        data points of metacarpal 2

    M5: np.array
        data points of metacarpal 5

    US: np.array
        data points of the ulna styloid

    RS: np.array
        data points of the radial styloid

    Returns
    -------
    local : dict[np.array]
        local coordinate system of the hand

    acs : np.array
        anatomical coordinate system of the hand

    origin: np.array
        origin of the local coordinate system of the hand
    """

    origin = (M5 + M2) / 2
    y_axis = origin - (US + RS) / 2
    x_axis = origin - M2
    z_axis = np.cross(x_axis, y_axis, axis=1)

    z_axis = normalize(z_axis)
    x_axis = normalize(x_axis)
    y_axis = normalize(y_axis)

    acs = np.stack([x_axis, y_axis, z_axis], axis=2)
    local = {}
    points = [M2, M5, US, RS]
    names = ['M2', 'M5', 'US', 'RS']
    for points, names in zip(points, names):
        local[names] = get_local_coordinate(points, acs, origin)

    return local, acs, origin


def flexext_prosup(GH, EL, EM, US, RS):
    """Flexion/extension as well as pro/supination in the DSEM reference frame
    Y pointing upward, X pointing laterally to the right and Z point backwards

    Parameters
    ----------
    GH: np.array
        data points of the glenohumeral rotation centre

    EL: np.array
        data points of the epicondylus lateral

    EM: np.array
        data points of the epicondylus medial

    US: np.array
        data points of the ulna styloid

    RS: np.array
        data points of the radial styloid

    Returns
    -------
    angles: np.array
        angles for flexion/extention, as well as pro/supination
    """

    local_hu, acs_hu, origin_hu = make_acs_hu(GH, EL, EM, DSEM=True)
    local_fa, acs_fa, origin_fa = make_acs_fa(US, RS, EL, EM, DSEM=True)

    elbow = solve(acs_hu, acs_fa)
    angles = acs_to_car_ang(elbow, order=[0, 2, 1], DSEM=True) * 180 / np.pi
    angles = angles[:, 0:2]

    return angles


def find_gh_regression(markers):
    """Get the location of the glenohumeral joint in the global coordinate system of DSEM
    In vivo estimation of the glenohumeral joint rotation center from scapular bony landmarks by linear regression
    C.G.M. Meskers, F.C.T. van der Helm, L.A. Rozendaal, P.M. Rozing

    Parameters
    ----------
    markers: np.array
        data points of upper-limb kinematics

    Returns
    -------
    GH: np.array
        Glenohumeral rotation center in the global coordinate system
    """

    origin = markers["AA"]
    local, acs, origin = make_acs_sc(origin, markers["TS"], markers["AI"], DSEM=True)
    data_r = {}

    for name in ["AA", "AC", "AI", "PC", "TS"]:
        data_r[name] = get_local_coordinate(markers[name], acs, origin)

    AI2AA = magnitude(data_r["AI"] - data_r["AA"])[0]
    AC2AA = magnitude(data_r["AC"] - data_r["AA"])[0]
    AC2PC = magnitude(data_r["AC"] - data_r["PC"])[0]
    TS2PC = magnitude(data_r["TS"] - data_r["PC"])[0]

    PC, AI, AA = data_r["PC"], data_r["AI"], data_r["AA"]

    x_pos = 0.0189743 + PC[:, 0] * 0.2434 + PC[:, 1] * 0.0558 + AI[:, 0] * 0.2341 + AI2AA * 0.1590
    y_pos = -0.0038791 + AC2AA * - 0.3940 + PC[:, 1] * 0.1732 + AI[:, 0] * 0.1205 + AC2PC * -0.1002
    z_pos = 0.0092629 + PC[:, 2] * 1.0255 + PC[:, 1] * -0.2403 + TS2PC * 0.1720

    gh_location = np.stack([x_pos, y_pos, z_pos], axis=1)
    gh_global = origin + np.einsum('ijk,ik->ij', acs, gh_location)

    return gh_global
