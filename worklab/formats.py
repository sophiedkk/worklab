"""
-Common data structures-
Description: Contains common data structures for wheelchair ergometer
and measurement wheels. Mostly dictionaries. Might become classes at
some point in time.
Author:     Rick de Klerk
Contact:    r.de.klerk@umcg.nl
Company:    University Medical Center Groningen
License:    GNU GPLv3.0
Date:       26/03/2018
"""

import copy


class Kinetics(object):
    """"Dataclass with behavior of nested dict, but can also store some additional data"""
    def __init__(self, filename="", wheelsize=0.31, rimsize=0.27, sfreq=100, data=None):
        self.filename = filename
        self.wheelsize = wheelsize
        self.handrimsize = rimsize
        self.samplefreq = sfreq
        self.rawdata = data
        self.data = copy.deepcopy(self.rawdata)
        self.pbp = None
        self.summary = None

    def __repr__(self):
        return f'{self.__class__.__name__} data object for file: {self.filename!r}'

    def __iter__(self):
        return self.data.__iter__()

    def __getitem__(self, index):
        return self.data[index]

    def reset(self):
        self.data = copy.deepcopy(self.rawdata)

    def set_variables(self, wheelsize=0.31, rimsize=0.27, sfreq=100):
        self.wheelsize = wheelsize
        self.handrimsize = rimsize
        self.samplefreq = sfreq


def get_erg_format():
    """ Ergometer only measures speed and force on the roller/wheel"""
    datadict = {"left": {"time":    [],  # [s]
                         "speed":   [],  # [m/s]
                         "force":   [],  # [N] @wheel
                         },
                "right": {"time":   [],  # [s]
                          "speed":  [],  # [m/s]
                          "force":  [],  # [N] @wheel
                          },
                }
    return datadict


def get_lem_format():
    """ Ergometer only measures speed and force on the roller/wheel"""
    datadict = {"left": {"time":    [],  # [s]
                         "speed":   [],  # [m/s]
                         "uforce":  [],  # [N] @wheel
                         },
                "right": {"time":   [],  # [s]
                          "speed":  [],  # [m/s]
                          "uforce": [],  # [N] @wheel
                          },
                }
    return datadict


def get_spline_format():
    datadict = {"left":     {"time":    [],  # [s]
                             "speed":   [],  # [m/s]
                             "force":   [],  # [N] @wheel
                             },
                "right":    {"time":    [],  # [s]
                             "speed":   [],  # [m/s]
                             "force":   [],  # [N] @wheel
                             },
                }
    return datadict


def get_mw_format():
    """ For Optipush and SMARTwheel data, contains 3D forces and torques
        Named mz 'torque' to match ergometer data format"""
    datadict = {"time":     [],  # [s]
                "angle":    [],  # [rad]
                "fx":       [],  # [N]
                "fy":       [],  # [N]
                "fz":       [],  # [N]
                "mx":       [],  # [Nm]
                "my":       [],  # [Nm]
                "torque":   [],  # [Nm]
                }
    return datadict


def get_pbp_format():
    pbp_format = {"start":      [],  # [#] sample nr
                  "stop":       [],  # [#] sample nr
                  "tstart":     [],  # [s] time
                  "tstop":      [],  # [s] time
                  "ptime":      [],  # [s] push time
                  "ctime":      [],  # [s] cycle time
                  "reltime":    [],  # [ ] relative push to cycle
                  "pout":       [],  # [W] power output
                  "maxpout":    [],  # [W] peak power output
                  "maxtorque":  [],  # [Nm] max torque
                  "meantorque": [],  # [Nm] mean torque
                  "slope":      [],  # [N/s] slope of upward part
                  "cangle":     [],  # [rad] contact angle
                  "work":       [],  # [J] work
                  "fpeak":      [],  # [N] peak effective force
                  "fmean":      [],  # [N] mean effective force
                  }
    return pbp_format


def get_sum_format():
    """ Contains the required dict for all summary statistics"""
    summary = {"ptime":      [],  # [s] push time
               "ctime":      [],  # [s] cycle time
               "reltime":    [],  # [ ] relative push to cycle
               "pout":       [],  # [W] power output
               "maxpout":    [],  # [W] peak power output
               "maxtorque":  [],  # [Nm] max torque
               "slope":      [],  # [N/s] slope of upward part
               "cangle":     [],  # [rad] contact angle
               "work":       [],  # [J] work
               "fpeak":      [],  # [N] peak effective force
               "fmean":      [],  # [N] mean effective force
               }
    return summary


def get_names_units():  # for plot module
    """ Contains the names and SI units of all data structures matched by dictionary key"""
    unit_dict = {"time":     ["Time", "s"],
                 "angle":    ["Angle", "rad"],
                 "dist":     ["Distance", "m"],
                 "aspeed":   ["Angular velocity", "rad/s"],
                 "speed":    ["Linear velocity", "m/s"],
                 "acc":      ["Linear acceleration", "m/s"],
                 "fx":       ["Force x-component", "N"],
                 "fy":       ["Force y-component", "N"],
                 "fz":       ["Force z-component", "N"],
                 "ftot":     ["Total force", "N"],
                 "feff":     ["Effective force", "N"],
                 "force":    ["Force at wheel", "N"],
                 "uforce":   ["Force at handrim", "N"],
                 "mx":       ["Torque x-component", "Nm"],
                 "my":       ["Torque y-component", "Nm"],
                 "torque":   ["Effective torque", "Nm"],
                 "power":    ["Power output", "W"],
                 "work":     ["Work", "J"],
                 }
    return unit_dict
