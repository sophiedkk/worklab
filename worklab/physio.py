"""
-Physiology module-
Description: Basics for working with physiological data. We only have a spirometer in the lab at the moment and this
involves very little processing. Might expand with EMG related function at some point in the future.
Author:     Rick de Klerk
Contact:    r.de.klerk@umcg.nl
Company:    University Medical Center Groningen
License:    GNU GPLv3.0
Date:       27/06/2019
"""


def get_spirometer_units() -> dict:
    """Mapping of DataFrame keys to units for spirometer data

    :return: dictionary with names and units as key-value pairs
    """
    units_dict = {"time": "s", "Rf": "b/min", "HR": "b/min", "power": "W",
                  "VO2": "ml/min", "VCO2": "ml/min", "weights": ""}
    return units_dict
