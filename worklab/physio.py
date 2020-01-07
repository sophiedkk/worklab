def get_spirometer_units():
    """
    Mapping of DataFrame keys to units for spirometer data

    Returns
    -------
    mapping : dict
        table with column names mapped to units
    """
    return {"time": "s", "HR": "b/min", "power": "W", "VO2": "ml/min", "VCO2": "ml/min", "weights": ""}
