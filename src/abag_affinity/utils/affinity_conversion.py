import numpy as np
from typing import Union


gas_constant =  8.31446261815324 # 0.0821 kcal


def calc_delta_g(temperature: float, affinity: float) -> float:
    """ Convert affinity value to delta G value

    Args:
        temperature: temperature of experiment
        affinity: affinity measurement (Kd)

    Returns:
        float: delta g value
    """
    delta_g = gas_constant * temperature * np.log(affinity)
    return delta_g / 4184 # convert to kcal


def clean_temp(value: Union[str, None]) -> float:
    """ Clean the temperature string and convert it to a float

    Args:
        value: String representation of temperature with optional artifacts

    Returns:
        float: temperature or NaN
    """
    try:
        value = value.replace("(assumed)", "")
        return int(value)
    except:
        return np.nan