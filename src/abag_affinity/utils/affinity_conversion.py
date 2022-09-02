import numpy as np


gas_constant =  8.31446261815324 # 0.0821 kcal

def calc_delta_g(temperature, affinity):
    delta_g = gas_constant * temperature * np.log(affinity)
    return delta_g / 4184 # convert to kcal

def clean_temp(value):
    value = value.replace("(assumed)", "")
    try:
        return int(value)
    except:
        return np.nan