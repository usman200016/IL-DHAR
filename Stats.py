import numpy as np
from scipy.stats import skew, kurtosis

def calculate_skewness(signal):
    return skew(signal)

def calculate_kurtosis(signal):
    return kurtosis(signal)
