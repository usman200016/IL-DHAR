import numpy as np
from scipy.signal import butter, lfilter

def butterworth_filter(signal, cutoff_freq, fs, order=2):
    nyquist_freq = 0.5 * fs
    normal_cutoff = cutoff_freq / nyquist_freq
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    filtered_signal = lfilter(b, a, signal)
    return filtered_signal
