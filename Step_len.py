import numpy as np

def find_peaks(accelerometer_data):
    peaks = []
    for i in range(1, len(accelerometer_data) - 1):
        if accelerometer_data[i] > accelerometer_data[i - 1] and accelerometer_data[i] > accelerometer_data[i + 1]:
            peaks.append(i)
    return peaks

def find_valleys(accelerometer_data, peaks):
    left_valleys = []
    right_valleys = []
    for peak in peaks:
        left_valley = np.argmin(accelerometer_data[peak::-1])  # Search left side (reversed) for minimum value
        right_valley = np.argmin(accelerometer_data[peak:]) + peak  # Search right side for minimum value
        left_valleys.append(left_valley)
        right_valleys.append(right_valley)
    return left_valleys, right_valleys

def calculate_step_lengths(timestamps, left_valleys, right_valleys):
    step_lengths = []
    for left, right in zip(left_valleys, right_valleys):
        step_length = timestamps[right] - timestamps[left]
        step_lengths.append(step_length)
    return step_lengths