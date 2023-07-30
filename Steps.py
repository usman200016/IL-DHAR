import numpy as np

def calculate_magnitude(acc_x, acc_y, acc_z):
    return np.sqrt(acc_x ** 2 + acc_y ** 2 + acc_z ** 2)

def count_steps(accelerometer_data):
    net_magnitude = accelerometer_data - np.mean(accelerometer_data)
    threshold = np.mean(net_magnitude)
    peaks = np.where(net_magnitude > threshold)[0]
    num_steps = len(peaks)

    return num_steps