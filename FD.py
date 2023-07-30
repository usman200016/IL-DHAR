import numpy as np

def calculate_fractal_dimension(signal, box_size_ratio=0.1):
    n = len(signal)
    box_size = int(n * box_size_ratio)
    num_boxes = n // box_size
    count_occupied_boxes = 0
    for i in range(num_boxes):
        start_idx = i * box_size
        end_idx = min(start_idx + box_size, n)
        if np.any(signal[start_idx:end_idx] != 0):
            count_occupied_boxes += 1
    fractal_dimension = -np.log(count_occupied_boxes) / np.log(1.0 / box_size_ratio)

    return fractal_dimension
