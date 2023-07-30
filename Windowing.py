import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import hamming

def window_and_stack(signal, window_size, stack_size, sampling_rate):
    window = hamming(int(window_size * sampling_rate))
    window_size_samples = int(window_size * sampling_rate)
    stack_size_samples = int(stack_size * window_size_samples)
    
    if len(signal) < stack_size_samples:
        print("Error: Signal length is too short to generate any window stack.")
        return []
    
    num_windows = (len(signal) - stack_size_samples) // window_size_samples + 1
    window_stacks = []
    for i in range(num_windows):
        start_idx = i * window_size_samples
        end_idx = start_idx + stack_size_samples
        window_stack = [signal[start_idx + j * window_size_samples: 
                                start_idx + (j + 1) * window_size_samples] * window
                        for j in range(stack_size)]
        window_stacks.append(window_stack)
    return window_stacks
