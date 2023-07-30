import numpy as np

def embed_time_series(signal, embedding_dimension, time_delay):
    num_samples = len(signal)
    num_vectors = num_samples - (embedding_dimension - 1) * time_delay
    embedded_space = np.zeros((num_vectors, embedding_dimension))
    for i in range(num_vectors):
        embedded_space[i] = signal[i:i + embedding_dimension * time_delay:time_delay]
    return embedded_space

def euclidean_distance(x, y):
    return np.sqrt(np.sum((x - y) ** 2))

def estimate_lyapunov_exponent(signal, embedding_dimension, time_delay, num_neighbors=10, max_num_vectors=1000):
    embedded_space = embed_time_series(signal, embedding_dimension, time_delay)

    num_vectors = min(max_num_vectors, len(embedded_space))
    indices = np.random.choice(len(embedded_space), num_vectors, replace=False)

    lyapunov_sum = 0.0

    for idx in indices:
        reference_vector = embedded_space[idx]

        distances = [euclidean_distance(reference_vector, v) for v in embedded_space]
        sorted_indices = np.argsort(distances)[1:num_neighbors + 1]  # Skip the first index (itself)

        for neighbor_idx in sorted_indices:
            neighbor_vector = embedded_space[neighbor_idx]
            distance = euclidean_distance(reference_vector, neighbor_vector)

            if distance > 1e-6:
                lyapunov_sum += np.log(distance / euclidean_distance(reference_vector + 1e-6, neighbor_vector + 1e-6))

    return lyapunov_sum / (num_vectors * num_neighbors)