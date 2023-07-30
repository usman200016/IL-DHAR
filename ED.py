import numpy as np

def embed_time_series(signal, embedding_dimension, time_delay):
    num_samples = len(signal)
    num_vectors = num_samples - (embedding_dimension - 1) * time_delay
    embedded_space = np.zeros((num_vectors, embedding_dimension))
    for i in range(num_vectors):
        embedded_space[i] = signal[i:i + embedding_dimension * time_delay:time_delay]
    return embedded_space

def false_nearest_neighbors(signal, max_embedding_dimension=20, time_delay=1, threshold=10.0, tolerance=0.1):
    min_embedding_dimension = 2

    for embedding_dimension in range(min_embedding_dimension, max_embedding_dimension + 1):
        embedded_space = embed_time_series(signal, embedding_dimension, time_delay)
        num_vectors = len(embedded_space)

        total_fnn_count = 0

        for i in range(num_vectors):
            reference_vector = embedded_space[i]
            distances = [np.linalg.norm(reference_vector - v) for v in embedded_space]
            sorted_indices = np.argsort(distances)[1:]

            for k in range(1, embedding_dimension + 1):
                ratio = distances[sorted_indices[k + 1]] / distances[sorted_indices[k]]
                if abs(ratio - 1.0) > tolerance:
                    total_fnn_count += 1

        fnn_ratio = total_fnn_count / (num_vectors * embedding_dimension)

        if fnn_ratio < threshold:
            return embedding_dimension

    return max_embedding_dimension