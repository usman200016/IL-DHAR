import numpy as np

def swap_central_parts(vector1, vector2):
    n = len(vector1)
    split_point = n // 3

    new_vector = np.concatenate([vector1[:split_point], vector2[split_point:2*split_point], vector1[2*split_point:]])
    return new_vector

def genetic_algorithm(feature_vectors):
    num_vectors = len(feature_vectors)
    new_generation = []
    np.random.shuffle(feature_vectors)

    for i in range(num_vectors):
        vector1 = feature_vectors[i]
        remaining_vectors = feature_vectors[:i] + feature_vectors[i+1:]
        vector2 = np.random.choice(remaining_vectors)
        new_vector = swap_central_parts(vector1, vector2)
        new_generation.append(new_vector)

    return new_generation