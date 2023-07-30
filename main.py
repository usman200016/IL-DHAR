import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from Preprocessing import  butterworth_filter
from Windowing import window_and_stack
from ED import false_nearest_neighbors
from MLE import estimate_lyapunov_exponent
from MFCC import extract_mfcc
from Stats import calculate_skewness
from Stats import calculate_kurtosis
from FD import calculate_fractal_dimension
from Steps import count_steps
from Step_len import calculate_step_lengths, find_peaks, find_valleys
from Heading import calculate_heading_direction
from feat_select import recursive_feature_elimination
from GA import genetic_algorithm
from DNDF import build_dndf, train_dndf
from sklearn.model_selection import train_test_split
import pickle 


def main(timestamps, acc, gyr, mag, mic, gps, act_classes, loc_classes):
    filtered_acc = butterworth_filter(acc, cutoff_frequency=0.001)
    filtered_gyr = butterworth_filter(gyr, cutoff_frequency=0.001)
    filtered_mag = butterworth_filter(mag, cutoff_frequency=0.001)
    filtered_mic = butterworth_filter(mic, cutoff_frequency=0.001)
    filtered_gps = butterworth_filter(gps, cutoff_frequency=0.001)

    windowed_stacked_acc = window_and_stack(filtered_acc, window_size=5, overlap=0)
    windowed_stacked_gyr = window_and_stack(filtered_gyr, window_size=5, overlap=0)
    windowed_stacked_mag = window_and_stack(filtered_mag, window_size=5, overlap=0)
    windowed_stacked_mic = window_and_stack(filtered_mic, window_size=5, overlap=0)
    windowed_stacked_gps = window_and_stack(filtered_gps, window_size=5, overlap=0)

    embedding_dimension_acc = false_nearest_neighbors(windowed_stacked_acc)
    embedding_dimension_gyr = false_nearest_neighbors(windowed_stacked_gyr)
    embedding_dimension_mag = false_nearest_neighbors(windowed_stacked_mag)

    lyapunov_exponent_acc = estimate_lyapunov_exponent(windowed_stacked_acc)
    lyapunov_exponent_gyr = estimate_lyapunov_exponent(windowed_stacked_gyr)
    lyapunov_exponent_mag = estimate_lyapunov_exponent(windowed_stacked_mag)

    mfcc_features_acc = extract_mfcc(windowed_stacked_acc)
    mfcc_features_gyr = extract_mfcc(windowed_stacked_gyr)
    mfcc_features_mag = extract_mfcc(windowed_stacked_mag)
    mfcc_features_mic = extract_mfcc(windowed_stacked_mic)

    skewness_acc = calculate_skewness(windowed_stacked_acc)
    skewness_gyr = calculate_skewness(windowed_stacked_gyr)
    skewness_mag = calculate_skewness(windowed_stacked_mag)
    skewness_mic = calculate_skewness(windowed_stacked_mic)
    skewness_gps = calculate_skewness(windowed_stacked_gps)

    kurtosis_acc = calculate_kurtosis(windowed_stacked_acc)
    kurtosis_gyr = calculate_kurtosis(windowed_stacked_gyr)
    kurtosis_mag = calculate_kurtosis(windowed_stacked_mag)
    kurtosis_mic = calculate_kurtosis(windowed_stacked_mic)
    kurtosis_gps = calculate_kurtosis(windowed_stacked_gps)

    fractal_dimension_acc = calculate_fractal_dimension(windowed_stacked_acc)
    fractal_dimension_gyr = calculate_fractal_dimension(windowed_stacked_gyr)
    fractal_dimension_mag = calculate_fractal_dimension(windowed_stacked_mag)

    step_count = count_steps(windowed_stacked_acc)

    peaks = find_peaks(windowed_stacked_acc)
    Lval, Rval = find_valleys(windowed_stacked_acc, peaks)
    step_lengths = calculate_step_lengths(timestamps, Lval, Rval)

    heading_direction = calculate_heading_direction(acc[0], acc[1], acc[2], 
                                                    mag[0], mag[1], mag[2], 
                                                    gyr[0], gyr[1], gyr[2] 
                                                    )

    act_feats = np.concatenate((embedding_dimension_acc, 
                                embedding_dimension_gyr, 
                                embedding_dimension_mag,
                                lyapunov_exponent_acc,
                                lyapunov_exponent_gyr,
                                lyapunov_exponent_mag,
                                mfcc_features_acc,
                                mfcc_features_gyr,
                                mfcc_features_mag,
                                skewness_acc,
                                skewness_gyr,
                                skewness_mag,
                                kurtosis_acc,
                                kurtosis_gyr,
                                kurtosis_mag,
                                fractal_dimension_acc,
                                fractal_dimension_gyr,
                                fractal_dimension_mag
                                ))
    
    loc_feats = np.concatenate((mfcc_features_mic,
                                skewness_mic,
                                skewness_gps,
                                kurtosis_mic,
                                kurtosis_gps,
                                step_count,
                                step_lengths,
                                heading_direction
                                ))

    selected_features_act = recursive_feature_elimination(act_feats, len(act_feats)-1)
    selected_features_loc = recursive_feature_elimination(loc_feats, len(loc_feats)-1)

    augmented_data_act = genetic_algorithm(selected_features_act)
    augmented_data_loc = genetic_algorithm(selected_features_loc)

    input_shape_act = augmented_data_act.shape[1:]
    input_shape_loc = augmented_data_loc.shape[1:]

    num_classes_act = input_shape_act
    num_classes_loc = input_shape_loc

    dndf_model_act = build_dndf(num_classes_act, act_classes)
    dndf_model_loc = build_dndf(num_classes_loc, loc_classes)

    X_train_act, X_test_act, y_train_act, y_test_act = train_test_split(augmented_data_act, augmented_data_act[:-1], 
                                                                        test_size=0.2, random_state=42)
    X_train_loc, X_test_loc, y_train_loc, y_test_loc = train_test_split(augmented_data_loc, augmented_data_act[:-1], 
                                                                        test_size=0.2, random_state=42)
    
    model_act = train_dndf(dndf_model_act, X_train_act, y_train_act, X_test_act, y_test_act, batch_size=64, epochs=200)
    model_loc = train_dndf(dndf_model_loc, X_train_loc, y_train_loc, X_test_loc, y_test_loc, batch_size=128, epochs=200)

    pickle.dump(model_act, 'activity.pkl')
    pickle.dump(model_loc, 'location.pkl')




if __name__ == "__main__":
    main()
