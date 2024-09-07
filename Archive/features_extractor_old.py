import numpy as np
import pandas as pd


def extract_time_features(segmented_signal):
    column_names = ['Standard deviation', 'Mean', 'Peak-to-peak factor', 'RMS', 'Crest factor', 'Kurtosis',
                    'Skewness']

    std_values = []
    mean_values = []
    pp_values = []
    rms_values = []
    crest_values = []
    kurtosis_values = []
    skew_values = []
    calculated_time_stats = pd.DataFrame()

    for j in range(np.shape(segmented_signal)[0]):  # for each row in array which represents a signal segment
        segment = segmented_signal[j]

        std = np.std(segment)  # Standard Deviation
        std_values.append(std)

        mean_signal = np.mean(segment)  # Mean
        mean_values.append(mean_signal)

        pp = np.max(segment) - np.min(segment)  # Peak-to-peak factor
        pp_values.append(pp)

        rms = np.sqrt(np.mean(segment ** 2))  # RMS
        rms_values.append(rms)

        crest = np.max(segment) / rms  # Crest factor
        crest_values.append(crest)

        kurtosis = pd.Series(segment).kurt()  # Kurtosis
        kurtosis_values.append(kurtosis)

        skew = pd.Series(segment).skew()  # Skewness
        skew_values.append(skew)

    calculated_time_stats = pd.DataFrame(list(zip(std_values, mean_values, pp_values, 
                                                  rms_values, crest_values, kurtosis_values, skew_values)), 
                                                  columns=column_names)

    return calculated_time_stats


# def extract_frequency_features(data) -> pd.DataFrame:
#     column_names = ['Peak frequency', 'Peak amplitude', 'RMS low-freq (below 65 Hz)', 'RMS mid-freq (65-300 Hz)',
#                     'RMS high-freq (above 300 Hz)', 'RMS overall']
#     peak_freq_values = []
#     peak_amplitude_values = []
#     rms_low_freq_values = []
#     rms_mid_freq_values = []
#     rms_high_freq_values = []
#     rms_overall_values = []

#     for i in range(np.shape(data)[0]):  # for each row in array which represents a signal segment
#         segment = data[i]

#         peak_freq =   # Peak frequency
#         peak_freq_values.append(peak_freq)

#         peak_amplitude = # Peak amplitude
#         peak_amplitude_values.append(peak_amplitude)

#         rms_low_freq =   # RMS in the low-frequency range (below 65 Hz)
#         rms_low_freq_values.append(rms_low_freq)

#         rms_mid_freq =   #  RMS in the mid-frequency range (between 65 Hz and 300 Hz)
#         rms_mid_freq_values.append(rms_mid_freq)

#         rms_high_freq =   # RMS in the high-frequency range (above 300 Hz)
#         rms_high_freq_values.append(rms_high_freq)

#         rms_overall =   # Overall RMS
#         rms_overall_values.append(rms_overall)

#         calculated_freq_stats = pd.DataFrame(list(zip(peak_freq_values, peak_amplitude_values, rms_low_freq_values,
#                                                  rms_mid_freq_values, rms_high_freq_values, rms_overall_values)),
#                                         columns=column_names)
#         return calculated_freq_stats
