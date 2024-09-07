import math
import numpy as np
import pandas as pd
from scipy import stats


class FrequencyFeatureExtractor:
    def __init__(self):
        self.column_names = ['Peak frequency', 'Peak amplitude', 'RMS low-freq (below 65 Hz)', 'RMS mid-freq (65-300 Hz)',
                             'RMS high-freq (above 300 Hz)', 'RMS overall']
        self.feature_calculators = [
            self.calculate_peak_freq,
            self.calculate_peak_amplitude,
            self.calculate_rms_low_freq,
            self.calculate_rms_mid_freq,
            self.calculate_rms_high_freq,
            self.calculate_rms_overall
            ]
    
    
    def calculate_peak_amplitude(self, segment):
        return segment.max()


    def calculate_peak_freq(self, segment): # frequency at peak amplitude
        return segment.idxmax()
    

    def calculate_rms_low_freq(self, segment): # RMS in the low-freqency range (below 65 Hz)
        squares = segment[:65].pow(2)
        mean = squares.mean()
        rms = math.sqrt(mean)
        return rms
    

    def calculate_rms_mid_freq(self, segment): # RMS in the mid-frequency range (between 65 Hz and 300 Hz)
        squares = segment[65:300].pow(2)
        mean = squares.mean()
        rms = math.sqrt(mean)
        return rms
    

    def calculate_rms_high_freq(self, segment): # RMS in the high-frequency range (above 300 Hz)
        squares = segment[300:].pow(2)
        mean = squares.mean()
        rms = math.sqrt(mean)
        return rms
    

    def calculate_rms_overall(self, segment):
        squares = segment.pow(2)
        mean = squares.mean()
        rms = math.sqrt(mean)
        return rms
    

    def extract_features(self, psd_data):
        feature_values = []
        for name, column in psd_data.iteritems(): # for each column in dataframe that corresponds to one segment
            segment = column # pd.Series containing PSD values with Frequency as index
            features = [calculator(segment) for calculator in self.feature_calculators]
            feature_values.append(features)
        
        return pd.DataFrame(feature_values, columns=self.column_names)


class TimeFeatureExtractor:
    def __init__(self):
        self.column_names = ['Standard deviation', 'Mean', 'Peak-to-peak factor', 'RMS', 
                             'Crest factor', 'Kurtosis', 'Skewness']
        self.feature_calculators = [
            self.calculate_std,
            self.calculate_mean,
            self.calculate_pp_factor,
            self.calculate_rms,
            self.calculate_crest_factor,
            self.calculate_kurtosis,
            self.calculate_skewness
        ]


    def calculate_std(self, segment):
        return np.std(segment)


    def calculate_mean(self, segment):
        return np.mean(segment)


    def calculate_pp_factor(self, segment):
        return np.max(segment) - np.min(segment)


    def calculate_rms(self, segment):
        return np.sqrt(np.mean(segment ** 2))


    def calculate_crest_factor(self, segment):
        return np.max(segment) / self.calculate_rms(segment)


    def calculate_kurtosis(self, segment):
        return stats.kurtosis(segment)


    def calculate_skewness(self, segment):
        return stats.skew(segment)


    def extract_features(self, segmented_signal):
        feature_values = []
        for j in range(len(segmented_signal)):
            segment = segmented_signal[j]
            features = [calculator(segment) for calculator in self.feature_calculators]
            feature_values.append(features)
        
        return pd.DataFrame(feature_values, columns=self.column_names)