from enum import Enum
import math
import numpy as np
import pandas as pd
from scipy import stats

"""
References:
https://www.mdpi.com/2079-9292/12/18/3971#sec3dot2dot1-electronics-12-03971
"""

class FrequencyFeatureExtractor:
    def __init__(self):
        self.feature_names = [
            'peak_frequency',
            'peak_amplitude',
            'rms_low_frequency',
            'rms_mid_frequency',
            'rms_high_frequency',
            'rms_overall'
            ]

        self.feature_calculators = [
            self._calculate_peak_frequency,
            self._calculate_peak_amplitude,
            self._calculate_rms_low_freq,
            self._calculate_rms_mid_freq,
            self._calculate_rms_high_freq,
            self._calculate_rms_overall
        ]


    def _calculate_peak_amplitude(self, freq_domain_segment):
        """Calculate the peak amplitude in the frequency domain."""
        return freq_domain_segment.max()


    def _calculate_peak_frequency(self, freq_domain_segment):
        """Calculate the frequency at the peak amplitude in the frequency domain."""
        return freq_domain_segment.idxmax()
    

    def _calculate_rms_low_freq(self, freq_domain_segment):
        """Calculate the RMS in the low-frequency range (below 65 Hz)."""
        low_freq_squares = freq_domain_segment[:65].pow(2)
        low_freq_mean = low_freq_squares.mean()
        low_freq_rms = math.sqrt(low_freq_mean)
        return low_freq_rms

    
    def _calculate_rms_mid_freq(self, freq_domain_segment):
        """Calculate the RMS in the mid-frequency range (between 65 Hz and 300 Hz)."""
        mid_freq_squares = freq_domain_segment[65:300].pow(2)
        mid_freq_mean = mid_freq_squares.mean()
        mid_freq_rms = math.sqrt(mid_freq_mean)
        return mid_freq_rms

    
    def _calculate_rms_high_freq(self, freq_domain_segment):
        """Calculate the RMS in the high-frequency range (above 300 Hz)."""
        high_freq_squares = freq_domain_segment[300:].pow(2)
        high_freq_mean = high_freq_squares.mean()
        high_freq_rms = math.sqrt(high_freq_mean)
        return high_freq_rms
    

    def _calculate_rms_overall(self, freq_domain_segment):
        """Calculate the RMS of the entire frequency range."""
        freq_domain_squares = freq_domain_segment.pow(2)
        freq_domain_mean = freq_domain_squares.mean()
        rms = math.sqrt(freq_domain_mean)
        return rms
    

    def extract_features(self, psd_data):
        """Extract frequency domain features from power spectral density data."""
        feature_values = []
        
        for name, column in psd_data.iteritems(): # for each column in dataframe that corresponds to one segment
            segment = column # pd.Series containing PSD values with Frequency as index
            features = [calculator(segment) for calculator in self.feature_calculators]
            feature_values.append(features)

        return pd.DataFrame(feature_values, columns=self.feature_names)


class TimeFeatureExtractor:
    def __init__(self):
        self.feature_names = [
            'Standard deviation', 
            'Mean', 
            'Peak-to-peak factor', 
            'RMS', 
            'Crest factor', 
            'Kurtosis', 
            'Skewness'
            ]

        self.feature_calculators = [
            self._calculate_std,
            self._calculate_mean,
            self._calculate_pp_factor,
            self._calculate_rms,
            self._calculate_crest_factor,
            self._calculate_kurtosis,
            self._calculate_skewness
        ]


    def _calculate_std(self, segment):
        return np.std(segment)


    def _calculate_mean(self, segment):
        return np.mean(segment)


    def _calculate_pp_factor(self, segment):
        return np.max(segment) - np.min(segment)


    def _calculate_rms(self, segment):
        return np.sqrt(np.mean(segment ** 2))


    def _calculate_crest_factor(self, segment):
        return np.max(segment) / self._calculate_rms(segment)


    def _calculate_kurtosis(self, segment):
        return stats.kurtosis(segment)


    def _calculate_skewness(self, segment):
        return stats.skew(segment)


    def extract_features(self, segmented_signal):
        """Extracts time domain features from segmented vibration signal.

        Parameters:
            segmented_signal (list): A list of signal segments.

        Returns:
            pandas.DataFrame: A DataFrame containing the extracted features.
        """
        feature_values = []

        for j in range(len(segmented_signal)):
            segment = segmented_signal[j]
            features = [calculator(segment) for calculator in self.feature_calculators]
            feature_values.append(features)

        return pd.DataFrame(feature_values, columns=self.feature_names)