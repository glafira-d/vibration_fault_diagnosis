import numpy as np
import pandas as pd
from scipy.fft import fft, fftfreq
import scipy.signal as signal


class SignalProcessor:
    def __init__(self):
        pass


    def get_fft(data, sample_points, time_step):
        y = fft(data)  # 1D discrete Fourier transform (DFT) of data
        y =  2/sample_points * np.abs(y[0:sample_points//2]) # multiplying by 2/sample_points to convert to dB for normalization
        x = fftfreq(sample_points, time_step)[:sample_points//2]  # sample frequencies
        return x, y


    def get_psd(data, bin_width, name, sampling_rate):
        f, psd = signal.welch(data, fs=sampling_rate, window='hann', nperseg=sampling_rate/bin_width, axis=0)
        df_psd = pd.DataFrame(psd, columns=[name])
        df_psd['Frequency (Hz)'] = f
        df_psd = df_psd.set_index('Frequency (Hz)')
        return df_psd
