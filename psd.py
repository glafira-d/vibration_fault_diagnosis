import pandas as pd
import scipy.signal as signal


def get_psd(data, bin_width, name, sampling_rate):
    f, psd = signal.welch(data, fs=sampling_rate, window='hann', nperseg=sampling_rate/bin_width, axis=0)
    df_psd = pd.DataFrame(psd, columns=[name])
    df_psd['Frequency (Hz)'] = f
    df_psd = df_psd.set_index('Frequency (Hz)')
    return df_psd
