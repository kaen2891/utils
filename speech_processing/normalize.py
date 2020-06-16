import librosa
import numpy as np

def instance_standanrd_normalize(spec):
    '''
    The input must magnitude of spectrum. Not mel-spectrogram

    Input dim: (F, T)
    Output dim: (F, T)

    '''
    spec = np.log1p(spec)
    
    mean = np.mean(spec)
    std = np.std(spec)
    spec -= mean
    spec /= std

    return spec

def log_normalize(mel_spec):
    '''
    The input must mel-spectrogram.

    Input dim: (F, T)
    Output dim: (F, T)
    '''

    mel_dim = mel_spec.shape[0]
    time = mel_spec.shape[-1]
    log_S = ma.log10(mel_spec)
    log_S = log_S.filled(0)

    for i in range(mel_dim):
        dim = log_S[i]
        dim_sum = dim.sum()
        average_dim_for_time = dim_sum / time
        normalize_dim = dim - average_dim_for_time

        mel_spec[i] = normalize_dim
        
    return mel_spec

def power_norm(mel_spec):
    '''
    The input must mel-spectrogram.

    Input dim: (F, T)
    Output dim: (F, T)
    '''

    mel_dim = mel_spec.shape[0]
    time = mel_spec.shape[-1]

    for i in range(mel_dim):
        dim = mel_spec[i]
        dim_sum = dim.sum()
        average_dim_for_time = dim_sum / time
        normalize_dim = dim / average_dim_for_time

        mel_spec[i] = normalize_dim
    
    return mel_spec

