import numpy as np

def rootMeanSquareSuccessiveDifferences(signal):
    signal = np.array(signal, dtype=float)
    signal_length = len(signal)
    diff = np.diff(signal)
    rmssd = np.sqrt(np.sum(diff**2) / (signal_length - 1))
    return rmssd

