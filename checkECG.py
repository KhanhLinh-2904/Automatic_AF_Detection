import numpy as np
import wfdb
from scipy.signal import welch, find_peaks
from scipy.stats import skew
import os
def divide_into_segments(ecg_signal, fs=125):
    """
    Divide ECG signal into 2-minute segments.

    Args:
        ecg_signal (np.ndarray): 1D ECG signal array.
        fs (int): Sampling frequency in Hz (default: 250 Hz).

    Returns:
        List[np.ndarray]: List of 2-minute segments.
    """
    segment_len = 2 * 60 * fs  # 2 minutes in samples
    total_len = len(ecg_signal)
    # print("segment_len: ", segment_len)
    # print("total_len: ", total_len)
    
    modulus = total_len % segment_len
    
    new_segment = ecg_signal[:(total_len-modulus)]
    new_segment = new_segment.reshape((-1,segment_len))
    # print("new_segment: ", new_segment.shape)
    return new_segment

def filter_zero(segments, fs=125):
    """
    Filters ECG segments based on:
    1. Removing segments containing NaNs.
    2. Removing segments with ≥400ms (e.g., 50 samples) of consecutive zeros.

    Args:
        segments (np.ndarray): Array of shape (N, L), N segments of L-length.
        fs (int): Sampling frequency in Hz (default 125 Hz).

    Returns:
        np.ndarray: Filtered segments.
    """
    valid_segments = []
    zero_threshold = int(0.4 * fs)  # 400ms in samples

    for segment in segments:
        # Check for NaN values
        if np.isnan(segment).any():
            continue

        # Check for ≥ zero_threshold consecutive zeros
        zero_run = 0
        max_zero_run = 0
        for val in segment:
            if val == 0:
                zero_run += 1
                max_zero_run = max(max_zero_run, zero_run)
            else:
                zero_run = 0

        if max_zero_run >= zero_threshold:
            continue

        valid_segments.append(segment)
    return np.array(valid_segments)

def filter_low_variance(segments, fs=125, piece_duration_sec=1, piecewise_var_threshold=1e-4, total_var_threshold=1e-4):
    """
    Filters out segments with:
    1. Very low total variance (likely flat signal).
    2. No short-term variance spikes (suggests low amplitude, non-ECG signals).

    Args:
        segments (np.ndarray): Input ECG segments of shape (N, L).
        fs (int): Sampling frequency in Hz (default=125).
        piece_duration_sec (int): Duration (in sec) for piecewise variance computation.
        piecewise_var_threshold (float): Threshold for piecewise variance.
        total_var_threshold (float): Threshold for overall segment variance.

    Returns:
        np.ndarray: Filtered ECG segments.
    """
    # print("**************************",segments.shape)
    
    segment_length = segments.shape[1]
    piece_length = piece_duration_sec * fs
    valid_segments = []

    for segment in segments:
        # Step 1: Check overall segment variance
        if np.var(segment) < total_var_threshold:
            continue

        # Step 2: Compute piecewise variance for each 1-second chunk
        modulus = segment_length % piece_length
        piece_segment = segment[:(segment_length-modulus)]
        piece_segment = piece_segment.reshape((-1,piece_length))
        
        piecewise_vars = [
            np.var(piece_segment[i,:])
            for i in range(piece_segment.shape[0])
        ]
        # print("piecewise_vars: ",piecewise_vars)

        # If all 1-sec chunks have very low variance, discard
        if all(pv < piecewise_var_threshold for pv in piecewise_vars):
            continue

        valid_segments.append(segment)

    return np.array(valid_segments)

def filter_psd_analysis(segments, fs=125, psd_peak_threshold=1e6):
    """
    Filters ECG segments using Power Spectral Density (PSD) analysis.

    Args:
        segments (np.ndarray): Input ECG segments of shape (N, L).
        fs (int): Sampling frequency in Hz.
        psd_peak_threshold (float): Threshold for low-frequency power.

    Returns:
        np.ndarray: Filtered ECG segments.
    """
    valid_segments = []

    for segment in segments:
        # Compute PSD using Welch's method
        freqs, psd = welch(segment, fs=fs, nperseg=1024)

        # Restrict analysis to 0-5 Hz
        mask_0_5Hz = (freqs >= 0) & (freqs <= 5.0)
        freqs_0_5Hz = freqs[mask_0_5Hz]
        psd_0_5Hz = psd[mask_0_5Hz]
        # print("freqs_0_5Hz: ", freqs_0_5Hz)
        # print("psd_0_5Hz: ", psd_0_5Hz)
        
        # Find all peaks in 0-5 Hz
        peaks, _ = find_peaks(psd_0_5Hz)
        peak_freqs = freqs_0_5Hz[peaks]
        peak_powers = psd_0_5Hz[peaks]
        # print("peak_freqs: ", peak_freqs)
        # print("peak_powers: ", peak_powers)
        

        # Check for any peaks in the 2.5-5 Hz range
        has_harmonic_peak = np.any((peak_freqs >= 2.5) & (peak_freqs <= 5.0))

        if has_harmonic_peak:
            valid_segments.append(segment)
        else:
            # Sum peak powers in 0-2.5 Hz range
            low_band_mask = (peak_freqs >= 0) & (peak_freqs < 2.5)
            low_band_power = np.sum(peak_powers[low_band_mask])

            if low_band_power <= psd_peak_threshold:
                valid_segments.append(segment)

    return np.array(valid_segments)

def detect_and_normalize_ecg_segments(segments):
    """
    Detects inverted ECG waveforms and normalizes the segments.

    Args:
        segments (np.ndarray): Input ECG segments of shape (N, L).

    Returns:
        np.ndarray: Processed ECG segments with inversion correction and normalization.
    """
    processed_segments = []

    for segment in segments:
        # Step 1: Detect inversion using skewness
        segment_skew = skew(segment)
        if segment_skew < 0:
            # Likely inverted: flip the signal
            segment = -segment

        # Step 2: Normalize the segment: zero mean, unit variance
        mean = np.mean(segment)
        std = np.std(segment)
        if std == 0:
            # Prevent division by zero
            continue
        segment = (segment - mean) / std

        processed_segments.append(segment)

    return np.array(processed_segments)

if __name__ =="__main__":
    
    datapath = "mimic_perform_af_wfdb/"
    record_files = [f.split('.')[0] for f in os.listdir(datapath) if f.endswith('.dat')]
    for record_name in record_files:
        record_path = os.path.join(datapath, record_name )   
        record = wfdb.rdrecord(record_path)
        fs = record.fs
        num_record = record.p_signal.shape[1]
        print("num_record: ", num_record)
        valid_segments = []
        for i in range(num_record):
            ecg_signal = record.p_signal[:, i]
            segments = divide_into_segments(ecg_signal, fs)
            # print("Number of original segments: ", len(segments))
            if len(segments) == 0:
                continue
            non_0_segments = filter_zero(segments, fs)
            if len(non_0_segments) == 0:
                continue
            # print("Number of 0 segments: ", segments.shape[0] - non_0_segments.shape[0])
            
            high_var_segments = filter_low_variance(non_0_segments)
            # print("Number of low var segments: ", non_0_segments.shape[0] - high_var_segments.shape[0])
            if len(high_var_segments) == 0:
                continue
            psd_segments = filter_psd_analysis(high_var_segments)
            # print("Number of under psd criterior segments: ", high_var_segments.shape[0] - psd_segments.shape[0])
            if len(psd_segments) == 0:
                continue
            valid_segment = detect_and_normalize_ecg_segments(psd_segments)
            if len(valid_segment) == 0:
                continue
            print("shape valid_segment: ", valid_segment.shape)
            valid_segments.append(valid_segment)
        if len(valid_segments) == 0:
            continue
        print("valid_segments: ", len(valid_segments))
        # Combine into one array
        valid_segments = np.concatenate(valid_segments, axis=0)
        valid_segments = np.array(valid_segments)    
        print("valid_segments: ", valid_segments.shape[0])
        
        name = "checkECG/AF/" + record_name +".npz"
        np.savez(name,
            valid_segments=valid_segments,
        )
        