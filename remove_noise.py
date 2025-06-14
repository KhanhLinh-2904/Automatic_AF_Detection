import numpy as np
from scipy.signal import butter, filtfilt, hilbert, savgol_filter
from torch import trapezoid
import wfdb
import scipy.signal as signal
import torch
import os 

def butter_lowpass(cutoff, fs, order=4):
    nyquist = 0.5 * fs
    norm_cutoff = cutoff / nyquist
    b, a = butter(order, norm_cutoff, btype='low', analog=False)
    return b, a

def lowpass_filter(data, cutoff, fs, order=4):
    b, a = butter_lowpass(cutoff, fs, order)
    return filtfilt(b, a, data)

def complex_demodulation(x, fs, f_inst):
    phase_inst = 2 * np.pi * np.cumsum(f_inst) / fs
    exp_term = np.exp(-1j * phase_inst)
    return x * exp_term

def extract_instantaneous_features(z, fs, cutoff=10):
    z_filtered = lowpass_filter(np.real(z), cutoff, fs) + 1j * lowpass_filter(np.imag(z), cutoff, fs)
    amplitude = np.abs(z_filtered)
    phase = np.unwrap(np.angle(z_filtered))
    inst_freq = np.gradient(phase) * fs / (2 * np.pi)
    return amplitude, inst_freq

def compute_vfcdm_dpa(ecg_signal, fs, f_range=(0.5, 20), num_freqs=100, lp_cutoff=10):
    ecg_signal = ecg_signal - np.mean(ecg_signal)
    duration = len(ecg_signal) / fs
    t = np.linspace(0, duration, len(ecg_signal))
    n = len(ecg_signal)
    freqs = np.linspace(f_range[0], f_range[1], num_freqs)
    tfs_matrix = np.zeros((num_freqs, n))

    for i, f_c in enumerate(freqs):
        f_inst = np.ones(n) * f_c
        z = complex_demodulation(ecg_signal, fs, f_inst)
        amp, inst_freq = extract_instantaneous_features(z, fs, cutoff=lp_cutoff)
        tfs_matrix[i, :] = amp

    dpa = np.max(tfs_matrix, axis=0)
    return dpa, t, amp, inst_freq, tfs_matrix

def compute_var_int_labels(dpa, sampling_rate=125, segment_duration=2, th_var=4.0, th_int=500.0, threshold_dpa=1.5):
    segment_length = segment_duration * sampling_rate
    segment = dpa[:segment_length]

    var = np.var(segment)
    segment_thresholded = np.where(segment > threshold_dpa, segment, 0)
    # print("segment_thresholded: ", segment_thresholded.shape)
    time = np.linspace(0, segment_duration, segment_length)
    segment_thresholded_tensor = torch.tensor(segment_thresholded)
    time_tensor = torch.tensor(time)
    integral = torch.trapezoid(segment_thresholded_tensor, time_tensor)
    if var >= th_var and integral >= th_int:
        return 1, var, integral  # Corrupted
    elif var <= th_var and integral <= th_int:
        return 0, var, integral  # Clean
    else:
        return 0.5, var, integral  # Moderately corrupted

def detect_hf_noise(second_component, tfs_matrix, fs, th_sdpa=240, th_var=0.08):
    freq_range = np.linspace(0, fs / 2, tfs_matrix.shape[0])
    high_freq_idx = freq_range > 10
    sdpa = np.sum(np.max(tfs_matrix[high_freq_idx, :], axis=0))
    var_second = np.var(second_component)
    return (sdpa > th_sdpa) and (var_second < th_var)

def discard_noise(segments, fs=125):
    """
    Receives a 2-minutes ECG segment and returns it if clean, else returns None.
    """

    # Compute VFCDM and DPA
    segment_length = 2*fs
    ecg_segments = segments.reshape((-1,segment_length))
    # print("ecg_segments: ", ecg_segments.shape)
    clean_segments = []
    for ecg_segment in ecg_segments:
        dpa, _, amp, _, tfs_matrix = compute_vfcdm_dpa(ecg_segment, fs)
        
        # Compute VAR and INT label
        label, var, integral = compute_var_int_labels(dpa, sampling_rate=fs)

        hf_noise = detect_hf_noise(ecg_segment, tfs_matrix, fs)

        # Final decision
        if label == 0 and not hf_noise:
            clean_segments.append(ecg_segment) # Clean
    return clean_segments

if __name__ == "__main__":
    datapath = "checkECG/AF/"
    save_clean_ECG = "cleanECG/AF/"
    record = os.listdir(datapath)
    for segment_record in record:
        folder_path = os.path.join(datapath, segment_record)
        data = np.load(folder_path)
        print("record: ", segment_record)
        ecg_segments = data["valid_segments"]
        fs = 125
        segment_length = ecg_segments.shape[0]
        clean_segments = []
        # print("segment_length: ", segment_length)
        for i in range(segment_length):
            segment = ecg_segments[i]
            # print("segment: ", len(segment))
            result = discard_noise(segment, fs)
            if result is not None:
                clean_segments.extend(result)
        clean_segments = np.array(clean_segments)
        clean_segments = clean_segments.reshape((1,-1))
        name = save_clean_ECG + segment_record
        np.savez(name,
         clean_segments=clean_segments,
         )
        print(f"Number of clean 2-second segments: {clean_segments.shape}")
