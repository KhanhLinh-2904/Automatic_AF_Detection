import os
import wfdb
from wfdb.processing import xqrs_detect
import numpy as np

from assignLabel import assignLabel

def removeNoise(N_segment_128, N_manual_label, N_auto_label, N_tpr_ratio, N_rmssd, N_se):
    new_N_segment_128 = []
    new_N_manual_label = []
    new_N_auto_label = []
    new_N_tpr_ratio = []
    new_N_rmssd = []
    new_N_se = []
    num_non_AF = len(N_manual_label)
    for i in range(num_non_AF):
        is_tpr =  N_tpr_ratio[i] >= 0.9
        is_rmssd = N_rmssd[i] >= 4
        is_se = N_se[i] >= 0.8
        if is_tpr or is_rmssd or is_se:
            print("True")
            continue
        else:
            new_N_segment_128.append(N_segment_128[i])
            new_N_manual_label.append(N_manual_label[i])
            new_N_auto_label.append(N_auto_label[i])
            new_N_tpr_ratio.append(N_tpr_ratio[i])
            new_N_rmssd.append(N_rmssd[i])
            new_N_se.append(N_se[i])
    return new_N_segment_128, new_N_manual_label, new_N_auto_label, new_N_tpr_ratio, new_N_rmssd, new_N_se

def loadData(datapath,label_value):
    segment_128 = []
    manual_label = []
    fs = 125
    record_files = os.listdir(datapath)
    for record_name in record_files:
        record_path = os.path.join(datapath, record_name )
        data = np.load(record_path)
        signal = data["clean_segments"]
        signal = signal.reshape(-1)
        qrs_indices = xqrs_detect(sig=signal, fs=fs)
        # Find R-peaks (maximum value between two QRS detections)
        max_indices = []

        for i in range(len(qrs_indices) - 1):
            start = qrs_indices[i]
            end = qrs_indices[i + 1]

            if end >= len(signal):
                break

            segment = signal[start:end+1]
            if len(segment) == 0:
                continue

            max_val = np.max(segment)
            for j in range(start, end + 1):
                if signal[j] == max_val:
                    max_indices.append(j)
                    break

        # Extract R-peak amplitudes
        r_peaks = np.array([signal[idx] for idx in max_indices])

        # Compute RR intervals
        RR_intervals = np.diff(max_indices)/fs
        print("RR_intervals: ", RR_intervals)
        # Trim and reshape
        if len(RR_intervals) < 128: 
            continue
        modulus = len(RR_intervals) % 128
        trimmed_rr = RR_intervals[:len(RR_intervals) - modulus]
        reshaped = trimmed_rr.reshape((-1, 128))

        # print("Reshaped RR intervals:", reshaped.shape)
        segment_128.extend(reshaped)
    segment_128 = np.array(segment_128)    
    # print("segment_128:", segment_128.shape)
    manual_label.extend([label_value] * segment_128.shape[0])
    auto_label, tpr_ratio, rmssd, se = assignLabel(segment_128)
    return segment_128, manual_label, auto_label, tpr_ratio, rmssd, se

if __name__ == "__main__":
    AF_datapath = "cleanECG/AF"
    N_datapath = "cleanECG/non_AF"
    segment_128 = []
    manual_label = [] 
    auto_label = [] 
    tpr_ratio = [] 
    rmssd = [] 
    se = []
    AF_segment_128, AF_manual_label, AF_auto_label, AF_tpr_ratio, AF_rmssd, AF_se = loadData(AF_datapath, True)
    N_segment_128, N_manual_label, N_auto_label, N_tpr_ratio, N_rmssd, N_se = loadData(N_datapath, False)
    # N_segment_128, N_manual_label, N_auto_label, N_tpr_ratio, N_rmssd, N_se = removeNoise(N_segment_128, N_manual_label, N_auto_label, N_tpr_ratio, N_rmssd, N_se)
    segment_128.extend(AF_segment_128)
    segment_128.extend(N_segment_128)
    manual_label.extend(AF_manual_label)
    manual_label.extend(N_manual_label)
    auto_label.extend(AF_auto_label)
    auto_label.extend(N_auto_label)
    tpr_ratio.extend(AF_tpr_ratio)
    tpr_ratio.extend(N_tpr_ratio)
    rmssd.extend(AF_rmssd)
    rmssd.extend(N_rmssd)
    se.extend(AF_se)
    se.extend(N_se)
    segment_128 = np.array(segment_128) 
    manual_label = np.array(manual_label) 
    auto_label = np.array(auto_label) 
    tpr_ratio = np.array(tpr_ratio) 
    rmssd = np.array(rmssd) 
    rmssd = rmssd 
    
    se = np.array(se) 
    
    print("segment_128: ", segment_128.shape)
    print("manual_label: ", manual_label.shape)
    print("auto_label: ", auto_label.shape)
    print("tpr_ratio: ", tpr_ratio.shape)
    print("rmssd: ", rmssd.shape)
    print("se: ", se.shape)
    
    np.savez('comparision_auto_manual_MIMIC.npz',
         segment=segment_128,
         manual_label=manual_label,
         auto_label=auto_label,
         tpr_ratio=tpr_ratio,
         rmssd=rmssd,
         se=se
         )
    
    
    
    
    
    
    