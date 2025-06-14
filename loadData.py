import wfdb
import numpy as np
import os

from assignLabel import assignLabel
from removeEtopic import remove_ectopic_beats



def extractData(ann_atr, ann_qrs, fs):
    max_indices = ann_qrs.sample
    if len(max_indices) < 2:
        return 0, 0, True

    RR_intervals = np.diff(max_indices) / fs
    print("RR_intervals length:", len(RR_intervals))

    segment_128 = []
    reshaped_truth = []

    ground_truth_index = ann_atr.sample
    labels = ann_atr.aux_note

    if len(ground_truth_index) == 0:
        return 0, 0, True

    for i in range(len(ground_truth_index)):
        ground_idx1 = np.searchsorted(max_indices, ground_truth_index[i], side='right')
        if i + 1 == len(ground_truth_index):
            ground_idx2 = len(RR_intervals)
        else:
            ground_idx2 = np.searchsorted(max_indices, ground_truth_index[i + 1], side='right')

        label = labels[i].replace('(', '').replace(')', '')

        if label == 'AFIB':
            label_value = True
        elif label == 'N':
            label_value = False
        else:
            continue  

        segment = RR_intervals[ground_idx1:ground_idx2]

        if len(segment) < 128:
            continue  

        print(f"=== Extracting segment: Label = {label} | Length = {len(segment)} ===")
        print("ground_idx1 vs ground_idx2:", ground_idx1, ground_idx2)

        # Trim and reshape
        modul = len(segment) % 128
        segment = segment[:len(segment) - modul]
        segment = segment.reshape((-1, 128))

        segment_128.append(segment)
        reshaped_truth.extend([label_value] * segment.shape[0])

    if len(segment_128) == 0:
        return 0, 0, True

    # Concatenate results
    segment_128 = np.concatenate(segment_128, axis=0)
    reshaped_truth = np.array(reshaped_truth)

    # Sanity check
    if segment_128.shape[0] != reshaped_truth.shape[0]:
        print("Mismatch between segments and labels!")
        return 0, 0, True

    print("Final segment_128 shape:", segment_128.shape)
    print("Final reshaped_truth shape:", reshaped_truth.shape)

    return segment_128, reshaped_truth, False



        
def loadData(data_path="/home/linhhima/AF-Detection/mit-bih-atrial-fibrillation-database-1.0.0/"):
    record_name = "04126"
    manual_label = []
    auto_label = []
    TPR = []
    RMSSD = []
    SE = []
    record_files = [f.split('.')[0] for f in os.listdir(data_path) if f.endswith('.dat')]
    for record_name in record_files:
        hea_path = os.path.join(data_path, record_name + ".hea")
        atr_path = os.path.join(data_path, record_name + ".atr")
        qrs_path = os.path.join(data_path, record_name + ".qrs")
    
        # Skip if the .hea or .atr file does not exist
        if not os.path.exists(hea_path) or not os.path.exists(atr_path) or not os.path.exists(qrs_path):
            print(f"Skipping {record_name}: Missing .hea or .atr or .qrs file.")
            continue  # Move to the next record
        print("====================================================")
        print(f"Processing record: {record_name}")

        # Read the ECG signal data (both channels)
        record1 = wfdb.rdrecord(os.path.join(data_path, record_name), channels=[0]) 
        # atr contains label AFIB, N
        ann_atr = wfdb.rdann(os.path.join(data_path,record_name), 'atr')  
        # contains index of QRS annotations
        ann_qrs = wfdb.rdann(os.path.join(data_path, record_name), 'qrs')
        Fs = record1.fs # frequency
        segment_128, reshaped_truth, flag = extractData(ann_atr, ann_qrs, Fs)
        if flag:
            continue
        # segment.append(segment_128)
        manual_label.extend(reshaped_truth)
        detected, tpr_ratio, rmssd, se = assignLabel(segment_128)
        auto_label.extend(detected)
        TPR.extend(tpr_ratio)
        RMSSD.extend(rmssd)
        SE.extend(se)
      
       
    np.savez('comparision_auto_manual_2.npz',
        #  segment=np.array(segment),
         manual_label=np.array(manual_label),
         auto_label=np.array(auto_label),
         tpr_ratio=np.array(TPR),
         rmssd=np.array(RMSSD),
         se=np.array(SE)
         )

    


       


# Call the function
if __name__ == "__main__":
    loadData()
    
