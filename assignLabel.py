import numpy as np
from rootMeanSquareSuccessiveDifferences import rootMeanSquareSuccessiveDifferences
from shannonEntropy import shannonEntropy
from turningPointRatio import turningPointRatio
import numpy as np

def assignLabel(segments_128):
    num_windows = segments_128.shape[0]

    # Initialize arrays for feature values and prediction
    detected = np.zeros(num_windows, dtype=bool)
    tpr_ratio = np.zeros(num_windows)
    rmssd = np.zeros(num_windows)
    se = np.zeros(num_windows)

    # Thresholds
    THRESHOLDS = {
        "TPR": (0.54, 0.77),
        "SE": (0.7, 1.5),
        "RMSSD": (0.1, 1.5),
    }

    for i in range(num_windows):
        segment = segments_128[i, :]

        _, tpr_actual, _, _ = turningPointRatio(segment)
        entropy = shannonEntropy(segment)
        rmssd_val = rootMeanSquareSuccessiveDifferences(segment) / np.mean(segment)
        tpr_val = tpr_actual / (128 - 2)

        # Save feature values
        tpr_ratio[i] = tpr_val
        se[i] = entropy
        rmssd[i] = rmssd_val

        # Check thresholds
        is_tpr_valid = THRESHOLDS["TPR"][0] < tpr_val < THRESHOLDS["TPR"][1]
        is_se_valid = THRESHOLDS["SE"][0] < entropy < THRESHOLDS["SE"][1]
        is_rmssd_valid = THRESHOLDS["RMSSD"][0] < rmssd_val < THRESHOLDS["RMSSD"][1]
        print("tpr_val vs entropy vs rmssd_val:", tpr_val, entropy, rmssd_val)
        print("is_tpr_valid vs is_se_valid vs is_rmssd_valid:", is_tpr_valid, is_se_valid, is_rmssd_valid)
        
        # Segment is detected as AFIB if all conditions are met
        if is_tpr_valid and is_se_valid and is_rmssd_valid:
            detected[i] = True

    return detected, tpr_ratio, rmssd, se
