import numpy as np

def shannonEntropy(signal, bin_size=16, window_size=128):
    signal = np.array(signal, dtype=float)
    signal_minus_outliers = signal.copy()

    # Remove 8 maximum values (outliers)
    for _ in range(8):
        max_index = np.argmax(signal_minus_outliers)
        signal_minus_outliers[max_index] = 0

    # Remove 8 minimum values (outliers)
    for _ in range(8):
        min_index = np.argmin(signal_minus_outliers)
        signal_minus_outliers[min_index] = 0

    # Remove all zeros (placeholders for removed outliers)
    signal_minus_outliers = signal_minus_outliers[signal_minus_outliers != 0]

    # Compute histogram (frequencies)
    counts, _ = np.histogram(signal_minus_outliers, bins=bin_size)

    # Avoid division by zero
    total_count = window_size - bin_size
    if total_count <= 0:
        return np.nan

    # Compute probabilities
    probabilities = counts / total_count

    # Compute Shannon entropy
    # Use base 1/16 = log2(prob)/log2(1/16) = -log2(prob)/4
    entropy = 0
    for p in probabilities:
        if p > 0:
            entropy += p * (np.log(p) / np.log(1 / bin_size))  # base 1/bin_size

    return entropy

