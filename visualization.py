import numpy as np
import matplotlib.pyplot as plt

def plot_ecg_segment(segment, label, fs=250):
    """Randomly selects an ECG segment and plots it."""
    time_axis = np.linspace(0, len(segment), len(segment))
    
    plt.figure(figsize=(10, 4))
    plt.plot(time_axis, segment, label=f"Label: {label}")
    plt.xlabel("RR intervals")
    plt.ylabel("Amplitude")
    plt.title("128 RR intervals in one segment of dataset")
    plt.legend()
    plt.grid()
    plt.show()