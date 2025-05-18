import numpy as np
import matplotlib.pyplot as plt

from visualization import plot_ecg_segment

def test():
    folder_path = "comparision_auto_manual.npz"
    data = np.load(folder_path)
    # rr_interval = data["segment"]
    manual_label = data["manual_label"]
    auto_label = data["auto_label"]
    tpr_ratio = data["tpr_ratio"]
    rmssd = data["rmssd"]
    se = data["se"]
    # print("rr_interval: ", rr_interval.shape)
    print("manual_label: ", manual_label.shape)
    print("auto_label: ", auto_label.shape)
    
    print("tpr_ratio: ", tpr_ratio.shape)
    print("rmssd: ", rmssd.shape)
    print("se: ", se.shape)
    # for i in range(len(manual_label)):
    #     if manual_label[i]:
    #        plot_ecg_segment(rr_interval[i,:], "AF") 
    #        break
    # plot_ecg_segment(rr_interval[0,:],"Non-AF")
    # # Pie chart for manual label
    AF_index = []
    N_index = []

    
    for i in range(len(manual_label)):
        if manual_label[i] == 1:
            AF_index.append(i)
        else:
            N_index.append(i)
    
    print("AF_index: ", len(AF_index))
    print("N_index: ", len(N_index))        
    draw_piechart(AF_index, N_index , manual_label, type="Manual")       
   
    #Pie chart for manual label
    auto_A_index = []
    auto_N_index = []
    #Pie chart for auto label
    for i in range(len(auto_label)):
        if auto_label[i] == 1:
            auto_A_index.append(i)
        else:
            auto_N_index.append(i)
    
    print("auto_A_index: ", len(auto_A_index))
    print("auto_N_index: ", len(auto_N_index)) 
    draw_piechart(auto_A_index, auto_N_index , auto_label, type="Auto")       
    
    
    draw_barchart(AF_index, N_index, "Turining Point Ratio", tpr_ratio)
    draw_barchart(AF_index, N_index, "RMSSD", rmssd)
    draw_barchart(AF_index, N_index, "Shannon Entropy", se)
    
    tp = np.sum((auto_label == True) & (manual_label == True))

    # True Negative (TN): detected = 0, groundtruth = 0
    tn = np.sum((auto_label == False) & (manual_label == False))

    # False Positive (FP): detected = 1, groundtruth = 0
    fp = np.sum((auto_label == True) & (manual_label == False))

    # False Negative (FN): detected = 0, groundtruth = 1
    fn = np.sum((auto_label == False) & (manual_label == True))

    # Accuracy
    accuracy = (tp + tn) / len(auto_label) if len(auto_label) > 0 else 0

    # Precision
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0

    # Recall (Sensitivity)
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0

    # F1 Score
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

    # Output results
    print("TP:", tp)
    print("TN:", tn)
    print("FP:", fp)
    print("FN:", fn)
    print("Accuracy:", accuracy)
    print("Precision:", precision)
    print("Recall:", recall)
    print("F1 Score:", f1)
    
def draw_piechart(AF_index, N_index, label, type="Manual"):
    AF_ratio = len(AF_index)  / len(label)   
    non_AF_ratio = len(N_index) / len(label) 
  
    # Pie chart
    labels = ['AF', 'Non-AF']
    sizes = [AF_ratio, non_AF_ratio]
    colors = ['red', 'green']
    explode = (0.1, 0)  # explode AF slice for emphasis

    plt.figure(figsize=(6, 6))
    plt.pie(sizes, explode=explode, labels=labels, colors=colors, 
            autopct='%1.1f%%', shadow=True, startangle=90)
    plt.title('AF vs Non-AF Ratio '+" "+type)
    plt.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.
    plt.show()
    
def draw_barchart(af, non_af, title_name, array):
    af_arr = [array[i] for i in af]    
    non_af_arr = [array[i] for i in non_af]
    
    fig, axes = plt.subplots(nrows=2, ncols=1, figsize=(8,6), sharex=True)
    
    #Top plot (AF)
    axes[0].hist(af_arr, color="darkorange", label="AF")
    axes[0].legend()
    axes[0].set_ylabel('# of windows')
    
    # Bottom plot (Non-AF)
    axes[1].hist(non_af_arr, color='steelblue', label='Non-AF')
    axes[1].legend()
    axes[1].set_ylabel('# of windows')
    axes[1].set_xlabel(title_name)

    # Adjust layout
    plt.tight_layout()
    plt.show()

def draw_linechart(af, non_af, title_name, array):
    x = list(range(len(array)))
    y = array

    fig, ax = plt.subplots(figsize=(12, 6))
    af_idx_start = len(non_af)

    # Plot non-AF segment (blue)
    ax.plot(x[:af_idx_start], y[:af_idx_start], color='blue', label='non-AF')

    # Plot AF segment (red)
    ax.plot(x[af_idx_start:], y[af_idx_start:], color='red', label='AF')

    # Axis titles and labels
    ax.set_title(title_name)


    # Show all ticks on the x-axis
    ax.set_xticks(x)

    plt.tight_layout()
    plt.show()


    
if __name__ == "__main__":
    test()