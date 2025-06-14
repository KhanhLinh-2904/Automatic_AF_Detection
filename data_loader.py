
import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
def normalize(sample):
    mean = np.mean(sample)
    std = np.std(sample)
    return (sample - mean) / std if std > 0 else sample
class CustomDataset(Dataset):
    def __init__(self, data_dir="train.npz", transform=normalize):
        """
        Args:
            data_dir (str): Path to the dataset directory containing the npz file with 'all_segments' and 'all_labels'.
            transform (callable, optional): Optional transform to be applied on a sample.
        """
        self.data_dir = data_dir
        self.transform = transform
        
        # Load the npz file containing 'all_segments' and 'all_labels'
        data = np.load(data_dir)
        
        # Assuming 'all_segments' and 'all_labels' are the correct keys
        self.ecg_segments = data['X']  # Update this if needed based on the printed keys
        self.labels = data['y']  # Update this if needed based on the printed keys
        
    def __len__(self):
        # Return the number of samples in the dataset (number of ECG segments)
        return len(self.ecg_segments)   
    
    def __getitem__(self, idx):
        # Get the ECG segment at the given index
        ecg_segment = self.ecg_segments[idx]
        
      
        
        # Get the corresponding label for the segment
        label = self.labels[idx]
        
     
        transformed_ecg_segment = self.transform(transformed_ecg_segment)
        
        # Return the transformed ECG segment and the corresponding label
        return (transformed_ecg_segment, label)

if __name__ == "__main__":
    # Create an instance of the dataset
    dataset = CustomDataset(data_dir='test.npz')

    # Create a DataLoader instance to load the dataset in batches
    data_loader = DataLoader(dataset, batch_size=1, shuffle=True, num_workers=4)

    # Iterate through the data
    for batch_idx, (data, labels) in enumerate(data_loader):
        # Print the shapes of data and labels in each batch
        print(f"Batch {batch_idx + 1}: Data shape: {data.shape}, Labels shape: {labels.shape}")
    
  

