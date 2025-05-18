## Automatic Real Time Detection of Atrial Fibrillation
This project is for progress meeting with Prof.Lie. The aim is to automatically detect AF based on its characteristic and probability without using deep learning model. It can address the problem of insufficient dataset as ECG and PPG for AF are rare.

This project divides into main stages:
- Preprocess dataset: detect R-peaks and then calculate RR_intervals. Each record will be divided into 128 RR segments with labels AF or non-AF.
- Calculate RMSSD, TPR and Shannon Entropy
- Classifiy AF vs non-AF based on RMSSD, TPR, and Shannon Entropy and then calculate: Accuracy, Sensitivity, Specificity

### Download dataset 
The link to download MIT-BIH Atrial Fibrillation Database
``` bash 
https://physionet.org/content/afdb/1.0.0/
```

### Download Paper 
The link to download Automatic Real Time Detection of Atrial Fibrillation
``` bash 
https://pubmed.ncbi.nlm.nih.gov/19533358/
```
### 📦 Requirements

- Python 3.8+
- PyTorch
- scikit-learn
- matplotlib
- numpy
- wdfb

### Install dependencies:

```bash
pip install -r requirements.txt
```
### Preprocess dataset 
```bash
python loadData.py
```
### Run 
```bash
python test.py
```
