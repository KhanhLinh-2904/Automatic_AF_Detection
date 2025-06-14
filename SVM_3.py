import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import classification_report, accuracy_score
from sklearn.preprocessing import StandardScaler
# 1. Load dataset
data_train = np.load('comparision_auto_manual_MIMIC.npz')
tpr_ratio_train = data_train['tpr_ratio']
rmssd_train = data_train['rmssd']
se_train = data_train['se']
groundtruth_train = data_train['manual_label']

data_test = np.load('comparision_auto_manual.npz')
tpr_ratio_test = data_test['tpr_ratio']
rmssd_test = data_test['rmssd']
se_test = data_test['se']
groundtruth_test = data_test['manual_label']

# 2. Create feature matrix X and label vector y
# Assuming all features are 1D arrays of equal length
X = np.vstack((tpr_ratio_train, rmssd_train, se_train)).T  # shape: (num_samples, 3)
y = groundtruth_train  # shape: (num_samples,)
scaler = StandardScaler()           # Transform val using train scaler
X = scaler.fit_transform(X)  
X_test = np.vstack((tpr_ratio_test, rmssd_test, se_test)).T  # shape: (num_samples, 3)
y_test = groundtruth_test  # shape: (num_samples,)
X_test = scaler.fit_transform(X_test)  
# 3. Split data: 80% train, 20% validation, cross data test
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# 4. Train SVM model
model = SVC(kernel='rbf', C=1.0, gamma='scale')  # Try 'linear' or tune C, gamma
model.fit(X_train, y_train)

# 5. Evaluate
def evaluate(model, X, y, name):
    y_pred = model.predict(X)
    print(f"\n--- {name} Results ---")
    print("Accuracy:", accuracy_score(y, y_pred))
    print(classification_report(y, y_pred))

evaluate(model, X_train, y_train, "Train")
evaluate(model, X_val, y_val, "Validation")
evaluate(model, X_test, y_test, "Test")
