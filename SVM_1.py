import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import classification_report, accuracy_score

# 1. Load dataset
data = np.load('comparision_auto_manual_MIMIC.npz')
tpr_ratio = data['tpr_ratio']
rmssd = data['rmssd']
se = data['se']
groundtruth = data['manual_label']

# 2. Create feature matrix X and label vector y
# Assuming all features are 1D arrays of equal length
X = np.vstack((tpr_ratio, rmssd, se)).T  # shape: (num_samples, 3)
y = groundtruth  # shape: (num_samples,)

scaler = StandardScaler()           # Transform val using train scaler
X = scaler.fit_transform(X)   
# 3. Split data: 60% train, 20% validation, 20% test
X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.4, random_state=42, stratify=y)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp)

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
