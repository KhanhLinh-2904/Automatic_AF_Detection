import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Load dataset
data = np.load('comparision_auto_manual_MIMIC.npz')
tpr_ratio = data['tpr_ratio']
rmssd = data['rmssd']
se = data['se']
groundtruth = data['manual_label']

# Create feature matrix X and label vector y
X_test = np.vstack((tpr_ratio, rmssd, se)).T  # shape: (num_samples, 3)
y = groundtruth  # shape: (num_samples,)

# # Split into train (70%) and temp (30%)
# X_train, X_temp, y_train, y_temp = train_test_split(
#     X, y, test_size=0.30, random_state=42, stratify=y
# )

# # Split temp into validation (15%) and test (15%)
# X_val, X_test, y_val, y_test = train_test_split(
#     X_temp, y_temp, test_size=0.50, random_state=42, stratify=y_temp
# )

# # Normalize using StandardScaler
# scaler = StandardScaler()
# X_train = scaler.fit_transform(X_train)     # Fit on train
# X_val = scaler.transform(X_val)             # Transform val using train scaler
# X_test = scaler.transform(X_test)           # Transform test using train scaler

# # Save to .npz files
# np.savez('train.npz', X=X_train, y=y_train)
# np.savez('validation.npz', X=X_val, y=y_val)
# np.savez('test.npz', X=X_test, y=y_test)

# print(f"Saved train ({len(y_train)}), validation ({len(y_val)}), and test ({len(y_test)}) sets.")
scaler = StandardScaler()           # Transform val using train scaler
X_test = scaler.fit_transform(X_test)   
np.savez('test.npz', X=X_test, y=y)
