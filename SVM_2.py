import numpy as np
import matplotlib.pyplot as plt
from sklearn.svm import SVC
from sklearn.metrics import (
    roc_curve, auc, accuracy_score,
    precision_score, recall_score, confusion_matrix
)
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler

# Load the dataset
data = np.load('comparision_auto_manual_MIMIC.npz')
X = np.vstack((data['tpr_ratio'], data['rmssd'], data['se'])).T
y = data['manual_label']

# Standardize features
scaler = StandardScaler()
X = scaler.fit_transform(X)

# 5-Fold Stratified CV
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# For ROC plotting
plt.figure(figsize=(8, 6))
mean_fpr = np.linspace(0, 1, 100)
tprs, aucs = [], []

# Store metrics for each fold
accuracies, precisions, recalls, specificities = [], [], [], []

for i, (train_idx, test_idx) in enumerate(skf.split(X, y)):
    X_train, X_test = X[train_idx], X[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]

    # Train SVM
    model = SVC(kernel='rbf', C=1.0, gamma='scale', probability=True)
    model.fit(X_train, y_train)

    # Predict
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    # Confusion matrix
    tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()

    # Metrics
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred)
    rec = recall_score(y_test, y_pred)
    spec = tn / (tn + fp) if (tn + fp) > 0 else 0

    accuracies.append(acc)
    precisions.append(prec)
    recalls.append(rec)
    specificities.append(spec)

    print(f"\nFold {i + 1}")
    print(f"Accuracy     : {acc:.4f}")
    print(f"Precision    : {prec:.4f}")
    print(f"Recall (Sens): {rec:.4f}")
    print(f"Specificity  : {spec:.4f}")

    # ROC
    fpr, tpr, _ = roc_curve(y_test, y_proba)
    roc_auc = auc(fpr, tpr)
    aucs.append(roc_auc)

    # Interpolate for mean ROC
    tpr_interp = np.interp(mean_fpr, fpr, tpr)
    tpr_interp[0] = 0.0
    tprs.append(tpr_interp)

    plt.plot(fpr, tpr, lw=1.5, label=f'Fold {i+1} (AUC = {roc_auc:.2f})')

# Mean ROC
mean_tpr = np.mean(tprs, axis=0)
mean_tpr[-1] = 1.0
mean_auc = auc(mean_fpr, mean_tpr)
std_auc = np.std(aucs)

plt.plot(mean_fpr, mean_tpr, color='b',
         label=f'Mean ROC (AUC = {mean_auc:.2f} ± {std_auc:.2f})',
         lw=2.5, linestyle='--')
plt.plot([0, 1], [0, 1], 'k--', lw=1)
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('5-Fold Cross-Validated ROC for SVM Classifier')
plt.legend(loc='lower right')
plt.grid(True)
plt.tight_layout()
plt.show()

# Print average metrics
print("\n=== Cross-Validation Summary ===")
print(f"Average Accuracy     : {np.mean(accuracies):.4f} ± {np.std(accuracies):.4f}")
print(f"Average Precision    : {np.mean(precisions):.4f}")
print(f"Average Recall       : {np.mean(recalls):.4f}")
print(f"Average Specificity  : {np.mean(specificities):.4f}")
