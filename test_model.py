import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from model import NeuralNetwork
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt
from sklearn.metrics import precision_recall_curve, average_precision_score

# ----------- Configuration -----------
CHECKPOINT_PATH = "checkpoints/best_model.pth"
TEST_DATA_PATH = "test.npz"
BATCH_SIZE = 32

# ----------- Load Test Data -----------
test_data = np.load(TEST_DATA_PATH)
X_test = torch.tensor(test_data['X'], dtype=torch.float32)
y_test = torch.tensor(test_data['y'], dtype=torch.long)

test_dataset = TensorDataset(X_test, y_test)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE)

# ----------- Load Model -----------
num_classes = len(torch.unique(y_test))
assert num_classes == 2, "This code assumes binary classification (2 classes)."
model = NeuralNetwork(num_classes=num_classes)
model.load_state_dict(torch.load(CHECKPOINT_PATH))
model.eval()

# ----------- Evaluation -----------
criterion = nn.CrossEntropyLoss()
total, correct = 0, 0
test_loss = 0.0

# ----------- Evaluation with AUROC -----------
all_labels = []
all_probs = []
# Initialize counters
TP = TN = FP = FN = 0
with torch.no_grad():
    for inputs, labels in test_loader:
        outputs = model(inputs)
        probs = torch.softmax(outputs, dim=1)[:, 1]  # Probability for class 1 (positive)

        all_labels.extend(labels.numpy())
        all_probs.extend(probs.numpy())

        loss = criterion(outputs, labels)
        test_loss += loss.item() * inputs.size(0)

        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

        for p, l in zip(predicted, labels):
            if l == 1 and p == 1:
                TP += 1
            elif l == 0 and p == 0:
                TN += 1
            elif l == 0 and p == 1:
                FP += 1
            elif l == 1 and p == 0:
                FN += 1


# Metrics calculation
avg_loss = test_loss / total
accuracy = 100 * correct / total
precision = TP / (TP + FP) if (TP + FP) > 0 else 0.0
recall = TP / (TP + FN) if (TP + FN) > 0 else 0.0  # Sensitivity
specificity = TN / (TN + FP) if (TN + FP) > 0 else 0.0

# ----------- Results -----------
print(f"Test Loss: {avg_loss:.4f}")
print(f"Test Accuracy: {accuracy:.2f}%")
print(f"True Positives (TP): {TP}")
print(f"True Negatives (TN): {TN}")
print(f"False Positives (FP): {FP}")
print(f"False Negatives (FN): {FN}")
print(f"Precision: {precision:.4f}")
print(f"Recall (Sensitivity): {recall:.4f}")
print(f"Specificity: {specificity:.4f}")

# Compute ROC curve and AUROC
fpr, tpr, thresholds = roc_curve(all_labels, all_probs)
roc_auc = auc(fpr, tpr)

# Plotting
plt.figure()
plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'AUROC = {roc_auc:.4f}')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate (Recall)')
plt.title('Receiver Operating Characteristic (ROC) Curve')
plt.legend(loc="lower right")
plt.grid(True)
plt.show()

# ----------- Compute PRC and Average Precision -----------
precision_curve, recall_curve, prc_thresholds = precision_recall_curve(all_labels, all_probs)
avg_precision = average_precision_score(all_labels, all_probs)

# ----------- Plot Precision-Recall Curve -----------
plt.figure()
plt.plot(recall_curve, precision_curve, color='blue', lw=2,
         label=f'Average Precision (AP) = {avg_precision:.4f}')
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Precision-Recall (PRC) Curve')
plt.grid(True)
plt.legend(loc='lower left')
plt.ylim([0.0, 1.05])
plt.xlim([0.0, 1.0])
plt.show()
