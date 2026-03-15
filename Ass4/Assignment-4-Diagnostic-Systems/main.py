# import libraries
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from utils import LoadData
from model import FullyConnectedNN
import matplotlib.pyplot as plt
import torch.nn.functional as F
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns

from sklearn.tree import DecisionTreeClassifier


# define settings
parser = argparse.ArgumentParser()
parser.add_argument('--num_classes', type=int, default=50, #default 50
                    help='number of classes used')
parser.add_argument('--num_samples_train', type=int, default=15, # default 15
                    help='number of samples per class used for training')
parser.add_argument('--num_samples_test', type=int, default=5, # default 5 
                    help='number of samples per class used for testing')
parser.add_argument('--seed', type=int, default=2, 
                    help='random seed')
args = parser.parse_args()

# define you model, loss functions, hyperparameters, and optimizers
### Your Code Here ###
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Training on {device}")


# load data
X_train, X_test, y_train, y_test = LoadData()

# make predictions with random forest model 
clf = DecisionTreeClassifier(max_depth = 4, random_state=21)
clf.fit(X_train, y_train)
rf_pred = clf.predict(X_test)
rf_probs = clf.predict_proba(X_test)

# # # Convert NumPy arrays to PyTorch tensors
X_train = torch.tensor(X_train.values, dtype=torch.float32)  # shape: (num_samples, num_features)
X_test  = torch.tensor(X_test.values, dtype=torch.float32)
y_train = torch.tensor(y_train.values, dtype=torch.long)       # use long for class labels
y_test  = torch.tensor(y_test.values, dtype=torch.long)

X_train = X_train.to(device)
X_test = X_test.to(device)
y_train = y_train.to(device)
y_test = y_test.to(device)

epochs = 5000
model = FullyConnectedNN(input_dim=X_train.shape[1], num_classes=2)
model.to(device)

optimizer = optim.Adam(model.parameters(), lr = 1e-5, weight_decay=1e-6) 
criterion = nn.CrossEntropyLoss()

losses = []

# train model using train_image and train_label
for epoch in range(epochs):

    model.train()
    optimizer.zero_grad()
    pred = model(X_train)
    ### Your Code Here ###
    loss = criterion(pred, y_train)
    loss.backward()
    optimizer.step()

    if (epoch + 1) % 100 == 0 or epoch == 0:
        losses.append(loss.item())
        print(f'Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.4f}')

  
# get predictions on test_image
model.eval()
with torch.no_grad():
    ### Your Code Here ###
    test_pred = model(X_test)
    test_pred = torch.argmax(test_pred, dim=1)
    nn_probs = F.softmax(model(X_test), dim=1).cpu().numpy()


# evaluation
print("Test Accuracy NN:", np.mean(1.0 * (test_pred.cpu().numpy() == y_test.cpu().numpy())))
print("Test Accuracy DT:", np.mean(1.0 * (rf_pred == y_test.cpu().numpy())))

# mixed model test accuruacy
alpha = 0.4
ensemble_probs = alpha * nn_probs + (1 - alpha) * rf_probs
ensemble_pred = ensemble_probs.argmax(axis=1)

# Evaluate ensemble accuracy:
accuracy = np.mean(ensemble_pred == y_test.cpu().numpy())
print("Test Accuracy Ensemble:", accuracy)


# loss curve
plt.figure(figsize=(8, 6))
plt.plot(range(epochs//100+1), losses, label='Training Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Loss Curve')
plt.legend()
plt.show()

# Convert tensors to numpy arrays for evaluation (if needed)
y_true = y_test.cpu().numpy()
nn_pred = test_pred.cpu().numpy()  # NN predictions

# Compute confusion matrices
cm_nn = confusion_matrix(y_true, nn_pred)
cm_rf = confusion_matrix(y_true, rf_pred)

# Print classification reports for F1, recall, and precision
print("Neural Network Classification Report:")
print(classification_report(y_true, nn_pred))

print("Decision Tree Classification Report:")
print(classification_report(y_true, rf_pred))

# Plot confusion matrices side-by-side
fig, axs = plt.subplots(1, 2, figsize=(12, 5))

sns.heatmap(cm_nn, annot=True, fmt="d", cmap="Blues", ax=axs[0])
axs[0].set_title("Confusion Matrix - Neural Network")
axs[0].set_xlabel("Predicted")
axs[0].set_ylabel("True")

sns.heatmap(cm_rf, annot=True, fmt="d", cmap="Blues", ax=axs[1])
axs[1].set_title("Confusion Matrix - Decision Tree")
axs[1].set_xlabel("Predicted")
axs[1].set_ylabel("True")

plt.tight_layout()
plt.show()




