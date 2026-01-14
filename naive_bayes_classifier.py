import numpy as np
import matplotlib.pyplot as plt
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Sample dataset
X = np.array([[1], [2], [3], [4], [5], [6], [7], [8]])
y = np.array([0, 0, 0, 0, 1, 1, 1, 1])

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=0
)

# Model
model = GaussianNB()
model.fit(X_train, y_train)

# Prediction
y_pred = model.predict(X_test)

# Accuracy
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy (Using Library):", accuracy)

# Plot
plt.scatter(X, y, label="Actual Data")
plt.scatter(X_test, y_pred, color='red', label="Predicted")
plt.xlabel("Feature")
plt.ylabel("Class")
plt.legend()
plt.title("Naive Bayes using Library")
plt.show()


import math
import matplotlib.pyplot as plt

# Dataset
X = [1, 2, 3, 4, 5, 6, 7, 8]
y = [0, 0, 0, 0, 1, 1, 1, 1]

# Separate classes
class0 = [X[i] for i in range(len(X)) if y[i] == 0]
class1 = [X[i] for i in range(len(X)) if y[i] == 1]

# Mean
def mean(data):
    return sum(data) / len(data)

# Variance
def variance(data, m):
    return sum((x - m) ** 2 for x in data) / len(data)

# Gaussian Probability
def gaussian_prob(x, m, v):
    return (1 / math.sqrt(2 * math.pi * v)) * math.exp(-(x - m) ** 2 / (2 * v))

# Training
mean0, mean1 = mean(class0), mean(class1)
var0, var1 = variance(class0, mean0), variance(class1, mean1)

# Prediction
def predict(x):
    p0 = gaussian_prob(x, mean0, var0)
    p1 = gaussian_prob(x, mean1, var1)
    return 0 if p0 > p1 else 1

# Predictions
y_pred = [predict(x) for x in X]

# Accuracy
accuracy = sum(1 for i in range(len(y)) if y[i] == y_pred[i]) / len(y)
print("Accuracy (Without Library):", accuracy)

# Plot
plt.scatter(X, y, label="Actual Data")
plt.scatter(X, y_pred, color='red', label="Predicted")
plt.xlabel("Feature")
plt.ylabel("Class")
plt.legend()
plt.title("Naive Bayes without Library")
plt.show()
