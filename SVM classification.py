import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

# --- 1. SETUP DATA ---
X, y = datasets.make_blobs(n_samples=100, centers=2, n_features=2, random_state=42)
y = np.where(y == 0, -1, 1) # SVM needs labels -1 and 1

# --- 2. FROM SCRATCH (NumPy) ---
class ScratchSVM:
    def __init__(self, lr=0.001, lambda_p=0.01, iters=1000):
        self.lr, self.lambda_p, self.iters = lr, lambda_p, iters
        self.w, self.b = None, None

    def fit(self, X, y):
        self.w = np.zeros(X.shape[1])
        self.b = 0
        for _ in range(self.iters):
            for i, x_i in enumerate(X):
                if y[i] * (np.dot(x_i, self.w) - self.b) >= 1:
                    self.w -= self.lr * (2 * self.lambda_p * self.w)
                else:
                    self.w -= self.lr * (2 * self.lambda_p * self.w - np.dot(x_i, y[i]))
                    self.b -= self.lr * y[i]

    def predict(self, X):
        return np.sign(np.dot(X, self.w) - self.b)

# Train Scratch Model
scratch_model = ScratchSVM()
scratch_model.fit(X, y)
scratch_preds = scratch_model.predict(X)

# --- 3. WITH LIBRARY (Scikit-Learn) ---
lib_model = SVC(kernel='linear', C=1.0)
lib_model.fit(X, y)
lib_preds = lib_model.predict(X)

# --- 4. PRINT OUTPUTS ---
print(f"Scratch SVM Accuracy: {accuracy_score(y, scratch_preds) * 100}%")
print(f"Library SVM Accuracy: {accuracy_score(y, lib_preds) * 100}%")

# --- 5. VISUALIZATION ---
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

def plot_svm(ax, model, title, is_library):
    ax.scatter(X[:, 0], X[:, 1], c=y, cmap='bwr', alpha=0.7)
    xlim, ylim = ax.get_xlim(), ax.get_ylim()
    xx, yy = np.meshgrid(np.linspace(xlim[0], xlim[1], 50), np.linspace(ylim[0], ylim[1], 50))
    
    if is_library:
        Z = model.decision_function(np.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape)
    else:
        Z = (np.dot(np.c_[xx.ravel(), yy.ravel()], model.w) - model.b).reshape(xx.shape)
        
    ax.contour(xx, yy, Z, colors='k', levels=[0], linestyles=['-'])
    ax.set_title(title)

plot_svm(ax1, scratch_model, "Scratch (NumPy)", False)
plot_svm(ax2, lib_model, "Library (Scikit-Learn)", True)
plt.show()
