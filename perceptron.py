import numpy as np
from sklearn.datasets import make_classification
import matplotlib.pyplot as plt

def step(x):
    return 1 if x >= 0 else 0

# Generate synthetic data
X, y = make_classification(n_samples=1000, n_features=2, n_informative=1, 
                            n_redundant=0, n_classes=2, n_clusters_per_class=1, 
                            random_state=41, hypercube=False, class_sep=10)

# Scatter plot of the synthetic data
plt.figure(figsize=(10, 6))
plt.scatter(X[:, 0], X[:, 1], c=y, cmap='winter', s=100, edgecolors='k')
plt.xlabel("Feature 1")
plt.ylabel("Feature 2")
plt.title("Scatter Plot of Synthetic Classification Data")
plt.show()

# Perceptron implementation
def perceptron(X, y):
    X = np.insert(X, 0, 1, axis=1)  # Add bias term (column of ones)
    weights = np.ones(X.shape[1])   # Initialize weights
    lr = 0.1  # Learning rate
    
    for i in range(1000):
        j = np.random.randint(0, len(X))  # Random sample index
        y_hat = step(np.dot(X[j], weights))  # Predicted output
        weights = weights + lr * (y[j] - y_hat) * X[j]  # Update weights
    
    return weights[0], weights[1:]  # Return intercept and coefficients

# Train perceptron
intercept, coef = perceptron(X, y)
print("Intercept:", intercept)
print("Coefficients:", coef)

# Calculate slope and intercept for the decision boundary
m = -(coef[0] / coef[1])  # Slope = -w1 / w0
b = -(intercept / coef[1])  # Intercept = -w0 / w1

# Generate a range of x values for plotting the decision boundary
x_vals = np.linspace(-3, 3, 100)

# Compute corresponding y values for the decision boundary
y_vals = m * x_vals + b

# Plot the decision boundary
plt.figure(figsize=(10, 6))
plt.scatter(X[:, 0], X[:, 1], c=y, cmap='winter', s=100, edgecolors='k')
plt.plot(x_vals, y_vals, label=f"Decision Boundary: y = {m:.2f}x + {b:.2f}", color='r')
plt.xlabel("Feature 1")
plt.ylabel("Feature 2")
plt.title("Perceptron Decision Boundary")
plt.legend()
plt.show()
