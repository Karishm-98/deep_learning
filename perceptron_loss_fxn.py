import numpy as np
from sklearn.datasets import make_classification
import matplotlib.pyplot as plt

X, y = make_classification(n_samples=100, n_features=2, n_informative=1,n_redundant=0,
                           n_classes=2, n_clusters_per_class=1, random_state=41,hypercube=False,class_sep=15)

# Perceptron function to train the model
def perceptron(X, y):
    w1 = w2 = b = 1  # Initial weights and bias
    lr = 0.1  # Learning rate
    for j in range(1000):  # Number of iterations
        for i in range(X.shape[0]):  # Loop over all data points
            # Compute the linear combination
            f = b + w1 * X[i][0] + w2 * X[i][1]
            
            # Update weights if the prediction is incorrect
            if y[i] * f < 0:  # Check if prediction is wrong
                w1 = w1 + lr * y[i] * X[i][0]
                w2 = w2 + lr * y[i] * X[i][1]  # Corrected to use y[i] here
                b = b + lr * y[i]  # Corrected to use y[i] here
    return w1, w2, b

# Train the perceptron
w1, w2, b = perceptron(X, y)

# Calculate the slope and intercept of the decision boundary
m = -(w1 / w2)
c = -(b / w2)

# Generate x values for the decision boundary line
X_input = np.linspace(-3, 3, 1000)
y_input = m * X_input + c  # Use 'c' for intercept instead of 'b'

# Plot the decision boundary and the data points
plt.figure(figsize=(10, 6))
plt.plot(X_input, y_input, color='red', linewidth=3, label="Decision Boundary")
plt.scatter(X[:, 0], X[:, 1], c=y, cmap='winter', s=100, label="Data Points")
plt.ylim(-3, 2)
plt.xlabel("Feature 1")
plt.ylabel("Feature 2")
plt.title("Perceptron - Decision Boundary")
plt.legend()
plt.show()
