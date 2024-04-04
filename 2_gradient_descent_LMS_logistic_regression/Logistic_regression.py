import numpy as np
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def sigmoid_cross_entropy(y, y_pred):
    '''
    y: labels
    y_pred: probabilistic output in [0,1]
    '''
    assert y.shape == y_pred.shape, "label and prediction shapes should be equal"
    L = -(np.mean(y*np.log(y_pred) + (1-y)*np.log(1-y_pred)))
    return L

def logistic_regression(X, y, learning_rate=0.01, num_iterations=1000):
    m, n = X.shape
    theta = np.zeros((n, 1))
    for i in range(num_iterations):
        z = np.dot(X, theta)
        y_pred = sigmoid(z)
        gradient = np.dot(X.T, (y_pred - y)) / m
        theta = theta - learning_rate * gradient
        loss = sigmoid_cross_entropy(y, y_pred)
        print(f"Step {i}, Loss: {loss}")
    return theta

# Generate synthetic dataset
X, y = make_classification(n_samples=100, n_features=10, n_classes=2, class_sep=0.3, random_state=42)
# Add intercept column to X (for bias)
X = np.hstack((np.ones((X.shape[0], 1)), X))
# make y matrix
y = y[:, np.newaxis]
print(X.shape, y.shape)

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train logistic regression model
theta = logistic_regression(X_train, y_train, num_iterations=1000)

# Evaluate the model
y_pred = sigmoid(np.dot(X_test, theta)) > 0.5
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy}")
