import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import accuracy_score, precision_score, recall_score, confusion_matrix
from sklearn.svm import SVC

# Data generation
np.random.seed(42)

circle_center = np.random.uniform(2, 5, size=(2,))
circle_radius = np.random.uniform(2, 5, size=(1,))

num_points = 10000
points = np.random.uniform(-10, 10, size=(num_points, 2))
labels = np.linalg.norm(points - circle_center, axis=1) < circle_radius
outlier_fraction = 0.05
num_outliers = int(outlier_fraction * num_points)
outlier_indices = np.random.choice(num_points, num_outliers, replace=False)
labels[outlier_indices] = ~labels[outlier_indices]

# Print the shape of the generated points and labels
print("Shape of points:", points.shape)
print("Shape of labels:", labels.shape)

# Create a DataFrame using class_dict
class_dict = {'x': points[:, 0], 'y': points[:, 1], 'class': labels}
data = pd.DataFrame(class_dict)

# Create polynomial features using PolynomialFeatures
poly = PolynomialFeatures(degree=2, include_bias=True)
features = poly.fit_transform(data[['x', 'y']])

# Create a new DataFrame with polynomial features
data_poly_features = pd.DataFrame(features, columns=['1', 'x', 'y', 'x^2', 'xy', 'y^2'])

# Split the data into training and testing sets using train_test_split
x_train, x_test, y_train, y_test = train_test_split(data_poly_features, labels, train_size=0.8)

# Create a Logistic Regression model
model = LogisticRegression()

# Train the model on the training set
model.fit(x_train, y_train)

# Predict labels on the testing set
y_predict = model.predict(x_test)

# Print the shapes of predicted and actual labels
print("Shape of predicted labels:", y_predict.shape)
print("Shape of actual labels (y_test):", y_test.shape)

# Define plotting functions
def plot_classes(ax, x, y, labels, circle_center, circle_radius, colors):
    for i, color in enumerate(colors):
        ax.scatter(x[y == i]['x'], x[y == i]['y'], c=color, label=f'{labels[i]}')
    circle = plt.Circle(circle_center, radius=circle_radius, edgecolor='black', facecolor='none', linewidth=2)
    ax.add_artist(circle)

def plot_class_distribution(x_test, y_test, y_predict, circle_center, circle_radius):
    plt.figure(figsize=(10, 5))
    ax1 = plt.subplot(1, 2, 1)
    plot_classes(ax1, x_test, y_test, ['Class 0', 'Class 1'], circle_center, circle_radius, ['b', 'r'])

    ax2 = plt.subplot(1, 2, 2)
    plot_classes(ax2, x_test, y_predict, ['Class 0', 'Class 1'], circle_center, circle_radius, ['b', 'y'])

# Plot class distributions
plot_class_distribution(x_test, y_test, y_predict, circle_center, circle_radius)
plt.show()

# Calculate model coefficients
print("Model Coefficients:", model.coef_[0])

# Calculate model evaluation metrics
accuracy = accuracy_score(y_test, y_predict)
precision1 = precision_score(y_test, y_predict, pos_label=1)
precision0 = precision_score(y_test, y_predict, pos_label=0)
recall1 = recall_score(y_test, y_predict, pos_label=1)
recall0 = recall_score(y_test, y_predict, pos_label=0)
conf_matrix = confusion_matrix(y_test, y_predict)

# Print model evaluation metrics
print(f'Accuracy: {accuracy:.3f}')
print(f'Precision1: {precision1:.3f}')
print(f'Precision0: {precision0:.3f}')
print(f'Recall1: {recall1:.3f}')
print(f'Recall0: {recall0:.3f}')
print(f'Confusion Matrix:\n{conf_matrix}')

# Creating an SVM with a quadratic kernel
svm_model = SVC(kernel='poly', degree=2)

# Training the SVM model
svm_model.fit(x_train[['x','y']], y_train)

# Predicting on training data
y_pred = svm_model.predict(x_train[['x','y']])

# Plot class distributions for SVM model
plot_class_distribution(x_train, y_train, y_pred, circle_center, circle_radius)
plt.show()

# Calculate SVM model evaluation metrics
accuracy = accuracy_score(y_train, y_pred)
precision1 = precision_score(y_train, y_pred, pos_label=1)
precision0 = precision_score(y_train, y_pred, pos_label=0)
recall1 = recall_score(y_train, y_pred, pos_label=1)
recall0 = recall_score(y_train, y_pred)
