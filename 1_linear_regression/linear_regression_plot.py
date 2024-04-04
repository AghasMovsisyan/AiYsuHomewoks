# linear_regression_plot.py

import numpy as np
import matplotlib.pyplot as plt

def plot_linear_regression(X, y, y_pred):
    plt.scatter(X, y, color='blue')
    plt.plot(X, y_pred, color='red')
    plt.xlabel('X')
    plt.ylabel('y')
    plt.show()

if __name__ == "__main__":
    X = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
    y = np.array([1, 3, 2, 5, 7, 8, 8, 9, 10, 12])

    w = np.polyfit(X, y, 1)[0]
    b = np.polyfit(X, y, 1)[1]
    y_pred = w * X + b

    plot_linear_regression(X, y, y_pred)
