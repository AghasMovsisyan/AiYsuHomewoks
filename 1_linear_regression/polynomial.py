import numpy as np
from sklearn.linear_model import LinearRegression

def lin_reg_coefs_sklearn(x, y):
    '''
    x: input data
    y: target data
    '''
    reg = LinearRegression().fit(x, y)
    return reg.coef_

def lin_reg_l2_sklearn(x, y, w):
    '''
    x: input data
    y: target data
    w: given parameters
    '''
    reg = LinearRegression().fit(x, y)
    y_pred = reg.predict(x)
    # mean square error
    return np.sqrt(np.mean((y_pred - y) ** 2))

def best_polynomial_coefficients(x, y, n):
    '''
    x: input data
    y: target data
    n: degree of the polynomial
    '''
    coefficients = np.polyfit(x, y, n)
    return coefficients


if __name__ == "__main__":
    x = np.random.rand(10, 15)
    w_0 = np.arange(15)
    y = np.dot(x, w_0)

    w_sklearn = lin_reg_coefs_sklearn(x, y)
    print(w_sklearn, w_sklearn.shape)

    error_sklearn = lin_reg_l2_sklearn(x, y, w_sklearn)
    print(error_sklearn)

    X = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
    y = np.array([1, 3, 2, 5, 7, 8, 8, 9, 10, 12])
    n = 3  # degree of the polynomial

    coefficients = best_polynomial_coefficients(X, y, n)
    print("Coefficients of the best polynomial:", coefficients)
