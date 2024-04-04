import numpy as np


def compute_gradients(X, Y, W):
    '''
    X: input data
    Y: output data
    W: parameters (weights)
    return the gradient of parameters W
    '''
    m = len(Y)
    predictions = np.dot(X, W)
    errors = predictions - Y
    gradients = (2 / m) * np.dot(X.T, errors)
    return gradients


def lin_reg_l2(x, y, w):
    '''
    x: input data
    y: target data
    w: given parameters
    '''
    y_pred = np.dot(x, w)
    # mean square error
    return np.sqrt(np.mean((y_pred - y) ** 2))


def gradient_descent(X, y, learn_rate, num_iters, theta=None):
    '''
    X: input data
    y: output data
    learn_rate: the learning rate for gradient descent 
    num_iterations
    theta: initial value for the given parameters 
    '''
    m = len(y)
    if not theta:
        theta = np.zeros((X.shape[1], 1))
    for i in range(num_iters):
        gradients = compute_gradients(X, y, theta)
        theta = theta - learn_rate * gradients
        loss = lin_reg_l2(X, y, theta)
        print(f'{i}) loss: {loss}')
    return theta


# Generate random data
np.random.seed(0)
X = np.random.rand(100, 2)
y = 2 + np.dot(X, np.array([[3, 4]]).T) + 0.1 * np.random.rand(100, 1)

# Add intercept column to X (for bias)
X = np.hstack((np.ones((X.shape[0], 1)), X))

# Calculate parameters using normal equation
theta_normal = np.dot(np.dot(np.linalg.inv(np.dot(X.T, X)), X.T), y)

# Calculate parameters using gradient descent
theta_gd = gradient_descent(X, y, 0.1, 400)

print("Parameters using normal equation: ", theta_normal.T)
print("Parameters using gradient descent: ", theta_gd.T)
