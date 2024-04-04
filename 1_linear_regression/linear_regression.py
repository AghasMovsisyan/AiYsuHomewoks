import numpy as np
import matplotlib.pyplot as plt

def lin_reg_coefs(x, y):
    '''
    x: input data, need to append column of 1s for bias
    y: target data
    '''
    # compute pseudoinverse using np.linalg
    x_pse = np.linalg.pinv(x)
    w = np.dot(x_pse, y)
    # another extended formula w = np.dot(np.linalg.inv(np.dot(x.T, x)), y)
    return w


def lin_reg_l2(x, y, w):
    '''
    x: input data
    y: target data
    w: given parameters
    '''
    y_pred = np.dot(x, w)
    # mean square error
    return np.sqrt(np.mean((y_pred - y) ** 2))


# Now let's test the functions

x = np.random.rand(10, 5)
w_0 = np.arange(5)
y = np.dot(x, w_0)
w = lin_reg_coefs(x, y)
print(w, w.shape)

error = lin_reg_l2(x, y, w)
print(error)
