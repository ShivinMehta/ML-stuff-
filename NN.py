import numpy as np

np.random.seed(1)
X = np.random.randn(3, 10)
Y = (np.random.randn(1, 10) > 0)


def relu(Z):
    s = np.maximum(0, Z)
    return s


def leakyrelu(Z):
    s = np.maximum(0.01*Z, Z)
    return s


def sigmoid(Z):
    s = 1/(1-np.exp(-Z))
    return s


def tanh(Z):
    s = (np.exp(Z) - np.exp(-Z))/(np.exp(Z)+np.exp(-Z))
    return s


def shape(X, Y):
    n_x = X.shape[0]
    n_h1 = 4
    n_h2 = 3
    n_h3 = 2
    n_h4 = 2
    n_y = Y.shape[0]
    m = Y.shape[1]

    return (n_x, n_h1, n_h2, n_h3, n_h4, n_y, m)


# Initialization of W & b

def init_parameters(n_x, n_h1, n_h2, n_h3, n_h4, n_y, m):
    np.random.seed(1)

    W1 = np.random.randn(n_h1, n_x)*0.01
    b1 = np.random.randn(n_h1, 1)*0.01

    W2 = np.random.randn(n_h2, n_h1)*0.01
    b2 = np.random.randn(n_h2, 1)*0.01

    W3 = np.random.randn(n_h3, n_h2)*0.01
    b3 = np.random.randn(n_h3, 1)*0.01

    W4 = np.random.randn(n_h4, n_h3)*0.01
    b4 = np.random.randn(n_h4, 1)*0.01

    W5 = np.random.randn(n_y, n_h4)*0.01
    b5 = np.random.randn(n_h4, 1)*0.01

    parameters = {"W1": W1, "b1": b1, "W2": W2, "b2": b2,
                  "W3": W3, "b3": b3, "W4": W4, "b4": b4, "W5": W5, "b5": b5}

    return parameters


# Forward Propagation

def FWP(X, parameters):
    W1 = parameters["W1"]
    b1 = parameters["b1"]
    W2 = parameters["W2"]
    b2 = parameters["b2"]
    W3 = parameters["W3"]
    b3 = parameters["b3"]
    W4 = parameters["W4"]
    b4 = parameters["b4"]
    W5 = parameters["W5"]
    b5 = parameters["b5"]

    Z1 = np.dot(W1, X) + b1
    A1 = relu(Z1)

    Z2 = np.dot(W2, A1) + b2
    A2 = relu(Z2)

    Z3 = np.dot(W3, A2) + b3
    A3 = relu(Z3)

    Z4 = np.dot(W4, A3) + b4
    A4 = relu(Z4)

    Z5 = np.dot(W5, A4) + b5
    A5 = relu(Z5)

    catch = {"Z1": Z1, "A1": A1, "Z2": Z2, "A2": A2, "Z3": Z3,
             "A3": A3, "Z4": Z4, "A4": A4, "Z5": Z5, "A5": A5}

    return A5, catch


# Cost function


def cost(A5, Y):
    m = Y.shape[1]
    logp = np.multiply(Y, np.log(A5))+np.multiply((1-Y), np.log(1 - A5))
    cost = -(np.sum(logp)/m)
    return cost


# Backward Propagation

def BWP(catch, parameters, X, Y):
    W1 = parameters["W1"]
    b1 = parameters["b1"]
    W2 = parameters["W2"]
    b2 = parameters["b2"]
    W3 = parameters["W3"]
    b3 = parameters["b3"]
    W4 = parameters["W4"]
    b4 = parameters["b4"]
    W5 = parameters["W5"]
    b5 = parameters["b5"]

    A1 = catch["A1"]
    A2 = catch["A2"]
    A3 = catch["A3"]
    A4 = catch["A4"]
    A5 = catch["A5"]

    m = Y.shape[1]

    dZ5 = A5 - Y
    dW5 = np.dot(dZ5, A4.T)/m
    db5 = np.sum(dZ5, axis=1, keepdims=True)/m

    dZ4 = np.dot(dW5.T, dZ5)
    dW4 = np.dot(dZ4, A3.T)/m
    db4 = np.sum(dZ4, axis=1, keepdims=True)/m

    dZ3 = np.dot(dW4.T, dZ4)
    dW3 = np.dot(dZ3, A2.T)/m
    db3 = np.sum(dZ3, axis=1, keepdims=True)/m

    dZ2 = np.dot(dW3.T, dZ3)
    dW2 = np.dot(dZ2, A1.T)/m
    db2 = np.sum(dZ2, axis=1, keepdims=True)/m

    dZ1 = np.dot(dW2.T, dZ2)
    dW1 = np.dot(dZ1, X.T)/m
    db1 = np.sum(dZ1, axis=1, keepdims=True)/m

    gradient = {"dW1": dW1, "db1": db1, "dW2": dW2, "db2": db2, "dW3": dW3,
                "db3": db3, "dW4": dW4, "db4": db4, "dW5": dW5, "db5": db5}

    return gradient


# Update


def update(parameters, gradient, lr=0.01):
    W1 = parameters["W1"]
    b1 = parameters["b1"]
    W2 = parameters["W2"]
    b2 = parameters["b2"]
    W3 = parameters["W3"]
    b3 = parameters["b3"]
    W4 = parameters["W4"]
    b4 = parameters["b4"]
    W5 = parameters["W5"]
    b5 = parameters["b5"]

    dW1 = gradient["dW1"]
    db1 = gradient["db1"]
    dW2 = gradient["dW2"]
    db2 = gradient["db2"]
    dW3 = gradient["dW3"]
    db3 = gradient["db3"]
    dW4 = gradient["dW4"]
    db4 = gradient["db4"]
    dW5 = gradient["dW5"]
    db5 = gradient["db5"]

    W1 = W1 - lr*dW1
    W2 = W2 - lr*dW2
    W3 = W3 - lr*dW3
    W4 = W4 - lr*dW4
    W5 = W5 - lr*dW5

    b1 = b1 - lr*db1
    b2 = b2 - lr*db2
    b3 = b3 - lr*db3
    b4 = b4 - lr*db4
    b5 = b5 - lr*db5

    parameters = {"W1": W1, "b1": b1, "W2": W2, "b2": b2,
                  "W3": W3, "b3": b3, "W4": W4, "b4": b4, "W5": W5, "b5": b5}

    return parameters


epoch = 100
n_x, n_h1, n_h2, n_h3, n_h4, n_y, m = shape(X, Y)
parameters = init_parameters(n_x, n_h1, n_h2, n_h3, n_h4, n_y, m)

for _ in range(epoch):
    A5, catch = FWP(X, parameters)
    cost(A5, Y)
    gradient = BWP(catch, parameters, X, Y)
    update(parameters, gradient, lr=0.01)

print(parameters)
