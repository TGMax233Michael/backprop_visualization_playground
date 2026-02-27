import numpy as np

def sgd(grad, weights, lr):
    return weights - lr * grad

def momentum(grad, weights, lr, beta_m, m):
    m_new = beta_m * m + grad
    return weights - lr * m_new, m_new

def adagrad(grad, weights, lr, V, epsilon):
    V += grad**2
    return weights - lr / (np.sqrt(V) + epsilon) * grad, V

def rmsprop(grad, weights, lr, V, epsilon, beta_v):
    V = beta_v * V + (1 - beta_v) * grad**2
    return weights - lr / (np.sqrt(V) + epsilon) * grad, V

def adam(grad, weights, lr, V, epsilon, beta_m, beta_v, m):
    V = beta_v * V + (1 - beta_v) * grad**2
    m = beta_m * m + (1 - beta_m) * grad
    return weights - lr / (np.sqrt(V) + epsilon) * m, V, m