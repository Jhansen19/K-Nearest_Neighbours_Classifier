# utils.py: Utility file for implementing helpful utility functions used by the ML algorithms.
#
# Submitted by: [Jonathan Hansen & Mark Green] -- [margree-jonjhans-a5]
#
# Based on skeleton code by CSCI-B 551 Fall 2022 Course Staff
#
# Resources for Jon:
# https://chat.openai.com/

import numpy as np


def euclidean_distance(x1, x2):
    """
    Computes and returns the Euclidean distance between two vectors.
    Args:
        x1: A numpy array of shape (n_features,).
        x2: A numpy array of shape (n_features,).
    Returns:
        A float representing the Euclidean distance between x1 and x2.
    """
    return np.sqrt(np.sum((x1 - x2) ** 2))


def manhattan_distance(x1, x2):
    """
    Computes and returns the Manhattan distance between two vectors.
    Args:
        x1: A numpy array of shape (n_features,).
        x2: A numpy array of shape (n_features,).
    Returns:
       A float representing the Manhattan distance between x1 and x2.
    """
    return np.sum(np.abs(x1 - x2))


def identity(x, derivative = False):
    """
    Computes and returns the identity activation function of the given input data x. If derivative = True,
    the derivative of the activation function is returned instead.
    Args:
        x: A numpy array of shape (n_samples, n_hidden).
        derivative: A boolean representing whether or not the derivative of the function should be returned instead.
    Returns:
        The output of the identity function or its derivative.
    """
    if derivative:
        # The derivative of the identity function is 1
        return np.ones_like(x)
    else:
        # The identity function is f(x) = x
        return x


def sigmoid(x, derivative = False):
    """
    Computes and returns the sigmoid (logistic) activation function of the given input data x. If derivative = True,
    the derivative of the activation function is returned instead.
    Args:
        x: A numpy array of shape (n_samples, n_hidden).
        derivative: A boolean representing whether or not the derivative of the function should be returned instead.
    
    Returns:
        The output of the sigmoid function or its derivative.
    """
    x = np.clip(x, -500, 500)  # Clipping to avoid overflow
    sigmoid_val = 1 / (1 + np.exp(-x))
    if derivative:
        # The derivative of the sigmoid function is sigmoid(x) * (1 - sigmoid(x))
        return sigmoid_val * (1 - sigmoid_val)
    else:
        # The sigmoid function is 1 / (1 + exp(-x))
        return sigmoid_val


def tanh(x, derivative = False):
    """
    Computes and returns the hyperbolic tangent activation function of the given input data x. If derivative = True,
    the derivative of the activation function is returned instead.
    Args:
        x: A numpy array of shape (n_samples, n_hidden).
        derivative: A boolean representing whether or not the derivative of the function should be returned instead.
    Returns:
        The output of the tanh function or its derivative.
    """
    if derivative:
        # The derivative of the tanh function is 1 - tanh^2(x)
        return 1 - np.tanh(x)**2
    else:
        # The tanh function is (exp(x) - exp(-x)) / (exp(x) + exp(-x))
        return np.tanh(x)


def relu(x, derivative = False):
    """
    Computes and returns the rectified linear unit activation function of the given input data x. If derivative = True,
    the derivative of the activation function is returned instead.
    Args:
        x: A numpy array of shape (n_samples, n_hidden).
        derivative: A boolean representing whether or not the derivative of the function should be returned instead.
    Returns:
        The output of the ReLU function or its derivative.
    """
    if derivative:
        # The derivative of ReLU is 1 where x > 0, and 0 otherwise
        return np.where(x > 0, 1, 0)
    else:
        # ReLU function is max(0, x)
        return np.maximum(0, x)


def softmax(x, derivative = False):
    if derivative:
        return softmax(x, derivative=False) * (1 - softmax(x, derivative=False))
    else:
        exp_x = np.exp(x - np.max(x, axis=1, keepdims=True))
        return exp_x / np.sum(exp_x, axis=1, keepdims=True)


def cross_entropy(y, p):
    """
    Computes and returns the cross-entropy loss, defined as the negative log-likelihood of a logistic model that returns
    p probabilities for its true class labels y.
    Args:
        y:
            A numpy array of shape (n_samples, n_outputs) representing the one-hot encoded target class values for the
            input data used when fitting the model.
        p:
            A numpy array of shape (n_samples, n_outputs) representing the predicted probabilities from the softmax
            output activation function.
    Returns:
        A float representing the cross-entropy loss.
    """
    # To avoid division by zero and log(0) errors, small p values clipped to epsilon
    epsilon = 1e-15
    p = np.clip(p, epsilon, 1)
    # Compute the cross-entropy loss
    return -np.sum(y * np.log(p)) / y.shape[0]


def one_hot_encoding(y):
    """
    Converts a vector y of categorical target class values into a one-hot numeric array using one-hot encoding: one-hot
    encoding creates new binary-valued columns, each of which indicate the presence of each possible value from the
    original data.
    Args:
        y: A numpy array of shape (n_samples,) representing the target class values for each sample in the input data.
    Returns:
        A numpy array of shape (n_samples, n_outputs) representing the one-hot encoded target class values for the input
        data. n_outputs is equal to the number of unique categorical class values in the numpy array y.
    """
    n_samples = len(y)
    n_outputs = np.unique(y).size
    one_hot = np.zeros((n_samples, n_outputs))
    one_hot[np.arange(n_samples), y] = 1
    return one_hot

