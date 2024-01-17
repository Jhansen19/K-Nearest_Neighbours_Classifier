# multilayer_perceptron.py: Machine learning implementation of a Multilayer Perceptron classifier from scratch.
#
# Submitted by: [Jonathan Hansen & Mark Green] -- [margree-jonjhans-a5]
#
# Based on skeleton code by CSCI-B 551 Fall 2023 Course Staff
#
# Resources for Jon:
# https://chat.openai.com/

import numpy as np
from utils import identity, sigmoid, tanh, relu, softmax, cross_entropy, one_hot_encoding


class MultilayerPerceptron:
    """
    A class representing the machine learning implementation of a Multilayer Perceptron classifier from scratch.
    Attributes:
        n_hidden
            An integer representing the number of neurons in the one hidden layer of the neural network.
        hidden_activation
            A string representing the activation function of the hidden layer. The possible options are
            {'identity', 'sigmoid', 'tanh', 'relu'}.
        n_iterations
            An integer representing the number of gradient descent iterations performed by the fit(X, y) method.
        learning_rate
            A float representing the learning rate used when updating neural network weights during gradient descent.
        _output_activation
            An attribute representing the activation function of the output layer. This is set to the softmax function
            defined in utils.py.
        _loss_function
            An attribute representing the loss function used to compute the loss for each iteration. This is set to the
            cross_entropy function defined in utils.py.
        _loss_history
            A Python list of floats representing the history of the loss function for every 20 iterations that the
            algorithm runs for. The first index of the list is the loss function computed at iteration 0, the second
            index is the loss function computed at iteration 20, and so on and so forth. Once all the iterations are
            complete, the _loss_history list should have length n_iterations / 20.
        _X
            A numpy array of shape (n_samples, n_features) representing the input data used when fitting the model. This
            is set in the _initialize(X, y) method.
        _y
            A numpy array of shape (n_samples, n_outputs) representing the one-hot encoded target class values for the
            input data used when fitting the model.
        _h_weights
            A numpy array of shape (n_features, n_hidden) representing the weights applied between the input layer
            features and the hidden layer neurons.
        _h_bias
            A numpy array of shape (1, n_hidden) representing the weights applied between the input layer bias term
            and the hidden layer neurons.
        _o_weights
            A numpy array of shape (n_hidden, n_outputs) representing the weights applied between the hidden layer
            neurons and the output layer neurons.
        _o_bias
            A numpy array of shape (1, n_outputs) representing the weights applied between the hidden layer bias term
            neuron and the output layer neurons.
    Methods:
        _initialize(X, y)
            Function called at the beginning of fit(X, y) that performs one-hot encoding for the target class values and
            initializes the neural network weights (_h_weights, _h_bias, _o_weights, and _o_bias).
        fit(X, y)
            Fits the model to the provided data matrix X and targets y.
        predict(X)
            Predicts class target values for the given test data matrix X using the fitted classifier model.
    """

    def __init__(self, n_hidden=16, hidden_activation='sigmoid', n_iterations=1000, learning_rate=0.01):
        # Create a dictionary linking the hidden_activation strings to the functions defined in utils.py
        activation_functions = {'identity': identity, 'sigmoid': sigmoid, 'tanh': tanh, 'relu': relu}

        # Check if the provided arguments are valid
        if not isinstance(n_hidden, int) \
                or hidden_activation not in activation_functions \
                or not isinstance(n_iterations, int) \
                or not isinstance(learning_rate, float):
            raise ValueError('The provided class parameter arguments are not recognized.')

        # Define and setup the attributes for the MultilayerPerceptron model object
        self.n_hidden = n_hidden
        self.hidden_activation = activation_functions[hidden_activation]
        self.n_iterations = n_iterations
        self.learning_rate = learning_rate
        self._output_activation = softmax
        self._loss_function = cross_entropy
        self._loss_history = []

    def _initialize(self, X, y):
        """
        Function called at the beginning of fit(X, y) that performs one hot encoding for the target class values and
        initializes the neural network weights (_h_weights, _h_bias, _o_weights, and _o_bias).
        Args:
            X: A numpy array of shape (n_samples, n_features) representing the input data.
            y: A numpy array of shape (n_samples,) representing the true class values for each sample in the input data.
        Returns:
            None.
        """
        n_features = X.shape[1]
        n_outputs = y.shape[1]

        # He Normal Initializer
        std = np.sqrt(2 / n_features)

        # Initialize weights and biases for the hidden layer
        self._h_weights = np.random.normal(0, std, size=(n_features, self.n_hidden))
        self._h_bias = np.zeros((1, self.n_hidden))

        # Initialize weights and biases for the output layer
        self._o_weights = np.random.normal(0, std, size=(self.n_hidden, n_outputs))
        self._o_bias = np.zeros((1, n_outputs))

    def fit(self, X, y):
        """
        Fits the model to the provided data matrix X and targets y and stores the cross-entropy loss.
        Args:
            X: A numpy array of shape (n_samples, n_features) representing the input data.
            y: A numpy array of shape (n_samples,) representing the true class values for each sample in the input data.
        Returns:
            None.
        """
        # one-hot-encode y
        Y = one_hot_encoding(y)

        # initialize weights and biases
        self._initialize(X, Y)

        # linalg notation
        W1 = self._h_weights
        b1 = self._h_bias
        W2 = self._o_weights
        b2 = self._o_bias
        sigma = self.hidden_activation
        g = self._output_activation
        eta = self.learning_rate
        epochs = range(self.n_iterations)

        for _ in epochs:
            # Forward propagation
            Z1 = X@W1 + b1   # hidden layer outputs
            X2 = sigma(Z1)   # hidden layer activation
            Z2 = X2@W2 + b2  # output layer outputs
            Yh = g(Z2)       # output layer activation

            # Record loss
            loss = self._loss_function(Y, Yh)
            self._loss_history.append(loss)
            
            # Backward propagation
            d2 = (Yh - Y) / X.shape[0]                 # output layer error delta
            dE_dW2 = X2.T@d2                           # output layer gradient
            dE_db2 = np.ones(d2.shape[0])@d2           # output bias gradient
            d1 = d2@W2.T * sigma(Z1, derivative=True)  # hidden layer error delta
            dE_dW1 = X.T@d1                            # hidden layer gradient
            dE_db1 = np.ones(d1.shape[0])@d1           # hidden bias gradient

            # Update weights and biases by gradient descent
            W2 -= eta * dE_dW2
            b2 -= eta * dE_db2
            W1 -= eta * dE_dW1
            b1 -= eta * dE_db1

    def predict(self, X):
        """
        Predicts class target values for the given test data matrix X using the fitted classifier model.
        Args:
            X: A numpy array of shape (n_samples, n_features) representing the test data.
        Returns:
            A numpy array of shape (n_samples,) representing the predicted target class values for the given test data.
        """
        # linalg notation
        W1 = self._h_weights
        b1 = self._h_bias
        W2 = self._o_weights
        b2 = self._o_bias
        sigma = self.hidden_activation
        g = self._output_activation
        
        # Forward propagation
        Z1 = X@W1 + b1   # hidden layer outputs
        X2 = sigma(Z1)   # hidden layer activation
        Z2 = X2@W2 + b2  # output layer outputs
        Yh = g(Z2)       # output layer activation

        # Convert probabilities to class labels
        return np.argmax(Yh, axis=1)
