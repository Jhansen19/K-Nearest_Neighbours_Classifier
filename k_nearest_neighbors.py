# k_nearest_neighbors.py: Machine learning implementation of a K-Nearest Neighbors classifier from scratch.
#
# Submitted by: [Jonathan Hansen & Mark Green] -- [margree-jonjhans-a5]
#
# Based on skeleton code by CSCI-B 551 Fall 2023 Course Staff
#
# Resources for Jon:
# https://chat.openai.com/

import numpy as np
from utils import euclidean_distance, manhattan_distance


class KNearestNeighbors:
    """
    A class representing the machine learning implementation of a K-Nearest Neighbors classifier from scratch.
    Attributes:
        n_neighbors
            An integer representing the number of neighbors a sample is compared with when predicting target class
            values.
        weights
            A string representing the weight function used when predicting target class values. The possible options are
            {'uniform', 'distance'}.
        _X
            A numpy array of shape (n_samples, n_features) representing the input data used when fitting the model and
            predicting target class values.
        _y
            A numpy array of shape (n_samples,) representing the true class values for each sample in the input data
            used when fitting the model and predicting target class values.
        _distance
            An attribute representing which distance metric is used to calculate distances between samples. This is set
            when creating the object to either the euclidean_distance or manhattan_distance functions defined in
            utils.py based on what argument is passed into the metric parameter of the class.
    Methods:
        fit(X, y)
            Fits the model to the provided data matrix X and targets y.
        predict(X)
            Predicts class target values for the given test data matrix X using the fitted classifier model.
    """

    def __init__(self, n_neighbors = 5, weights = 'uniform', metric = 'l2'):
        # Check if the provided arguments are valid
        if weights not in ['uniform', 'distance'] or metric not in ['l1', 'l2'] or not isinstance(n_neighbors, int):
            raise ValueError('The provided class parameter arguments are not recognized.')

        # Define and setup the attributes for the KNearestNeighbors model object
        self.n_neighbors = n_neighbors
        self.weights = weights
        self._X = None
        self._y = None
        self._distance = euclidean_distance if metric == 'l2' else manhattan_distance

    def fit(self, X, y):
        """
        Fits the model to the provided data matrix X and targets y.
        Args:
            X: A numpy array of shape (n_samples, n_features) representing the input data.
            y: A numpy array of shape (n_samples,) representing the true class values for each sample in the input data.
        Returns:
            None.
        """
        self._X = X
        self._y = y

    def predict(self, X):
        """
        Predicts class target values for the given test data matrix X using the fitted classifier model.
        Args:
            X: A numpy array of shape (n_samples, n_features) representing the test data.
        Returns:
            A numpy array of shape (n_samples,) representing the predicted target class values for the given test data.
        """

        predictions = []
        for x_test in X:
            # Calculate distances from x_test to all training samples
            distances = np.array([self._distance(x_test, x_train) for x_train in self._X])
            distances = np.where(distances == 0, 1e-15, distances)

            # Find the indices of the n_neighbors closest samples
            neighbors_indices = np.argsort(distances)[:self.n_neighbors]

            # Retrieve the labels of the closest samples
            neighbors_labels = self._y[neighbors_indices]

            if self.weights == 'uniform':
                # Majority voting
                predicted_label = np.bincount(neighbors_labels).argmax()
            else:
                # Weighted voting (inverse of distance)
                weights = 1 / distances[neighbors_indices]
                predicted_label = np.bincount(neighbors_labels, weights=weights).argmax()

            predictions.append(predicted_label)

        return np.array(predictions)
