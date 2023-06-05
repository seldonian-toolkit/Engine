""" Main module containing Seldonian machine learning models """

import autograd.numpy as np  # Thinly-wrapped version of Numpy
from autograd.extend import primitive, defvjp
from sklearn.linear_model import LinearRegression, LogisticRegression, SGDClassifier
from functools import partial, lru_cache

import numpy as np
import graphviz
import scipy.stats
from collections import Counter
import math

from seldonian.models import ClassificationModel

class LeafNode():
    def __init__(self,label):
        """ Leaf nodes just have a single label"""
        self.label = label
        self.left = None
        self.right = None
    def __repr__(self):
        return f"Leaf node label={self.label}"
        
class InternalNode():
    def __init__(self,feature_name,value):
        """ Internal nodes have one feature and one value that splits the feature
        Left child nodes have feature <= value and right child nodes have feature > value
        """
        self.feature_name = feature_name
        self.value = value
        self.left = None
        self.right = None
    def __repr__(self):
        return f"Internal node feature_name={self.feature_name}, value={self.value}"
        
class DecisionTreeClassifier(ClassificationModel):
    def __init__(self):
        super().__init__()

    def fit(data,features,labels,num_quantiles_split=5):
        """ 
        :param data: The entire dataframe
        :param features: The dataframe of just features 
        :param labels: The series of labels
        :param num_quantiles_split: The number of quantiles to use when checking values
            of a feature with float dtype
        """
        # If all examples in the data have the same label, return a leaf node with that label
        if np.all(labels == labels.iloc[0]):
            return LeafNode(labels.iloc[0])

        # If there are no more features to split on, return a leaf node with the most common label
        if features.empty:
            most_common_label = scipy.stats.mode(labels)[0][0]
            return LeafNode(most_common_label)

        # Find the feature and value that give the highest information gain
        best_feature_name, best_value = find_best_split(data, features, labels, num_quantiles_split)

        # Remove the best feature from the list of features
        best_feature = data[best_feature_name]
        features = features.drop(columns=[best_feature_name])
        data = data.drop(columns=[best_feature_name])

        # Create a decision tree node with the best feature and value
        tree = InternalNode(best_feature_name, best_value)
        # Split the data into two subsets based on the best feature and value
        left_subset, right_subset = split_data(data, best_feature, best_value)

        # Recursively build the left and right subtrees
        tree.left = create_decision_tree(left_subset, left_subset[features.columns], left_subset[labels.name],num_quantiles_split)
        tree.right = create_decision_tree(right_subset, left_subset[features.columns], right_subset[labels.name],num_quantiles_split)

        return tree

def find_best_split(data, features, labels, num_quantiles_split):
    best_feature_name = None
    best_value = None
    best_gain = 0

    # Iterate over each feature
    for feature_name in features:
        feature = data[feature_name]
        # Iterate over each unique value of the feature
        for value in unique_values(feature,num_quantiles_split):
            # Split the data into two subsets based on the feature and value
            left_subset, right_subset = split_data(data, feature, value)

            # Calculate the information gain of the split
            gain = information_gain(data, left_subset, right_subset, labels)
            # If the gain is greater than the current best gain, update the best feature and value
            if gain > best_gain:
                best_feature_name = feature_name
                best_value = value
                best_gain = gain

    # Edge case: no feature/val combo provided any information gain.
    # In this case, pick best feature randomly and then choose the mean value of that feature
    # to split on.
    if best_gain == 0:
        print("No split adds information. Choosing randomly")
        best_feature_name = np.random.choice(features.columns)
        best_feature = features[best_feature_name]
        best_value = np.mean(best_feature)
    return best_feature_name, best_value

def unique_values(feature,num_quantiles_split):
    step = 100/num_quantiles_split
    percentiles_to_check = np.linspace(step,100-step,num_quantiles_split-1)
    if feature.dtype == float:
        qs = np.percentile(feature,percentiles_to_check)
        return qs
    else:
        return set(np.sort(feature))
    
def split_data(data, feature, value):
    lmask = feature<=value
    l_subset = data.loc[feature<=value]
    r_subset = data.loc[feature>value]
    return l_subset,r_subset

def information_gain(data, left_subset, right_subset, label):
    # Calculate the entropy of the data before the split
    original_entropy = entropy(data)

    # Calculate the weighted average entropy of the left and right subsets after the split
    weighted_average_entropy = (len(left_subset) / len(data)) * entropy(left_subset) + (len(right_subset) / len(data)) * entropy(right_subset)

    # Calculate the information gain as the difference between the original entropy and the weighted average entropy
    info_gain = original_entropy - weighted_average_entropy

    return info_gain

def entropy(data, label_index=-1):
    # Count the number of examples with each label value
    count = Counter(data.iloc[:,label_index])

    # Calculate the entropy as the sum of the probabilities of each label value times the log of the probability
    entropy = sum(-count[value] / len(data) * math.log(count[value] / len(data)) for value in count)

    return entropy

def forward_pass_tree(node,row):
    # check if we are at a leaf node
    if isinstance(node,LeafNode):
        return node.label
    f,v = node.feature_name,node.value
    if row[f]<=v:
        return forward_pass_tree(node.left,row)
    else:
        return forward_pass_tree(node.right,row)
        


class LinearRegressionModel(RegressionModel):
    def __init__(self):
        """Implements linear regression"""
        super().__init__()
        self.model_class = LinearRegression

    def predict(self, theta, X):
        """Predict label using the linear model

        :param theta: The parameter weights
        :type theta: numpy ndarray
        :param X: The features
        :type X: numpy ndarray
        :return: predicted labels
        :rtype: numpy ndarray
        """
        return theta[0] + (X @ theta[1:])

    def fit(self, X, Y):
        """Train the model using the feature,label pairs

        :param X: features
        :type X: NxM numpy ndarray
        :param Y: labels
        :type Y: Nx1 numpy ndarray
        :return: weights from the fitted model
        :rtype: numpy ndarray
        """
        reg = self.model_class().fit(X, Y)
        return np.hstack([reg.intercept_, reg.coef_])


class LinearRegressionModelListFeatures(RegressionModel):
    def __init__(self):
        """Implements linear regression"""
        super().__init__()
        self.model_class = LinearRegression

    def predict(self, theta, X):
        """Predict label using the linear model

        :param theta: The parameter weights
        :type theta: numpy ndarray
        :param X: The features
        :type X: numpy ndarray
        :return: predicted labels
        :rtype: numpy ndarray
        """
        X_array = np.hstack(X)
        return theta[0] + (X_array @ theta[1:])

    def fit(self, X, Y):
        """Train the model using the feature,label pairs

        :param X: features
        :type X: NxM numpy ndarray
        :param Y: labels
        :type Y: Nx1 numpy ndarray
        :return: weights from the fitted model
        :rtype: numpy ndarray
        """
        X_array = np.hstack(X)
        reg = self.model_class().fit(X_array, Y)
        return np.hstack([reg.intercept_, reg.coef_])


class BoundedLinearRegressionModel(LinearRegressionModel):
    def __init__(self):
        """Implements linear regression
        with a bounded predict function.
        Overrides several parent methods.
        Assumes y-intercept is 0."""
        super().__init__()

    def _sigmoid(self, X):
        return 1 / (1 + np.exp(-X))

    def predict(self, theta, X):
        """Overrides the original predict
        function to squash predictions

        :param theta: The parameter weights
        :type theta: numpy ndarray
        :param X: The features
        :type X: numpy ndarray
        :return: predicted labels
        :rtype: numpy ndarray
        """
        y_min, y_max = -3, 3
        # Want range of Y_hat to be twice that of Y
        # and want size of interval on either side of Y_min and Y_max
        # to be the same. The unique solution to this is:
        s = 2.0  # 1 gives you the same bound size as y
        y_hat_min = y_min * (1 + s) / 2 + y_max * (1 - s) / 2
        y_hat_max = y_max * (1 + s) / 2 + y_min * (1 - s) / 2
        Z = theta[0] + (X @ theta[1:])
        return self._sigmoid(Z) * (y_hat_max - y_hat_min) + y_hat_min


class ClassificationModel(SupervisedModel):
    def __init__(self):
        """Parent class for all classification-based
        machine learning models

        Currently only supports binary classification
        """
        super().__init__()

    def predict(self):
        raise NotImplementedError("Implement this method in child class")


class BaseLogisticRegressionModel(ClassificationModel):
    def __init__(self):
        """Base class for binary and multi-class
        logistic regression"""
        super().__init__()
        self.model_class = LogisticRegression
        self.has_intercept = True

    def fit(self, X, Y):
        """Train the model using features and labels.
        Let:
                i = number of datapoints
                j = number of features (including bias term, if provied)
                k = number of classes 

        :param X: The features
        :type X: array of shape (i,j)
        :param Y: The labels
        :type Y: array of shape (i,k) 
        :return: fitted model weights
        :rtype: array of shape (j,k)
        """
        reg = self.model_class().fit(X, Y)
        theta = np.squeeze(np.vstack([reg.intercept_, reg.coef_.T]))
        return theta


class BinaryLogisticRegressionModel(BaseLogisticRegressionModel):
    def __init__(self):
        """Implements binary logistic regression"""
        super().__init__()

    def predict(self, theta, X):
        """Predict the probability of
        having the positive class label for each data point
        in X. Let:
                i = number of datapoints
                j = number of features (including bias term, if provied)

        :param theta: The parameter weights
        :type theta: array of length j or shape (j,1)
        :param X: The features
        :type X: array of shape (i,j)
        :return: predictions for each class each observation
        :rtype: array of length i or shape (i,1)
        """
        Z = theta[0] + (X @ theta[1:])  # (i,j) x (j,1) -> (i,1)
        Y_pred = 1 / (1 + np.exp(-Z))
        return Y_pred


class MultiClassLogisticRegressionModel(BaseLogisticRegressionModel):
    def __init__(self):
        """Implements multi-class
        logistic regression"""
        super().__init__()

    def predict(self, theta, X):
        """Predict the probability of
        having each class label for each data point
        in X. Let:
                i = number of datapoints
                j = number of features (including bias term, if provied)
                k = number of classes

        :param theta: The parameter weights
        :type theta: array of shape (j,k)
        :param X: The features
        :type X: array of shape (i,j)
        :return: predictions for each class each observation
        :rtype: array of shape (i x k)
        """
        Z = theta[0] + (X @ theta[1:])  # (i,j) x (j,k) -> (i,k)
        # softmax to get probabilites
        Y_pred = np.exp(Z) / np.sum(np.exp(Z), axis=-1, keepdims=True)

        return Y_pred


class DummyClassifierModel(ClassificationModel):
    def __init__(self):
        """Implements a classifier that always predicts
        the positive class, regardless of input"""
        super().__init__()
        self.has_intercept = False

    def predict(self, theta, X):
        """Predict the probability of
        having the positive class label

        :param theta: The parameter weights
        :type theta: numpy ndarray
        :param X: The features
        :type X: numpy ndarray
        :return: predictions for each observation
        :rtype: float
        """

        return np.ones(len(X))


class RandomClassifierModel(ClassificationModel):
    def __init__(self):
        """Implements a classifier that always predicts
        that the positive class has prob=0.5,
        regardless of input"""
        super().__init__()
        self.has_intercept = False

    def predict(self, theta, X):
        """Predict the probability of
        having the positive class label

        :param theta: The parameter weights
        :type theta: numpy ndarray
        :param X: The features
        :type X: numpy ndarray
        :return: predictions for each observation
        :rtype: float
        """
        return 0.5 * np.ones(len(X))
