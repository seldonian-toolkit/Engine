""" Main module containing Seldonian machine learning models """
import copy
import autograd.numpy as np  # Thinly-wrapped version of Numpy
from autograd.extend import primitive, defvjp
from sklearn.linear_model import LinearRegression, LogisticRegression, SGDClassifier
from functools import partial, lru_cache

from seldonian.utils.stats_utils import softmax


class SeldonianModel(object):
    def __init__(self):
        """Parent class for all machine learning models"""
        pass


class SupervisedModel(SeldonianModel):
    def __init__(self):
        """Parent class for all supervised learning models"""
        super().__init__()


class RegressionModel(SupervisedModel):
    def __init__(self):
        """Parent class for all regression-based models"""
        super().__init__()
        self.has_intercept = True

    def predict(self):
        raise NotImplementedError("Implement this method in child class")


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

import numpy as np
import graphviz
import scipy.stats
from collections import Counter
import math

from seldonian.models.models import ClassificationModel

class DTNode():
    def __init__(self):
        """ Base class for decision tree nodes """
        self.left = None
        self.right = None
        self.parent = None


class DTLeafNode(DTNode):
    def __init__(self,label):
        """ Leaf nodes just have a single label"""
        super().__init__()
        self.label = label
    
    def __repr__(self):
        return f"Leaf node label={self.label}"
        

class DTInternalNode(DTNode):
    def __init__(self,feature_name,value):
        """ Internal nodes have one feature and one value that splits the feature
        Left child nodes have feature <= value and right child nodes have feature > value
        """
        super().__init__()
        self.feature_name = feature_name
        self.value = value
    
    def __repr__(self):
        return f"Internal node feature_name={self.feature_name}, value={self.value}"
        

class SeldoDecisionTreeClassifier():
    def __init__(self,all_feature_names,num_quantiles_split=5,min_samples_split=2):
        """
        :param num_quantiles_split: The number of quantiles to use when checking values
            of a feature with float dtype
        """
        super().__init__()
        self.all_feature_names = all_feature_names
        self.num_quantiles_split = num_quantiles_split
        self.min_samples_split = min_samples_split

    def fit(self,features,labels,feature_names,candidate_dataset,parse_trees,n_safety,parent_node=None):
        """ 
        """
        
        # If all examples in the data have the same label, return a leaf node with that label
        if np.all(labels == labels[0]):
            print("making leaf node")
            return DTLeafNode(labels[0])

        # If there are no more features to split on, return a leaf node with the most common label
        if features.size == 0:
            print("making leaf node")
            most_common_label = scipy.stats.mode(labels)[0][0]
            return DTLeafNode(most_common_label)

        # Find the feature and value that give the highest information gain
        best_feature_index, best_value = self.find_best_split(
            features, 
            labels, 
            feature_names, 
            candidate_dataset,
            parse_trees,
            n_safety,
            parent_node)
        
        if not best_feature_index:
            # Was not able to find a split that resulted in at least min_samples_split samples
            most_common_label = scipy.stats.mode(labels)[0][0]
            return DTLeafNode(most_common_label)

        # Remove the best feature from the list of features and names 
        best_feature_name = feature_names[best_feature_index]
        print(f"Best feature,val: {best_feature_name}, {best_value}")
        best_feature = features[:,best_feature_index]
        features = np.delete(features,best_feature_index,1) # the final 1 indicates column-wise since features is 2D
        feature_names = np.delete(feature_names,best_feature_index) 

        # Create a decision tree node with the best feature and value
        root = DTInternalNode(best_feature_name, best_value)
        if parent_node:
            # print("Parent node exists. Setting parent.")
            root.parent = parent_node

        # Split the data into two subsets based on the best feature and value
        F_l_subset, L_l_subset, F_r_subset, L_r_subset = self.split_data(
            best_feature, best_value, features, labels)

        # Recursively build the left and right subtrees
        root.left = self.fit(F_l_subset, L_l_subset, feature_names, candidate_dataset, 
            parse_trees, n_safety, parent_node=root)
        root.right = self.fit(F_r_subset,L_r_subset, feature_names, candidate_dataset, 
            parse_trees, n_safety, parent_node=root)

        return root

    def find_best_split(
        self, 
        features, 
        labels, 
        feature_names, 
        candidate_dataset, 
        parse_trees, 
        n_safety, 
        parent_node):
        print("finding best split")
        best_feature_index = None
        best_value = None
        best_gain = 0
        # Iterate over each feature
        n_col = features.shape[1]
        for ii in range(n_col):
            feature = features[:,ii]
            # Iterate over each unique value of the feature
            for value in self.unique_values(feature):
                # Split the data into two subsets based on the feature and value
                F_l_subset, L_l_subset, F_r_subset, L_r_subset = self.split_data(feature, value, features, labels)
                if (len(L_l_subset) < self.min_samples_split) or (len(L_r_subset) < self.min_samples_split):
                    continue
                # Now add these potential children so that we can see what the tree would predict
                # if the children were there
                
                prob_pos_left = sum(L_l_subset)/len(L_l_subset) # fraction of labels with value 1    
                prob_pos_right = sum(L_r_subset)/len(L_r_subset) # fraction of labels with value 1

                if parent_node == None: 
                    this_root = DTInternalNode(feature_names[ii],value=value)
                    this_root.left = DTLeafNode(prob_pos_left)
                    this_root.right = DTLeafNode(prob_pos_right)    
                else:
                    this_parent = copy.deepcopy(parent_node)

                    this_root = self.get_projenitor(this_parent)
                    self.update_progenitor_line(this_root,prob_pos_left,prob_pos_right)
                    # print("found this_root:")
                    # print(this_root)
                    # print("with left child:")
                    # print(this_root.left)
                    # print("with right child:")
                    # print(this_root.right)
                    # print("But parent_node.left:")
                    # print(parent_node.left)
                # Calculate the information gain of the split
                # this_root=None
                e_gain = self.effective_gain(labels, L_l_subset, L_r_subset, candidate_dataset, parse_trees, n_safety, this_root)
                # remove prospective children from parent_node
                # parent_node.left = None
                # parent_node.right = None
                # If the gain is greater than the current best gain, update the best feature and value
                if e_gain > best_gain:
                    best_feature_index = ii
                    best_value = value
                    best_gain = e_gain
                
        # Edge case: no feature/val combo provided any information gain.
        # In this case, pick best feature randomly and then choose the mean value of that feature
        # to split on.
        if best_gain == 0:
            # print("No split adds information. Choosing randomly")
            n_cols = features.shape[1]
            rand_col_index = np.random.choice(n_cols)
            best_feature_index = rand_col_index
            best_feature = features[:,rand_col_index]
            best_value = np.mean(best_feature)
            # Make sure splitting on this would result in leaf nodes with at least min_samples_split samples
            L_l_subset, L_l_subset, F_r_subset, L_r_subset = self.split_data(best_feature, best_value, features, labels)
            if (len(L_l_subset) < self.min_samples_split) or (len(L_r_subset) < self.min_samples_split):
                # print("Split results in leaf nodes with not enough samples")
                return None,None
        return best_feature_index, best_value

    def unique_values(self,feature):
        step = 100/self.num_quantiles_split
        percentiles_to_check = np.linspace(step,100-step,self.num_quantiles_split-1)
        if feature.dtype == float:
            qs = np.percentile(feature,percentiles_to_check) # automatically sorted
            return qs
        else:
            return np.sort(np.unique(feature))
        
    def split_data(self, feature, value, features, labels):
        # Split features and labels on feature=val
        lmask = feature<=value
        rmask = feature>value
        F_l_subset = features[lmask]
        L_l_subset = labels[lmask]
        F_r_subset = features[rmask]
        L_r_subset = labels[rmask]
        return F_l_subset, L_l_subset, F_r_subset, L_r_subset

    def information_gain(self, labels, L_l_subset, L_r_subset):
        # Calculate the entropy of the data before the split
        original_entropy = self.entropy(labels)
        n_tot = len(labels)
        n_left = len(L_l_subset)
        n_right = len(L_r_subset)
        # Calculate the weighted average self.entropy of the left and right subsets after the split
        weighted_average_entropy = (n_left / n_tot) * self.entropy(L_l_subset) + (n_right / n_tot) * self.entropy(L_r_subset)

        # Calculate the information gain as the difference between the original entropy and the weighted average entropy
        info_gain = original_entropy - weighted_average_entropy

        return info_gain

    def effective_gain(self, labels, L_l_subset, L_r_subset, candidate_dataset, parse_trees, n_safety, root):
        # First information gain
        # print("calculating effective gain using root:")
        # print(root)
        e_gain = self.information_gain(labels, L_l_subset, L_r_subset)

        # Now check the g_upper values from the parse tree  
        # Prediction of what the safety test will return.
        # Initialized to pass
        predictSafetyTest = True
        for tree_i, pt in enumerate(parse_trees):
            # before we propagate, reset the bounds on all base nodes
            pt.reset_base_node_dict()

            bounds_kwargs = dict(
                theta=root,
                dataset=candidate_dataset,
                model=self,
                branch="candidate_selection",
                n_safety=n_safety,
                regime="supervised_learning",
            )

            pt.propagate_bounds(**bounds_kwargs)

            # Check if the i-th behavioral constraint is satisfied
            upper_bound = pt.root.upper

            if (
                upper_bound > 0.0
            ):  # If the current constraint was not satisfied, the safety test failed
                # If up until now all previous constraints passed,
                # then we need to predict that the test will fail
                # and potentially add a penalty to the objective
                print("Upper bound greater than 0")
                if predictSafetyTest:
                    # Set this flag to indicate that we don't think the safety test will pass
                    predictSafetyTest = False

                    # Put a barrier in the objective. Any solution
                    # that we think will fail the safety test
                    # will have a large cost associated with it
                    
                    penalty = -100000.0
                    e_gain += penalty
                # Add a shaping to the objective function that will
                # push the search toward solutions that will pass
                # the prediction of the safety test
                e_gain -= upper_bound

        return e_gain

    def entropy(self, labels, label_index=-1):
        # Count the number of examples with each label value
        count = Counter(labels)

        # Calculate the entropy as the sum of the probabilities of each label value times the log of the probability
        entropy = sum(-count[value] / len(labels) * math.log(count[value] / len(labels)) for value in count)

        return entropy

    def predict_single_row(self,node,row,feature_names):
        # print("in predict_single_row with node: ")
        # print(node)
        # check if we are at a leaf node
        if isinstance(node,DTLeafNode):
            return node.label
        f,v = node.feature_name,node.value
        f_index = feature_names.index(f)
        if row[f_index]<=v:
            return self.predict_single_row(node.left,row,feature_names)
        else:
            return self.predict_single_row(node.right,row,feature_names)

    def predict(self,root,X):
        # check if we are at a leaf node
        y_pred = []
        for row in X:
            y_pred.append(self.predict_single_row(root,row,self.all_feature_names))
        return np.array(y_pred)

    def get_projenitor(self,node):
        # print("in get_projenitor with node:")
        # print(node)
        if node.parent:
            return self.get_projenitor(node.parent)
        else:
            return node

    def update_progenitor_line(self,node,prob_pos_left,prob_pos_right):
        # print("in get_projenitor with node:")
        # print(node)
        if not (node.left or node.right):
            node.left = DTLeafNode(prob_pos_left)
            node.right = DTLeafNode(prob_pos_right)  
            return
        else:
            self.update_progenitor_line(node.left,prob_pos_left,prob_pos_right)
            self.update_progenitor_line(node.right,prob_pos_left,prob_pos_right)

    def print_progenitor_line(self,node):
        
        print(node)
        if node.left:
            print("node.left:")
            self.print_progenitor_line(node.left)
        if node.right:
            print("node.right:")
            self.print_progenitor_line(node.right)
        return
