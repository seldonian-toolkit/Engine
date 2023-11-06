from seldonian.models.models import ClassificationModel
from sklearn.ensemble import RandomForestClassifier
import autograd.numpy as np

from autograd.extend import primitive, defvjp


def probs2theta(probs):
    # need to add a constant for stability in case prob=0 or 1,
    # which can happen in the decision tree.
    const = 1e-15
    probs[probs < 0.5] += const
    probs[probs >= 0.5] -= const
    return np.log(1 / (1 / probs - 1))


def sigmoid(theta):
    return 1 / (1 + np.exp(-1 * theta))


@primitive
def sklearn_predict(theta, X, model, **kwargs):
    """Do a forward pass through the sklearn random forest.

    :param theta: model weights
    :type theta: numpy ndarray
    :param X: model features
    :type X: numpy ndarray

    :param model: An instance of a class inheriting from
            SupervisedSkLearnBaseModel

    :return:
        probs_pos_class: the vector of probabilities of predicting the positive class,
        leaf_nodes_hit: the ids of the leaf nodes that were
            hit by each sample. These are needed for computing the Jacobian
    """
    # First convert weights to probs
    probs = sigmoid(theta)
    # First update model weights
    if not model.params_updated:
        model.set_leaf_node_values(probs, **kwargs)
        model.params_updated = True
    # Do the forward pass
    pred, leaf_nodes_hit = model.forward_pass(X, **kwargs)
    # set the predictions attribute of the model

    # Predictions must be a numpy array

    return pred, leaf_nodes_hit


def sklearn_predict_vjp(ans, theta, X, model):
    """Do a backward pass through the Sklearn model,
    obtaining the Jacobian d pred / dtheta.
    Must convert back to numpy array before returning

    :param ans: The result from the forward pass
    :type ans: numpy ndarray
    :param theta: model weights
    :type theta: numpy ndarray
    :param X: model features
    :type X: numpy ndarray
    :param model: An instance of the SeldonianRandomForest model

    :return fn: A function representing the vector Jacobian operator
    """

    def fn(v):
        # v is a vector of shape ans, the return value of the forward pass()
        # This function returns a 1D array:
        # [dF_i/dtheta[0],dF_i/dtheta[1],dF_i/dtheta[2],...],
        # where i is the data row index
        dpred_dtheta = model.get_jacobian(ans, theta, X)
        model.params_updated = False  # resets for the next forward pass
        return v[0].T @ dpred_dtheta
        # return v * ans

    return fn


# Link the predict function with its gradient,
# telling autograd not to look inside either of these functions
defvjp(sklearn_predict, sklearn_predict_vjp)


class SeldonianRandomForest(ClassificationModel):
    def __init__(self, **rf_kwargs):
        """A Seldonian random forest model that re-labels leaf node probabilities
        from a vanilla decision tree built using SKLearn's RandomForestClassifier
        object.

        :ivar classifier: The SKLearn classifier object
        :ivar n_trees: The number of decision trees in the forest
        :ivar has_intercept: Whether the model has an intercept term
        :ivar params_updated: An internal flag used during the optimization
        """
        self.classifier = RandomForestClassifier(**rf_kwargs)
        self.n_trees = self.classifier.n_estimators
        self.has_intercept = False
        self.params_updated = False

    def fit(self, features, labels, **kwargs):
        """A wrapper around SKLearn's fit() method. Returns the leaf node probabilities
        of SKLearn's built trees in the forest. Assigns leaf node ids in a list of lists, where each sublist
        contains the ids for a single tree, ordered from left to right.

        :param features: Features
        :type features: numpy ndarray
        :param labels: Labels
        :type labels: 1D numpy array

        :return: Flattend array of leaf node probabilites (of predicting the positive class)
            for all trees.
        """

        self.classifier.fit(features, labels)
        # Get an array of the leaf node ids
        # Node i is a leaf node if children_left[i] == -1
        self.leaf_node_ids = []
        leaf_node_ids = []
        for estimator in self.classifier.estimators_:
            self.leaf_node_ids.append(
                [
                    ii
                    for ii in range(estimator.tree_.node_count)
                    if estimator.tree_.children_left[ii] == -1
                ]
            )
        self.n_leaf_nodes = sum([len(sublist) for sublist in self.leaf_node_ids])
        return self.get_leaf_node_probs()

    def get_leaf_node_probs(
        self,
    ):
        """Retrieve the leaf node probabilities from the current forest of trees
        from left to right.

        :return: Flattend array of leaf node probabilites (of predicting the positive class)
            for all trees.
        """
        probs = []
        for estimator in self.classifier.estimators_:
            leaf_counter = 0
            node_id = 0
            while leaf_counter < estimator.tree_.n_leaves:
                if estimator.tree_.children_left[node_id] == -1:
                    # leaf node
                    prob_pos = estimator.tree_.value[node_id][0][1] / sum(
                        estimator.tree_.value[node_id][0]
                    )
                    probs.append(prob_pos)
                    leaf_counter += 1
                node_id += 1
        return np.array(probs)

    def set_leaf_node_values(self, probs):
        """Update the leaf node values, i.e., the number of
        samples get categorized as 0 or 1, using the new
        probabilities, probs.

        :param probs: A flattened array of the leaf node probabilities
            from all trees
        """
        flat_leaf_counter = 0  # ranges from 0 to number of leaves in all trees -- needed because probs is flattened
        for estimator in self.classifier.estimators_:
            leaf_counter = 0  # ranges from 0 to number of leaves in THIS tree.
            node_id = 0  # range from 0 to number of nodes in this tree
            while leaf_counter < estimator.tree_.n_leaves:
                if estimator.tree_.children_left[node_id] == -1:
                    # leaf node
                    prob_pos = probs[flat_leaf_counter]
                    prob_neg = 1.0 - prob_pos
                    num_this_leaf = sum(estimator.tree_.value[node_id][0])
                    n_neg_new = num_this_leaf * prob_neg
                    n_pos_new = num_this_leaf * prob_pos
                    estimator.tree_.value[node_id][0] = n_neg_new, n_pos_new
                    leaf_counter += 1
                    flat_leaf_counter += 1
                node_id += 1
        return

    def predict(self, theta, X, **kwargs):
        """Call the autograd primitive (a workaround since our forward pass involves an external library)

        :param theta: model weights (not probabilities)
        :type theta: numpy ndarray

        :param X: model features
        :type X: numpy ndarray

        :return: model predictions
        :rtype: numpy ndarray same shape as labels
        """
        return sklearn_predict(theta, X, self)[0]

    def forward_pass(self, X):
        """Predict the probability of the postive class for each
        sample in X.

        :param X: Feature matrix
        """
        probs_both_classes = self.classifier.predict_proba(X)
        probs_pos_class = probs_both_classes[:, 1]
        # apply() provides the ids of the nodes hit by each sample in X
        # This will be a list of lists where each sublist contains the
        # leaf node ids hit by a single tree.
        leaf_nodes_hit = self.classifier.apply(X)
        return probs_pos_class, leaf_nodes_hit

    def get_jacobian(self, ans, theta, X):
        """Return the Jacobian d(forward_pass)_i/dtheta_{j+1},
        where i run over datapoints and j run over model parameters.
        Here, a forward pass is 1/n * sum_k { forward_k(theta,X) },
        where forward_k is the forward pass of a single decision tree.
        We can compute Jacobians for each tree separately and then horizontally stack them
        and add a 1/n out front.

        :param ans: The result of the forward pass function evaluated on theta and X
        :param theta: The weight vector, which isn't used in this method
        :param X: The features

        :return: J, the Jacobian matrix
        """
        pred, leaf_nodes_hit = ans
        # J is N x M, where N is number of samples in X and
        # M is the total number of leaf nodes in all trees
        J = np.zeros((len(X), self.n_leaf_nodes))
        n_leaves_prev_trees = 0
        for tree_index in range(self.n_trees):
            leaf_node_ids_this_tree = self.leaf_node_ids[tree_index]
            leaf_nodes_hit_this_tree = leaf_nodes_hit[:, tree_index]
            indices_this_tree = n_leaves_prev_trees + np.searchsorted(
                leaf_node_ids_this_tree, leaf_nodes_hit_this_tree
            )
            J[np.arange(len(leaf_nodes_hit_this_tree)), indices_this_tree] = 1
            n_leaves_prev_trees += len(leaf_node_ids_this_tree)
        J /= self.n_trees
        return J
