from seldonian.models.models import ClassificationModel
from sklearn.tree import DecisionTreeClassifier
import autograd.numpy as np

from autograd.extend import primitive, defvjp

@primitive
def sklearn_predict(theta, X, model, **kwargs):
    """Do a forward pass through the sklearn tree.
    Must convert back to numpy array before returning

    :param theta: model weights
    :type theta: numpy ndarray
    :param X: model features
    :type X: numpy ndarray
    :param model: SKTreeModel object

    :return (pred,leaf_nodes_hit): 
        (model predictions, array of leaf node ids encountered when each sample was forward passed)
    :rtype (pred,leaf_nodes_hit): (numpy ndarray same shape as labels, numpy ndarray same shape as labels)
    """
    # First update model weights
    if not model.params_updated:
        model.set_leaf_node_values(theta, **kwargs)
        model.params_updated = True
    # Do the forward pass
    pred,leaf_nodes_hit = model.forward_pass(X, **kwargs)
    # set the predictions attribute of the model

    # Predictions must be a numpy array

    return pred, leaf_nodes_hit


def sklearn_predict_vjp(ans, theta, X, model):
    """Do a backward pass through the sklearn decision tree,
    obtaining the Jacobian d pred / dtheta.

    :param ans: The result from the forward pass
    :type ans: numpy ndarray
    :param theta: model weights
    :type theta: numpy ndarray
    :param X: model features
    :type X: numpy ndarray

    :param model: SKTreeModel object

    :return fn: A function that calculates the vector Jacobian operator
    """

    def fn(v):
        # v is a vector of shape ans, the return value of the forward pass()
        # This function returns a 1D array containing the vector Jacobian product
        dpred_dtheta = model.backward_pass(ans, theta, X)
        model.params_updated = False  # resets for the next forward pass
        return v[0].T @ dpred_dtheta

    return fn

# Link the predict function with its gradient,
# telling autograd not to look inside either of these functions
defvjp(sklearn_predict, sklearn_predict_vjp)

class SKTreeModel(ClassificationModel):
    def __init__(self,max_depth=6):
        """ USED FOR BINARY CLASSIFICATION ONLY.
        A parametric model that builds an instance of scikit Learn's DecisionTreeClassifier: https://scikit-learn.org/stable/modules/generated/sklearn.tree.DecisionTreeClassifier.html
        and then uses the leaf node probabilities of predicting the positive class as the model parameters
        which can be optimized using KKT or black box techniques. 

        :param max_depth: Maximum depth of the tree
        :ivar classifier: An instance of the scikit-learn decision tree classifier
        """ 
        self.classifier = DecisionTreeClassifier(max_depth=max_depth)
        self.has_intercept = False
        self.params_updated = False
    
    def fit(self,features,labels,**kwargs):
        """ Build the tree and return the leaf node probabilities of predicting the positive class
        :param features: Candidate features
        :type features: numpy ndarray
        :param labels: Candidate labels 
        :type labels: 1D numpy array

        :return: Leaf node probabilities
        :rtype: 1D numpy array, one element for each leaf node.
        """
        self.classifier.fit(features,labels)
        # Get a list of the leaf node ids
        # Node i is leaf node if children_left[i] == -1 
        self.leaf_node_ids = np.array(
            [ii for ii in range(self.classifier.tree_.node_count) if self.classifier.tree_.children_left[ii] == -1]
        )
        return self.get_leaf_node_probs()

    def get_leaf_node_probs(self):
        """Get the leaf node probabilities from the current tree.

        :return: 1D numpy array, one element for each leaf node.
        """
        theta = []
        leaf_counter = 0 
        node_id = 0
        while leaf_counter < self.classifier.tree_.n_leaves:
            if self.classifier.tree_.children_left[node_id] == self.classifier.tree_.children_right[node_id]:
                # leaf node
                prob_pos = self.classifier.tree_.value[node_id][0][1]/sum(self.classifier.tree_.value[node_id][0])
                theta.append(prob_pos)
                leaf_counter += 1
            node_id += 1
        return np.array(theta)

    def set_leaf_node_values(self,theta):
        """Set the leaf node probabilities in the tree.
        
        :param theta: New leaf node probabilities to set in the tree
        :type theta: 1D numpy array
        """
        leaf_counter = 0 
        node_id = 0
        while leaf_counter < self.classifier.tree_.n_leaves:
            if self.classifier.tree_.children_left[node_id] == self.classifier.tree_.children_right[node_id]:
                # leaf node
                prob_pos = theta[leaf_counter] 
                prob_neg = 1.0 - prob_pos
                num_this_leaf  = sum(self.classifier.tree_.value[node_id][0])
                # n_neg_new = np.rint(num_this_leaf*prob_neg)
                # n_pos_new = np.rint(num_this_leaf*prob_pos)
                n_neg_new = num_this_leaf*prob_neg
                n_pos_new = num_this_leaf*prob_pos
                self.classifier.tree_.value[node_id][0] = n_neg_new,n_pos_new
                leaf_counter += 1
            node_id += 1
        return 

    def predict(self, theta, X, **kwargs):
        """Make predictions given weights and features. Wrapper to the primitive above. 

        :param theta: model weights
        :type theta: numpy ndarray
        :param X: model features
        :type X: numpy ndarray

        :return: model predictions
        :rtype: 1D numpy ndarray
        """
        return sklearn_predict(theta, X, self)[0]

    def forward_pass(self,X):
        """Do a forward pass through the model.
        :param X: model features
        :type X: numpy ndarray

        :return (probs_pos_class, leaf_nodes_hit): 
            (model predictions, array of leaf node ids encountered when each sample was forward passed)
        :rtype (probs_pos_class, leaf_nodes_hit): (numpy ndarray same shape as labels, numpy ndarray same shape as labels)
        """
        probs_both_classes = self.classifier.predict_proba(X)
        probs_pos_class = probs_both_classes[:,1]
        leaf_nodes_hit = self.classifier.apply(X)
        return probs_pos_class, leaf_nodes_hit

    def backward_pass(self, ans, theta, X):
        """Return the Jacobian d(forward_pass)_i/dtheta_j,
        where i run over datapoints and j run over model parameters.

        The forward pass returns a leaf node probability, which is an element of theta.
        The trick here is to find the indices of the leaf nodes that are hit for each sample. 
        The Jacobian has rows corresponding to samples and columns corresponding to theta, 
        and it consists entirely of 0s and 1s. 
        The element is 1 when for a row of data the theta value matches the prediction 
        for that row of data. 

        :param ans: The result from the forward pass
        :type ans: numpy ndarray
        :param theta: model weights
        :type theta: numpy ndarray
        :param X: model features
        :type X: numpy ndarray

        :return J: The Jacobian matrix
        :rtype J: Numpy ndarray of shape (len(X),len(theta))
        """
        pred,leaf_nodes_hit = ans
        indices = np.searchsorted(self.leaf_node_ids, leaf_nodes_hit)
        J = np.zeros((len(X),len(self.leaf_node_ids)))
        J[np.arange(len(leaf_nodes_hit)), indices] = 1
        return J
