from seldonian.models.models import ClassificationModel
from sklearn.tree import DecisionTreeClassifier
import autograd.numpy as np

from autograd.extend import primitive, defvjp

def probs2theta(probs):
    # need to add a constant for stability in case prob=0 or 1,
    # which can happen in the decision tree.
    const = 1e-15
    probs[probs<0.5]+=const
    probs[probs>=0.5]-=const
    return np.log(1/(1/probs-1))

def sigmoid(theta):
    return 1/(1+np.exp(-1*theta))

@primitive
def sklearn_predict(theta, X, model, **kwargs):
    """Do a forward pass through the sklearn tree.

    :param theta: model weights
    :type theta: numpy ndarray
    :param X: model features
    :type X: numpy ndarray

    :param model: An instance of the Seclass 

    :return: 
        probs_pos_class: the vector of probabilities of predicting the positive class, 
        leaf_nodes_hit: the ids of the leaf nodes that were
            hit by each sample. These are needed for computing the Jacobian
    """
    # First convert weights to probs
    probs = sigmoid(theta)
    
    # Update model weights
    if not model.params_updated:
        model.set_leaf_node_values(probs, **kwargs)
        model.params_updated = True
    
    # Do the forward pass
    pred,leaf_nodes_hit = model.forward_pass(X, **kwargs)

    return pred, leaf_nodes_hit


def sklearn_predict_vjp(ans, theta, X, model):
    """Do a backward pass through the Sklearn model,
    obtaining the Jacobian d pred / dtheta.

    :param ans: The result from the forward pass
    :type ans: numpy ndarray
    :param theta: model weights
    :type theta: numpy ndarray
    :param X: model features
    :type X: numpy ndarray
    :param model: An instance of the SeldonianDecisionTree model

    :return fn: A function representing the vector Jacobian operator
    """

    def fn(v):
        # v is a vector of shape ans, the return value of the forward pass, F.
        # This function returns a 1D array:
        # [dF_i/dtheta[0],dF_i/dtheta[1],dF_i/dtheta[2],...],
        # where i is the data row index
        dpred_dtheta = model.get_jacobian(ans, theta, X)
        model.params_updated = False  # resets for the next forward pass
        return v[0].T @ dpred_dtheta
    return fn

# Link the predict function with its gradient,
# telling autograd not to look inside either of these functions
defvjp(sklearn_predict, sklearn_predict_vjp)

class SeldonianDecisionTree(ClassificationModel):
    def __init__(self,**dt_kwargs):
        """ A Seldonian decision tree model that re-labels leaf node probabilities
        from a vanilla decision tree built using SKLearn's DecisionTreeClassifier
        object. 

        :ivar classifier: The SKLearn classifier object
        :ivar has_intercept: Whether the model has an intercept term 
        :ivar params_updated: An internal flag used during the optimization
        """
        self.classifier = DecisionTreeClassifier(**dt_kwargs)
        self.has_intercept = False
        self.params_updated = False
    
    def fit(self,features,labels,**kwargs):
        """A wrapper around SKLearn's fit() method. Returns the leaf node probabilities
        of SKLearn's built tree.

        :param features: Features
        :type features: numpy ndarray
        :param labels: Labels
        :type labels: 1D numpy array

        :return: Leaf node probabilities (of predicting the positive class only),
            ordered from left to right
        """
        self.classifier.fit(features,labels)
        # Get a list of the leaf node ids
        # Node i is a leaf node if children_left[i] == -1 
        self.leaf_node_ids = np.array(
            [ii for ii in range(self.classifier.tree_.node_count) if self.classifier.tree_.children_left[ii] == -1]
        )
        return self.get_leaf_node_probs()

    def get_leaf_node_probs(self,):
        """ Retrieve the leaf node probabilities from the current tree from left to right
        """
        probs = []
        leaf_counter = 0 
        node_id = 0
        while leaf_counter < self.classifier.tree_.n_leaves:
            if self.classifier.tree_.children_left[node_id] == self.classifier.tree_.children_right[node_id]:
                # leaf node
                prob_pos = self.classifier.tree_.value[node_id][0][1]/sum(self.classifier.tree_.value[node_id][0])
                probs.append(prob_pos)
                leaf_counter += 1
            node_id += 1
        return np.array(probs)

    def set_leaf_node_values(self,probs):
        """ Update the leaf node probabilities 
            (actually the numbers in each label)

        :param probs: The vector of probabilities to set on the
            leaf nodes from left to right
        """
        leaf_counter = 0 
        node_id = 0
        while leaf_counter < self.classifier.tree_.n_leaves:
            if self.classifier.tree_.children_left[node_id] == self.classifier.tree_.children_right[node_id]:
                # leaf node
                prob_pos = probs[leaf_counter] 
                prob_neg = 1.0 - prob_pos
                num_this_leaf  = sum(self.classifier.tree_.value[node_id][0])
                n_neg_new = num_this_leaf*prob_neg
                n_pos_new = num_this_leaf*prob_pos
                self.classifier.tree_.value[node_id][0] = n_neg_new,n_pos_new
                leaf_counter += 1
            node_id += 1
        return 


    def predict(self, theta, X, **kwargs):
        """Call the autograd primitive (a workaround since our forward pass involves an external library)

        :param theta: model weights (not probabilities)
        :type theta: numpy ndarray

        :param X: model features
        :type X: numpy ndarray

        :return pred: model predictions
        :rtype pred: numpy ndarray same shape as labels
        """
        return sklearn_predict(theta, X, self)[0]

    def forward_pass(self,X):
        """Do a forward pass through the sklearn model. 
        
        :param X: model features
        :type X: numpy ndarray

        :return: 
            probs_pos_class: the vector of probabilities, 
            leaf_nodes_hit: the ids of the leaf nodes that were
                hit by each sample. These are needed for computing the Jacobian
        """
        probs_both_classes = self.classifier.predict_proba(X)
        probs_pos_class = probs_both_classes[:,1]
        # apply() provides the ids of the nodes hit by each sample in X
        leaf_nodes_hit = self.classifier.apply(X) 
        return probs_pos_class, leaf_nodes_hit

    def get_jacobian(self, ans, theta, X):
        """Return the Jacobian d(forward_pass)_i/dtheta_j,
        where i run over datapoints and j run over model parameters.

        :param ans: The result of the forward pass function evaluated on theta and X
        :param theta: The weight vector, which isn't used in this method
        :param X: The features 
        """
        pred,leaf_nodes_hit = ans
        indices = np.searchsorted(self.leaf_node_ids, leaf_nodes_hit)
        J = np.zeros((len(X),len(self.leaf_node_ids)))
        J[np.arange(len(leaf_nodes_hit)), indices] = 1
        return J
