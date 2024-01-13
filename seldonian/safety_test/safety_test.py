""" Module for running safety test """

import autograd.numpy as np  # Thinly-wrapped version of Numpy
import copy


class SafetyTest(object):
    def __init__(
        self,
        safety_dataset,
        model,
        parse_trees,
        regime="supervised_learning",
        additional_datasets={},
        **kwargs
    ):
        """
        Object for running safety test

        :param safety_dataset: The dataset object containing safety data
        :type safety_dataset: :py:class:`.DataSet` object

        :param model: The Seldonian model object
        :type model: :py:class:`.SeldonianModel` object

        :param parse_trees: List of parse tree objects containing the
                behavioral constraints
        :type parse_trees: List(:py:class:`.ParseTree` objects)

        :param regime: The category of the machine learning algorithm,
                e.g., supervised_learning or reinforcement_learning
        :type regime: str

        :param additional_datasets: Specifies optional additional datasets to use
            for bounding the base nodes of the parse trees.
        :type additional_datasets: dict, defaults to {}
        """
        self.safety_dataset = safety_dataset
        self.model = model
        self.parse_trees = parse_trees
        self.regime = regime
        self.additional_datasets = additional_datasets

        self.st_result = {}  # stores parse tree evaluated on safety test data

    def run(self, solution, batch_size_safety=None, **kwargs):
        """Loop over parse trees, calculate the bounds on leaf nodes
        and propagate to the root node. The safety test passes if
        the upper bounds of all parse tree root nodes are less than or equal to 0.

        :param solution:
                The candidate solution found by candidate selection
        :type solution: numpy ndarray

        :param batch_size_safety: The number of datapoints
                to pass through the model in a single forward pass
        :type batch_size_safety: int

        :return: passed, whether the candidate solution passed the safety test
        :rtype: bool
        """
        passed = True

        for tree_i, pt in enumerate(self.parse_trees):
            # before we propagate reset the tree
            pt.reset_base_node_dict()

            cstr = pt.constraint_str
            if cstr in self.additional_datasets:
                dataset_dict = {
                    bn: self.additional_datasets[cstr][bn]["safety_dataset"]
                    for bn in self.additional_datasets[cstr]
                }
            else:
                dataset_dict = {"all": self.safety_dataset}

            bounds_kwargs = dict(
                theta=solution,
                tree_dataset_dict=dataset_dict,
                model=self.model,
                branch="safety_test",
                regime=self.regime,
                batch_size_safety=batch_size_safety,
                sub_regime=self.safety_dataset.meta.sub_regime,
                **kwargs
            )

            pt.propagate_bounds(**bounds_kwargs)
            # Check if the i-th behavioral constraint is satisfied
            upperBound = pt.root.upper
            self.st_result[pt.constraint_str] = copy.deepcopy(pt)
            if (
                upperBound > 0.0
            ):  # If the current constraint was not satisfied, the safety test failed
                passed = False

        return passed

    def evaluate_primary_objective(self, theta, primary_objective):
        """Get value of the primary objective given model weights
        theta, and the safety data, D_s. This is a wrapper for primary_objective where
        data is fixed.

        :param theta: model weights
        :type theta: numpy.ndarray

        :param primary_objective: The primary objective function
                you want to evaluate

        :return: The primary objective function evaluated at theta, D_s
        :rtype: float
        """
        # Get value of the primary objective given model weights
        if self.regime == "supervised_learning":
            result = primary_objective(
                self.model,
                theta,
                self.safety_dataset.features,
                self.safety_dataset.labels,
            )
            return result

        elif self.regime == "reinforcement_learning":
            # Want to maximize the importance weight so minimize negative importance weight
            # Adding regularization term so that large thetas make this less negative
            # and therefore worse
            result = -1.0 * primary_objective(
                model=self.model,
                theta=theta,
                episodes=self.safety_dataset.episodes,
                weighted_returns=None,
            )

            if hasattr(self, "reg_coef"):
                # reg_term = self.reg_coef*np.linalg.norm(theta)
                reg_term = self.reg_coef * np.dot(theta.T, theta)
            else:
                reg_term = 0
            result += reg_term
            return result

        elif self.regime == "custom":
            result = primary_objective(self.model, theta, self.safety_dataset.data,)
            return result

    def get_importance_weights(self, theta):
        """Get an array of importance weights given model weights, theta,
        and the safety data, D_s. Only relevant for RL.

        :param theta: model weights
        :type theta: numpy.ndarray

        :return: The importance weights 
        :rtype: numpy.ndarray
        """
        assert self.regime == "reinforcement_learning"
        rho_is = []

        for ii, ep in enumerate(self.safety_dataset.episodes):
            pi_news = self.model.get_probs_from_observations_and_actions(
                theta, ep.observations, ep.actions, ep.action_probs
            )
            rho_is.append(np.prod(pi_news / ep.action_probs))

        return np.array(rho_is)
