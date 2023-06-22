""" Candidate selection module """

import os, pickle
import autograd.numpy as np  # Thinly-wrapped version of Numpy
import math
import pandas as pd
from functools import partial

from seldonian.models import objectives
from seldonian.dataset import SupervisedDataSet, RLDataSet


class CandidateSelection(object):
    def __init__(
        self,
        model,
        candidate_dataset,
        n_safety,
        parse_trees,
        primary_objective,
        optimization_technique="barrier_function",
        optimizer="Powell",
        initial_solution=None,
        regime="supervised_learning",
        write_logfile=False,
        **kwargs,
    ):
        """Object for running candidate selection

        :param model: The Seldonian model object
        :type model: models.model.SeldonianModel object

        :param candidate_dataset: The dataset object containing candidate data
        :type candidate_dataset: dataset.Dataset object

        :param n_safety: The length of the safety dataset, used
                when predicting confidence bounds during candidate selection
        :type n_safety: int

        :param parse_trees: List of parse tree objects containing the
                behavioral constraints
        :type parse_trees: List(parse_tree.ParseTree objects)

        :param primary_objective: The objective function that would
                be solely optimized in the absence of behavioral constraints,
                i.e. the loss function
        :type primary_objective: function or class method

        :param optimization_technique: The method for optimization during
                candidate selection. E.g. 'gradient_descent', 'barrier_function'
        :type optimization_technique: str

        :param optimizer: The string name of the optimizer used
                during candidate selection
        :type optimizer: str

        :param initial_solution: The model weights used to initialize
                the optimizer
        :type initial_solution: array

        :param regime: The category of the machine learning algorithm,
                        e.g., supervised_learning or reinforcement_learning
        :type regime: str

        :param write_logfile: Whether to write outputs of candidate selection
                to disk
        :type write_logfile: bool

        """
        self.regime = regime
        self.model = model
        self.candidate_dataset = candidate_dataset
        self.n_safety = n_safety
        if self.regime == "supervised_learning":
            self.features = self.candidate_dataset.features
            self.labels = self.candidate_dataset.labels

        self.parse_trees = parse_trees
        self.primary_objective = (
            primary_objective  # must accept theta, features, labels
        )
        self.optimization_technique = optimization_technique
        self.optimizer = optimizer
        self.initial_solution = initial_solution
        self.candidate_solution = None
        self.write_logfile = write_logfile

        if "reg_coef" in kwargs:
            self.reg_coef = kwargs["reg_coef"]

    def calculate_batches(self, batch_index, batch_size):
        """Create a batch dataset to be used in gradient descent.
        Does not return anything, instead sets self.batch_dataset.

        :param batch_index: The batch number (0-indexed)
        :type batch_index: int
        :param batch_size: The size of the batches 
        :type batch_size: int
        
        :return: None
        """
        batch_start = batch_index * batch_size
        batch_end = batch_start + batch_size

        num_datapoints = self.candidate_dataset.num_datapoints
        if self.regime == "supervised_learning":
            if batch_size < num_datapoints:
                if type(self.features) == list:
                    self.batch_features = [
                        x[batch_start:batch_end] for x in self.features
                    ]
                    batch_num_datapoints = len(self.batch_features[0])
                else:
                    self.batch_features = self.features[batch_start:batch_end]
                    batch_num_datapoints = len(self.batch_features)

                self.batch_labels = self.labels[batch_start:batch_end]
                self.batch_sensitive_attrs = self.candidate_dataset.sensitive_attrs[
                    batch_start:batch_end
                ]
            else:
                self.batch_features = self.features
                self.batch_labels = self.labels
                self.batch_sensitive_attrs = self.candidate_dataset.sensitive_attrs
                self.batch_dataset = self.candidate_dataset
                batch_num_datapoints = num_datapoints

            self.batch_dataset = SupervisedDataSet(
                self.batch_features,
                self.batch_labels,
                self.batch_sensitive_attrs,
                num_datapoints=batch_num_datapoints,
                meta=self.candidate_dataset.meta,
            )

        elif self.regime == "reinforcement_learning":
            if batch_size < num_datapoints:
                batch_episodes = self.candidate_dataset.episodes[batch_start:batch_end]
                batch_num_datapoints = len(batch_episodes)
                self.batch_sensitive_attrs = self.candidate_dataset.sensitive_attrs[
                    batch_start:batch_end
                ]
            else:
                batch_episodes = self.candidate_dataset.episodes
                batch_num_datapoints = num_datapoints
                self.batch_sensitive_attrs = self.candidate_dataset.sensitive_attrs

            self.batch_dataset = RLDataSet(
                episodes=batch_episodes,
                sensitive_attrs=self.batch_sensitive_attrs,
                num_datapoints=batch_num_datapoints,
                meta=self.candidate_dataset.meta,
            )
        # If this batch is smaller than the batch size and not the first batch
        # then that means we shouldn't consider a candidate solution calculated from it 
        if batch_index > 0 and (batch_num_datapoints < batch_size):
            return True
        else: 
            return False

    def run(self, **kwargs):
        """Run candidate selection

        :return: Optimized model weights or 'NSF'
        :rtype: array or str
        """
        if self.optimization_technique == "gradient_descent":
            if self.optimizer != "adam":
                raise NotImplementedError(
                    f"Optimizer: {self.optimizer} is not supported"
                )

            from seldonian.optimizers.gradient_descent import gradient_descent_adam

            # Figure out number of batches
            if "use_batches" not in kwargs:
                raise KeyError(
                    "'use_batches' key is required in optimization_hyperparameters dictionary"
                )

            if kwargs["use_batches"] == True:
                batch_size = kwargs["batch_size"]
                n_batches = math.ceil(
                    self.candidate_dataset.num_datapoints / batch_size
                )
                n_epochs = kwargs["n_epochs"]
            else:
                if "num_iters" not in kwargs:
                    raise KeyError(
                        "'num_iters' key is required in optimization_hyperparameters dictionary"
                        "if 'use_batches' == False"
                    )
                n_batches = 1
                batch_size = self.candidate_dataset.num_datapoints
                n_epochs = kwargs["num_iters"]

            gd_kwargs = dict(
                primary_objective=self.evaluate_primary_objective,
                n_constraints=len(self.parse_trees),
                upper_bounds_function=self.get_constraint_upper_bounds,
                n_batches=n_batches,
                batch_size=batch_size,
                n_epochs=n_epochs,
                batch_calculator=self.calculate_batches,
                gradient_library=kwargs["gradient_library"],
                alpha_theta=kwargs["alpha_theta"],
                alpha_lamb=kwargs["alpha_lamb"],
                beta_velocity=kwargs["beta_velocity"],
                beta_rmsprop=kwargs["beta_rmsprop"],
                theta_init=self.initial_solution,
                lambda_init=kwargs["lambda_init"],
                verbose=kwargs["verbose"],
                debug=kwargs["debug"],
            )
            # Option to use builtin primary gradient (could be faster than autograd)
            if "use_builtin_primary_gradient_fn" in kwargs:
                if kwargs["use_builtin_primary_gradient_fn"] == True:
                    if self.regime == "supervised_learning":
                        # need to know name of primary objective first
                        primary_objective_name = self.primary_objective.__name__
                        grad_primary_objective = getattr(
                            objectives, f"gradient_{primary_objective_name}"
                        )

                        # Now fix the features and labels so that the function
                        # is only a function of theta

                        def grad_primary_objective_theta(theta, **kwargs):
                            return grad_primary_objective(
                                model=self.model,
                                theta=theta,
                                X=self.batch_features,
                                Y=self.batch_labels,
                            )

                        gd_kwargs["primary_gradient"] = grad_primary_objective_theta
                    else:
                        raise NotImplementedError(
                            "Using a builtin primary objective gradient"
                            " is not yet supported for regimes other"
                            " than supervised learning"
                        )

            # If user specified the gradient of the primary
            # objective, then pass it here
            if kwargs["custom_primary_gradient_fn"] != None:
                if self.regime == "supervised_learning":
                    # need to know name of primary objective first
                    grad_primary_objective = kwargs["custom_primary_gradient_fn"]

                    def grad_primary_objective_theta(theta):
                        return grad_primary_objective(
                            model=self.model,
                            theta=theta,
                            X=self.batch_features,
                            Y=self.batch_labels,
                        )

                    gd_kwargs["primary_gradient"] = grad_primary_objective_theta
                else:
                    raise NotImplementedError(
                        "Using a provided primary objective gradient"
                        " is not yet supported for regimes other"
                        " than supervised learning"
                    )

            res = gradient_descent_adam(**gd_kwargs)
            self.optimization_result = res
            res["constraint_strs"] = [pt.constraint_str for pt in self.parse_trees]
            res["batch_size"] = batch_size
            res["n_epochs"] = n_epochs

            if self.write_logfile:
                log_counter = 0
                logdir = os.path.join(os.getcwd(), "logs")
                os.makedirs(logdir, exist_ok=True)
                filename = os.path.join(
                    logdir, f"candidate_selection_log{log_counter}.p"
                )

                while os.path.exists(filename):
                    filename = filename.replace(
                        f"log{log_counter}", f"log{log_counter+1}"
                    )
                    log_counter += 1
                with open(filename, "wb") as outfile:
                    pickle.dump(res, outfile)
                    print(f"Wrote {filename} with candidate selection log info")

            candidate_solution = res["candidate_solution"]

        elif self.optimization_technique == "barrier_function":
            if self.regime == "reinforcement_learning":
                raise NotImplementedError(
                    "barrier_function optimization_technique "
                    "is not supported for reinforcement learning. "
                    "Use gradient_descent instead."
                )
            opts = {}
            if "maxiter" in kwargs:
                opts["maxiter"] = kwargs["maxiter"]

            if self.optimizer in ["Powell", "CG", "Nelder-Mead", "BFGS"]:
                from scipy.optimize import minimize

                res = minimize(
                    self.objective_with_barrier,
                    x0=self.initial_solution,
                    method=self.optimizer,
                    options=opts,
                    args=(),
                )

                candidate_solution = res.x
                self.optimization_result = res

            elif self.optimizer == "CMA-ES":
                import cma

                if "seed" in kwargs:
                    opts["seed"] = kwargs["seed"]

                es = cma.CMAEvolutionStrategy(self.initial_solution, 0.2, opts)

                es.optimize(self.objective_with_barrier)
                candidate_solution = es.result.xbest
                self.optimization_result = es.result
            else:
                raise NotImplementedError(
                    f"Optimizer: {self.optimizer} is not supported"
                )
        else:
            raise NotImplementedError(
                f"Optimization technique: {self.optimization_technique} is not implemented"
            )

        # Reset parse tree base node dicts,
        # including data and datasize attributes
        for pt in self.parse_trees:
            pt.reset_base_node_dict(reset_data=True)

        # Unset data and datasize on base nodes
        # Return the candidate solution we believe will pass the safety test
        return candidate_solution

    def objective_with_barrier(self, theta):
        """The objective function to be optimized if
        minimization_technique == 'barrier'. Adds in a
        large penalty when any of the constraints are violated.

        :param theta: model weights
        :type theta: numpy.ndarray

        :return: the value of the objective function
                evaluated at theta
        """

        # Get the primary objective evaluated at the given theta
        # and the entire candidate dataset
        if self.regime == "supervised_learning":
            result = self.primary_objective(
                self.model, theta, self.features, self.labels
            )

        # elif self.regime == 'reinforcement_learning':
        # 	data_dict = {'episodes':self.candidate_dataset.episodes}
        # 	# Want to maximize the importance weight so minimize negative importance weight
        # 	result = -1.0*self.primary_objective(self.model,theta,
        # 		data_dict)

        # Optionally adding regularization term so that large thetas
        # make this less negative
        # and therefore worse
        if hasattr(self, "reg_coef"):
            reg_term = self.reg_coef * np.linalg.norm(theta)
            result += reg_term

        # Prediction of what the safety test will return.
        # Initialized to pass
        predictSafetyTest = True
        for tree_i, pt in enumerate(self.parse_trees):
            # before we propagate, reset the bounds on all base nodes
            pt.reset_base_node_dict()

            bounds_kwargs = dict(
                theta=theta,
                dataset=self.candidate_dataset,
                model=self.model,
                branch="candidate_selection",
                n_safety=self.n_safety,
                regime=self.regime,
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
                if predictSafetyTest:
                    # Set this flag to indicate that we don't think the safety test will pass
                    predictSafetyTest = False

                    # Put a barrier in the objective. Any solution
                    # that we think will fail the safety test
                    # will have a large cost associated with it
                    if self.optimization_technique == "barrier_function":
                        result = 100000.0
                # Add a shaping to the objective function that will
                # push the search toward solutions that will pass
                # the prediction of the safety test

                result = result + upper_bound

        return result

    def evaluate_primary_objective(self, theta):
        """Get value of the primary objective given model weights,
        theta. Wrapper for self.primary_objective where
        data is fixed. Used as input to gradient descent

        :param theta: model weights
        :type theta: numpy.ndarray

        :return: The value of the primary objective function
                evaluated at theta
        """

        if self.regime == "supervised_learning":
            result = self.primary_objective(
                self.model, theta, self.batch_features, self.batch_labels
            )

        elif self.regime == "reinforcement_learning":
            # Want to maximize the importance weight so minimize negative importance weight
            # Adding regularization term so that large thetas make this less negative
            # and therefore worse
            result = -1.0 * self.primary_objective(
                model=self.model,
                theta=theta,
                episodes=self.batch_dataset.episodes,
                weighted_returns=None,
            )

        if hasattr(self, "reg_coef"):
            # reg_term = self.reg_coef*np.linalg.norm(theta)
            reg_term = self.reg_coef * np.dot(theta.T, theta)
        else:
            reg_term = 0

        result += reg_term
        return result

    def get_constraint_upper_bounds(self, theta):
        """Get value of the upper bounds of the constraint functions
        given model weights, theta. Used as input to gradient descent.

        :param theta: model weights
        :type theta: numpy.ndarray

        :return: Array of upper bounds on the constraint
        :rtype: array
        """

        upper_bounds = []

        for pt in self.parse_trees:
            pt.reset_base_node_dict()

            bounds_kwargs = dict(
                theta=theta,
                dataset=self.batch_dataset,
                model=self.model,
                branch="candidate_selection",
                n_safety=self.n_safety,
                regime=self.regime,
            )

            pt.propagate_bounds(**bounds_kwargs)
            upper_bounds.append(pt.root.upper)

        return np.array(upper_bounds, dtype="float")
