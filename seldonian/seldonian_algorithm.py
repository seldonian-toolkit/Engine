""" Module for running Seldonian algorithms """
import copy

from sklearn.model_selection import train_test_split
import autograd.numpy as np  # Thinly-wrapped version of Numpy

import warnings
from seldonian.warnings.custom_warnings import *
from seldonian.dataset import SupervisedDataSet, RLDataSet
from seldonian.candidate_selection.candidate_selection import CandidateSelection
from seldonian.safety_test.safety_test import SafetyTest
from seldonian.models import objectives


class SeldonianAlgorithm:
    def __init__(self, spec):
        """Object for running the Seldonian algorithm and getting
        introspection into candidate selection and safety test

        :param spec: The specification object with the complete
                set of parameters for running the Seldonian algorithm
        :type spec: :py:class:`.Spec` object
        """
        self.spec = spec
        self.cs_has_been_run = False
        self.cs_result = None
        self.st_has_been_run = False
        self.st_result = None

        self.parse_trees = self.spec.parse_trees
        # user can pass a dictionary that specifies
        # the bounding method for each base node
        # any base nodes not in this dictionary will
        # be bounded using the default method
        self.base_node_bound_method_dict = self.spec.base_node_bound_method_dict
        if self.base_node_bound_method_dict != {}:
            all_pt_constraint_strs = [pt.constraint_str for pt in self.parse_trees]
            for constraint_str in self.base_node_bound_method_dict:
                this_bound_method_dict = self.base_node_bound_method_dict[
                    constraint_str
                ]
                # figure out which parse tree this comes from
                this_pt_index = all_pt_constraint_strs.index(constraint_str)
                this_pt = self.parse_trees[this_pt_index]
                # change the bound method for each node provided
                for node_name in this_bound_method_dict:
                    this_pt.base_node_dict[node_name][
                        "bound_method"
                    ] = this_bound_method_dict[node_name]

        self.dataset = self.spec.dataset
        self.regime = self.dataset.regime
        self.column_names = self.dataset.meta_information


        if self.spec.primary_objective is None:
            if self.regime == "reinforcement_learning":
                self.spec.primary_objective = objectives.IS_estimate
            elif self.regime == "supervised_learning":
                if self.spec.sub_regime in ["classification", "binary_classification"]:
                    self.spec.primary_objective = objectives.binary_logistic_loss
                elif self.spec.sub_regime == "multiclass_classification":
                    self.spec.primary_objective = objectives.multiclass_logistic_loss
                elif self.spec.sub_regime == "regression":
                    self.spec.primary_objective = objectives.Mean_Squared_Error

        # Create or load candidate selection and safety data splits.
        if self.regime == "supervised_learning":
            self.sub_regime = self.spec.sub_regime
            self.model = self.spec.model

            # Split dataset into candidate and safety datasets.
            (
                self.candidate_features,
                self.safety_features,
                self.candidate_labels,
                self.safety_labels,
                self.candidate_sensitive_attrs,
                self.safety_sensitive_attrs,
                self.n_candidate,
                self.n_safety,
            ) = self.candidate_safety_split(self.spec.frac_data_in_safety)
            self.candidate_dataset = SupervisedDataSet(
                features=self.candidate_features,
                labels=self.candidate_labels,
                sensitive_attrs=self.candidate_sensitive_attrs,
                num_datapoints=self.n_candidate,
                meta_information=self.dataset.meta_information,
            )
            self.safety_dataset = SupervisedDataSet(
                features=self.safety_features,
                labels=self.safety_labels,
                sensitive_attrs=self.safety_sensitive_attrs,
                num_datapoints=self.n_safety,
                meta_information=self.dataset.meta_information,
            )

            # Warnings.
            if self.n_candidate < 2 or self.n_safety < 2:
                warning_msg = (
                    "Warning: not enough data to " "run the Seldonian algorithm."
                )
                warnings.warn(warning_msg)
            if self.spec.verbose:
                print(f"Safety dataset has {self.n_safety} datapoints")
                print(f"Candidate dataset has {self.n_candidate} datapoints")

        elif self.regime == "reinforcement_learning":
            self.model = self.spec.model

            if ((self.spec.candidate_dataset is None) or (self.spec.safety_dataset is None)
                    or (self.spec.datasplit_info is None)):
                # Split into candidate and safety datasets if not done already in spec.
                (
                    self.candidate_episodes,
                    self.safety_episodes,
                    self.candidate_sensitive_attrs,
                    self.safety_sensitive_attrs,
                    self.n_candidate,
                    self.n_safety,
                ) = self.candidate_safety_split(self.spec.frac_data_in_safety)

                self.candidate_dataset = RLDataSet(
                    episodes=self.candidate_episodes,
                    sensitive_attrs=self.candidate_sensitive_attrs,
                    meta_information=self.column_names,
                )

                self.safety_dataset = RLDataSet(
                    episodes=self.safety_episodes,
                    sensitive_attrs=self.safety_sensitive_attrs,
                    meta_information=self.column_names,
                )

            print(f"Safety dataset has {self.n_safety} episodes")
            print(f"Candidate dataset has {self.n_candidate} episodes")
            print("Candidate sensitive_attrs:")

        else:
            # Load datasplit and datasplit information from the spec.
            self.candidate_dataset = self.spec.candidate_dataset
            self.safety_dataset = self.spec.safety_dataset
            self.candidate_episodes = self.spec.datasplit_info["candidate_episodes"]
            self.safety_episodes = self.spec.datasplit_info["safety_episodes"]
            self.candidate_sensitive_attrs = self.spec.datasplit_info["candidate_sensitive_attrs"]
            self.safety_sensitive_attrs = self.spec.datasplit_info["safety_sensitive_attrs"]
            self.n_candidate = self.spec.datasplit_info["n_candidate"]
            self.n_safety = self.spec.datasplit_info["n_safety"]


    def candidate_safety_split(self, frac_data_in_safety):
        """Split features, labels and sensitive attributes
        into candidate and safety sets

        :param frac_data_in_safety: Fraction of data used in safety test.
                The remaining fraction will be used in candidate selection

        :return: F_c,F_s,L_c,L_s,S_c,S_s
                where F=features, L=labels, S=sensitive attributes
        """
        n_points_tot = self.dataset.num_datapoints
        n_candidate = int(round(n_points_tot * (1.0 - frac_data_in_safety)))
        n_safety = n_points_tot - n_candidate

        if self.regime == "supervised_learning":
            # Split features
            if type(self.dataset.features) == list:
                F_c = [x[:n_candidate] for x in self.dataset.features]
                F_s = [x[n_candidate:] for x in self.dataset.features]
            else:
                F_c = self.dataset.features[:n_candidate]
                F_s = self.dataset.features[n_candidate:]
            # Split labels - must be numpy array
            L_c = self.dataset.labels[:n_candidate]
            L_s = self.dataset.labels[n_candidate:]

            # Split sensitive attributes - must be numpy array
            S_c = self.dataset.sensitive_attrs[:n_candidate]
            S_s = self.dataset.sensitive_attrs[n_candidate:]
            return F_c, F_s, L_c, L_s, S_c, S_s, n_candidate, n_safety

        elif self.regime == "reinforcement_learning":
            # Split episodes
            E_c = self.dataset.episodes[0:n_candidate]
            E_s = self.dataset.episodes[n_candidate:]

            # Split sensitive attributes - must be numpy array
            S_c = self.dataset.sensitive_attrs[:n_candidate]
            S_s = self.dataset.sensitive_attrs[n_candidate:]
            return E_c, E_s, S_c, S_s, n_candidate, n_safety

    def candidate_selection(self, write_logfile=False):
        """Create the candidate selection object

        :param write_logfile: Whether to write out a pickle file
                containing details of candidate selection
        """
        cs_kwargs = dict(
            model=self.model,
            candidate_dataset=self.candidate_dataset,
            n_safety=self.n_safety,
            parse_trees=self.parse_trees,
            primary_objective=self.spec.primary_objective,
            optimization_technique=self.spec.optimization_technique,
            optimizer=self.spec.optimizer,
            initial_solution=self.initial_solution,
            regime=self.regime,
            write_logfile=write_logfile,
        )

        cs = CandidateSelection(**cs_kwargs, **self.spec.regularization_hyperparams)

        return cs

    def safety_test(self):
        """Create the safety test object"""
        st_kwargs = dict(
            safety_dataset=self.safety_dataset,
            model=self.model,
            parse_trees=self.spec.parse_trees,
            regime=self.regime,
        )

        st = SafetyTest(**st_kwargs)
        return st

    def set_initial_solution(self, verbose=False):
        if self.regime == "supervised_learning":
            if self.spec.initial_solution_fn is None:
                if verbose:
                    print(
                        "No initial_solution_fn provided. "
                        "Attempting to initialize with a zeros matrix "
                        " of the correct shape"
                    )
                n_features = self.candidate_dataset.n_features
                if self.model.has_intercept:
                    n_features += 1
                if self.sub_regime == "multiclass_classification":
                    n_classes = len(np.unique(self.candidate_labels))
                    self.initial_solution = np.zeros((n_features, n_classes))
                else:
                    self.initial_solution = np.zeros(n_features)

            else:
                self.initial_solution = self.spec.initial_solution_fn(
                    self.candidate_features, self.candidate_labels
                )
        elif self.regime == "reinforcement_learning":
            if self.spec.initial_solution_fn is None:
                if verbose:
                    print(
                        "No initial_solution_fn provided. "
                        "Attempting to get initial weights from policy"
                    )
                self.initial_solution = self.model.policy.get_params()
            else:
                self.initial_solution = self.spec.initial_solution_fn(
                    self.candidate_dataset
                )

        if verbose:
            print("Initial solution: ")
            print(self.initial_solution)

        return self.initial_solution

    def run(self, write_cs_logfile=False, debug=False):
        """
        Runs seldonian algorithm using spec object

        :param write_cs_logfile: Whether to write candidate selection
                log file
        :param debug: Whether to print out debugging info
        :return: (passed_safety, solution). passed_safety
                indicates whether solution found during candidate selection
                passes the safety test. solution is the optimized
                model weights found during candidate selection or 'NSF'.
        :rtype: Tuple
        """
        self.set_initial_solution(
            verbose=debug
        )  # sets self.initial_solution so it can be used in candidate selection
        candidate_solution = self.run_candidate_selection(
            write_logfile=write_cs_logfile, debug=debug
        )

        if type(candidate_solution) == str and candidate_solution == "NSF":
            # can happen if nan or inf appeared in theta during optimization
            solution = "NSF"
            passed_safety = False
            return passed_safety, solution

        # Safety test
        batch_size_safety = self.spec.batch_size_safety
        passed_safety, solution = self.run_safety_test(
            candidate_solution=candidate_solution,
            batch_size_safety=batch_size_safety,
            debug=debug,
        )

        return passed_safety, solution

    def run_candidate_selection(self, write_logfile=False, debug=False):
        cs = self.candidate_selection(write_logfile=write_logfile)
        candidate_solution = cs.run(
            **self.spec.optimization_hyperparams,
            use_builtin_primary_gradient_fn=self.spec.use_builtin_primary_gradient_fn,
            custom_primary_gradient_fn=self.spec.custom_primary_gradient_fn,
            debug=debug,
        )

        self.cs_has_been_run = True
        self.cs_result = cs.optimization_result
        return candidate_solution

    def run_safety_test(self, candidate_solution, batch_size_safety=None, debug=False):
        """
        Runs safety test using solution from candidate selection
        or some other means

        :param candidate_solution: model weights from candidate selection
                or other process
        :param debug: Whether to print out debugging info
        :return: (passed_safety, solution). passed_safety
                indicates whether solution found during candidate selection
                passes the safety test. solution is the optimized
                model weights found during candidate selection or 'NSF'.
        :rtype: Tuple
        """

        st = self.safety_test()
        passed_safety = st.run(candidate_solution, batch_size_safety=batch_size_safety)
        if not passed_safety:
            if debug:
                print("Failed safety test")
            solution = "NSF"
        else:
            solution = candidate_solution
            if debug:
                print("Passed safety test!")
        self.st_has_been_run = True
        self.st_result = st.st_result
        return passed_safety, solution

    def get_cs_result(self):
        """Get the dictionary
        returned from running candidate selection
        """
        if not self.cs_has_been_run:
            raise ValueError(
                "Candidate selection has not "
                "been run yet, so result is not available. "
            )
        return self.cs_result

    def get_st_upper_bounds(self):
        """Get the upper bounds on each constraint
        evaluated on the safety data from the last
        time the safety test was run.

        return: upper_bounds_dict, a dictionary where the keys
                are the constraint strings and the values are the
                values of the upper bounds for that constraint
        """
        if not self.st_has_been_run:
            raise ValueError(
                "Safety test has not "
                "been run yet, so upper bounds are not available."
            )
        upper_bounds_dict = {}
        for pt in self.parse_trees:
            cstr = pt.constraint_str
            upper_bounds_dict[cstr] = self.st_result[cstr].root.upper

        return upper_bounds_dict

    def evaluate_primary_objective(self, branch, theta):
        """Get value of the primary objective given model weights,
        theta, on either the candidate selection dataset
        or the safety dataset. This is just a wrapper for
        primary_objective where data is fixed.

        :param branch: 'candidate_selection' or 'safety_test'
        :type branch: str
        :param theta: model weights
        :type theta: numpy.ndarray
        :return: the value of the primary objective
                evaluated for the given branch at the provided
                value of theta
        :rtype: float
        """
        if type(theta) == str and theta == "NSF":
            raise ValueError("Cannot evaluate primary objective because theta='NSF'")

        if branch == "safety_test":
            st = self.safety_test()
            result = st.evaluate_primary_objective(theta, self.spec.primary_objective)

        elif branch == "candidate_selection":
            cs = self.candidate_selection()
            cs.calculate_batches(
                batch_index=0, batch_size=self.candidate_dataset.num_datapoints
            )
            result = cs.evaluate_primary_objective(theta)
        return result
