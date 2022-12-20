""" Wrapper module for hyperparameter selection for Seldonian algorithms """

from sklearn.model_selection import train_test_split
import autograd.numpy as np

import warnings
from seldonian.dataset import (SupervisedDataSet, RLDataSet)
from seldonian.candidate_selection.candidate_selection import CandidateSelection
from seldonian.safety_test.safety_test import SafetyTest
from seldonian.models import objectives


class HyperparamSearch():
    def __init__(self, spec, all_frac_data_in_candidate_selection):
        """ Object for finding the best hyperparameters to use to optimize for probability
        of returning a safe solution for Seldonian algorithms. Note: currently only used to
        find optimal data split.

        List of hyperparameters to optimize:
        - Percentage of data in candidate and safety datasets

        :param spec: The specification object with the complete
                set of parameters for running the Seldonian algorithm
        :type spec: :py:class:`.Spec` object
        :param all_frac_data_in_candidate_selection: Array containing the values of fraction of data in 
                candidate solution that are being considered
        :type all_frac_data_in_candidate_selection: numpy.ndarray
        """
        self.spec = spec
        self.all_frac_data_in_candidate_selection = all_frac_data_in_candidate_selection 

        self.parse_trees = self.spec.parse_trees
        # user can pass a dictionary that specifies 
        # the bounding method for each base node
        # any base nodes not in this dictionary will
        # be bounded using the default method
        self.base_node_bound_method_dict = self.spec.base_node_bound_method_dict
        if self.base_node_bound_method_dict != {}:
                all_pt_constraint_strs = [pt.constraint_str for pt in self.parse_trees]
                for constraint_str in self.base_node_bound_method_dict:
                        this_bound_method_dict = self.base_node_bound_method_dict[constraint_str]
                        # figure out which parse tree this comes from
                        this_pt_index = all_pt_constraint_strs.index(constraint_str)
                        this_pt = self.parse_trees[this_pt_index]
                        # change the bound method for each node provided
                        for node_name in this_bound_method_dict:
                                this_pt.base_node_dict[node_name]['bound_method'] = this_bound_method_dict[node_name]

        self.dataset = self.spec.dataset
        self.regime = self.dataset.regime
        self.column_names = self.dataset.meta_information

        if self.spec.primary_objective is None:
                if self.regime == 'reinforcement_learning':
                        self.spec.primary_objective = objectives.IS_estimate
                elif self.regime == 'supervised_learning':
                        if self.spec.sub_regime == 'classification':
                                self.spec.primary_objective = objectives.logistic_loss
                        elif self.spec.sub_regime == 'regression':
                                self.spec.primary_objective = objectives.Mean_Squared_Error


    def get_safety_size(self, n_total, frac_data_in_safety):
        """ Determine the number of data points in the safety dataset.

        :param n_total: the total amount of data
        :type n_total: int
        :param frac_data_in_safety: fraction of data used in safety test, 
                the remaining fraction will be used in candidate selection
        :type frac_data_in_safety: float
        """
        n_safety = int(frac_data_in_safety * n_total)
        n_safety = max(n_safety, 2) # >=2 point in safety
        n_safety = min(n_safety, n_total-2) # >=1 point in selection

        return n_safety


    def create_dataset(self, dataset, frac_data_in_safety, shuffle=False):
        """ Partition data to create candidate and safety dataset according to frac_data_in_safety

        :param dataset: a dataset object containing data
        :type dataset: :py:class:`.DataSet` object
        :param frac_data_in_safety: fraction of data used in safety test, 
                the remaining fraction will be used in candidate selection
        :type frac_data_in_safety: float
        """
        if self.regime == 'supervised_learning':
                n_total = len(dataset.df)
                if n_total < 4:
                        warning_msg = (
                                "Warning: not enough data to "
                                "run the Seldonian algorithm.")
                        warnings.warn(warning_msg)

                # Make sure at least two datapoints in safety and candidate.
                n_safety_split = self.get_safety_size(n_total, frac_data_in_safety)

                # Split the data.
                candidate_df, safety_df = train_test_split(
                        dataset.df, test_size=n_safety_split,
                        shuffle=shuffle)

                # Create candidate and safety datasets
                candidate_dataset = SupervisedDataSet(
                        candidate_df,meta_information=self.column_names,
                        sensitive_column_names=dataset.sensitive_column_names,
                        include_sensitive_columns=dataset.include_sensitive_columns,
                        include_intercept_term=dataset.include_intercept_term,
                        label_column=dataset.label_column)

                safety_dataset = SupervisedDataSet(
                        safety_df,meta_information=self.column_names,
                        sensitive_column_names=dataset.sensitive_column_names,
                        include_sensitive_columns=dataset.include_sensitive_columns,
                        include_intercept_term=dataset.include_intercept_term,
                        label_column=dataset.label_column)
                
                n_candidate = len(candidate_df)
                n_safety = len(safety_df)
                if n_candidate < 2 or n_safety < 2:
                        warning_msg = (
                                "Warning: not enough data to "
                                "run the Seldonian algorithm.")
                        warnings.warn(warning_msg)
                        print(n_candidate)
                        print(n_safety)


        elif self.regime == 'reinforcement_learning':
                self.env_kwargs = self.spec.model.env_kwargs

                episodes = dataset.episodes
                # Create candidate and safety datasets
                n_episodes = len(episodes)

                # Make sure at least two datapoints in safety and candidate.
                n_safety = self.get_safety_size(n_episodes, n_episodes)
                n_candidate = n_episodes - n_safety

                candidate_episodes = episodes[0:n_candidate]
                safety_episodes = episodes[n_candidate:]

                candidate_dataset = RLDataSet(
                        episodes=candidate_episodes,
                        meta_information=self.column_names)

                safety_dataset = RLDataSet(
                        episodes=safety_episodes,
                        meta_information=self.column_names)

                print(f"Safety dataset has {n_safety} episodes")
                print(f"Candidate dataset has {n_candidate} episodes")

        return candidate_dataset, safety_dataset, n_candidate, n_safety


    def bootstrap_sample_dataset(self, dataset, n_bootstrap):
        """ Bootstrap sample a dataset of size n_bootstrap from data points in dataset.

        :param dataset: a dataset object containing data
        :type dataset: :py:class:`.DataSet` object
        :param n_bootstrap: the desired number of points in the bootstrapped dataset.
        :type n_bootstrap: int
        """

        if self.regime == 'supervised_learning':
            bs_df = dataset.df.sample(n=n_bootstrap, replace=True)
            bs_dataset = SupervisedDataSet(
                    bs_df,meta_information=self.column_names,
                    sensitive_column_names=dataset.sensitive_column_names,
                    include_sensitive_columns=dataset.include_sensitive_columns,
                    include_intercept_term=dataset.include_intercept_term,
                    label_column=dataset.label_column)

        elif self.regime == 'reinforcement_learning':
            bs_episodes = np.random.choice(dataset.episodes, n_bootstrap)
            bs_dataset = RLDataSet(
                    episodes=bs_episodes,
                    meta_information=self.column_names)

        return bs_dataset
                

    def get_initial_solution(self, candidate_dataset, frac_data_in_safety):
        """ Get the initial solution used in candidate selection.

        :param candidate_dataset: a dataset object containing data that should be used to   
                to find the initial soluion.
        :type candidate_dataset: :py:class:`.DataSet` object
        :param frac_data_in_safety: fraction of data used in safety test.
                The remaining fraction will be used in candidate selection
        :type frac_data_in_safety: float
        """
        if self.regime == 'supervised_learning':
                candidate_df = candidate_dataset.df

                # Set up initial solution
                initial_solution_fn = self.spec.initial_solution_fn

                candidate_labels = candidate_df[candidate_dataset.label_column]
                candidate_features = candidate_df.loc[:,
                        candidate_df.columns != candidate_dataset.label_column]

                if not candidate_dataset.include_sensitive_columns:
                        candidate_features = candidate_features.drop(
                                columns=candidate_dataset.sensitive_column_names)
        
                if candidate_dataset.include_intercept_term:
                        candidate_features.insert(0,'offset',1.0) # inserts a column of 1's

                if initial_solution_fn is None:
                        initial_solution = np.zeros(candidate_features.shape[1])
                else:
                        try: 
                                initial_solution = initial_solution_fn(
                                        candidate_features,candidate_labels)
                        except Exception as e: 
                                # handle off-by-one error due to intercept not being included
                                warning_msg = (
                                        "Warning: initial solution function failed with this error:"
                                        f" {e}")
                                warnings.warn(warning_msg)
                                initial_solution = np.random.normal(
                                        loc=0.0,scale=1.0,size=(candidate_features.shape[1]+1)
                                        )

        elif self.regime == 'reinforcement_learning':
                self.env_kwargs = self.spec.model.env_kwargs

                initial_solution_fn = self.spec.initial_solution_fn

                if initial_solution_fn is None:
                        initial_solution = self.spec.model.policy.get_params()
                else:
                        initial_solution = initial_solution_fn(candidate_dataset)


        return initial_solution


    def candidate_selection(self, candidate_dataset, n_safety, initial_solution,
            write_logfile=False):
        """ Create the candidate selection object

        :param candidate_dataset: a dataset object containing candidate selection dataset
        :type candidate_dataset: :py:class:`.DataSet` object
        :param n_safety: size of safety dataset
        :type n_safety: int
        :param initial_solution: initial solution that should be used in candidate selection
        :type initial_solution: float
        """

        cs_kwargs = dict(
                model=self.spec.model,
                candidate_dataset=candidate_dataset,
                n_safety=n_safety,
                parse_trees=self.parse_trees,
                primary_objective=self.spec.primary_objective,
                optimization_technique=self.spec.optimization_technique,
                optimizer=self.spec.optimizer,
                initial_solution=initial_solution,
                regime=self.regime,
                write_logfile=write_logfile)

        cs = CandidateSelection(**cs_kwargs,**self.spec.regularization_hyperparams)

        return cs


    def safety_test(self, safety_dataset):
        """ Create the safety test object

        :param safety_dataset: a dataset object containing safety dataset
        :type safety_dataset: :py:class:`.DataSet` object
        """
        st_kwargs = dict(
                safety_dataset=safety_dataset,
                model=self.spec.model,parse_trees=self.spec.parse_trees,
                regime=self.regime,
                )	
        
        st = SafetyTest(**st_kwargs)
        return st


    def run_safety_test(self, candidate_solution, safety_dataset, debug=False):
        """
        Runs safety test using solution from candidate selection
        or some other means

        :param candidate_solution: model weights from candidate selection
                or other process
        :param safety_dataset: a dataset object containing safety dataset
        :type safety_dataset: :py:class:`.DataSet` object
        :param debug: Whether to print out debugging info
        :return: (passed_safety, solution). passed_safety 
                indicates whether solution found during candidate selection
                passes the safety test. solution is the optimized
                model weights found during candidate selection or 'NSF'.
        :rtype: Tuple 
        """
                
        st = self.safety_test(safety_dataset)
        passed_safety = st.run(candidate_solution)
        if not passed_safety:
                if debug:
                        print("Failed safety test")
                solution = "NSF"
        else:
                solution = candidate_solution
                if debug:
                        print("Passed safety test!")
        return passed_safety, solution



    def run_core(self, candidate_dataset, safety_dataset, n_safety, frac_data_in_safety, 
            write_cs_logfile=False,debug=False):
        """
        Runs seldonian algorithm core using spec object

        :param candidate_dataset: a dataset object containing candidate solution dataset
        :type candidate_dataset: :py:class:`.DataSet` object
        :param safety_dataset: a dataset object containing safety dataset
        :type safety_dataset: :py:class:`.DataSet` object
        :param n_safety: size of safety dataset
        :type n_safety: int
        :param frac_data_in_safety: fraction of data used in safety test, 
                the remaining fraction will be used in candidate selection
        :type frac_data_in_safety: float
        :param write_cs_logfile: Whether to write candidate selection
                log file
        :param debug: Whether to print out debugging info
        :return: (passed_safety, solution). passed_safety 
                indicates whether solution found during candidate selection
                passes the safety test. solution is the optimized
                model weights found during candidate selection or 'NSF'.
        :rtype: Tuple 
        """

        # Find candidate solution.
        initial_solution = self.get_initial_solution(candidate_dataset, frac_data_in_safety)
        cs = self.candidate_selection(candidate_dataset, n_safety, initial_solution,
                write_logfile=write_cs_logfile)
        candidate_solution = cs.run(**self.spec.optimization_hyperparams,
                use_builtin_primary_gradient_fn=self.spec.use_builtin_primary_gradient_fn,
                custom_primary_gradient_fn=self.spec.custom_primary_gradient_fn,
                debug=debug)

        if type(candidate_solution) == str and candidate_solution == 'NSF':
                # can happen if nan or inf appeared in theta during optimization
                solution = 'NSF'
                passed_safety = False
                return passed_safety,solution
                
        # Safety test
        passed_safety, solution = self.run_safety_test(candidate_solution, safety_dataset,
                debug=debug)

        return passed_safety, solution


    def find_best_hyperparams(self, write_cs_logfile=False, debug=False):
        """ Find the best hyperparameter values to use for the Seldonian algorithm.
        Note: currently only implemented for frac_data_in_safety.

        :return: (frac_data_in_safety, candidate_dataset, safety_dataset). frac_data_in_safety
                indicates the percentage of total data that is included in the safety dataset.
                candidate_dataset and safety_dataset are dataset objects containing data from
                self.dataset split according to frac_data_in_safety
        :rtyle: Tuple
        """

        # Sort from lowest amount of data in candidate selection from lowest to highest.
        self.all_frac_data_in_candidate_selection.sort() 

        for rho in self.all_frac_data_in_candidate_selection: # Move data from safety to cs dataset.
            frac_data_in_safety = 1 - rho

            # Partition data according to rho.
            candidate_dataset, safety_dataset, n_candidate, n_safety = self.create_dataset(
                    self.dataset, frac_data_in_safety, shuffle=False)

            # Estimate probability of passing.
            rho_prob_pass = self.est_prob_pass(rho, candidate_dataset, safety_dataset, 
                    n_candidate, n_safety, write_cs_logfile, debug) # 

            # Estimate if any of the future splits of data lead to higher P(pass)
            rho_prime_better = False
            for rho_prime in self.all_frac_data_in_candidate_selection: 
                if rho_prime <= rho: # Only look at larger rho.
                    continue
                    
                rho_prime_prob_pass = self.est_prob_pass(rho_prime, candidate_dataset, 
                        safety_dataset, n_candidate, n_safety, write_cs_logfile, debug)
                if rho_prime_prob_pass > rho_prob_pass: # Predict a future split is better.
                    rho_prime_better = True
                    break 

            # If do not predict any greater rho is better, return rho and datasplit.
            if rho_prime_better is False: 
                break

        frac_data_in_safety = 1 - rho

        return (frac_data_in_safety, candidate_dataset, safety_dataset)


    def est_prob_pass(self, rho_prime, candidate_dataset, safety_dataset, n_candidate, 
            n_safety, write_cs_logfile=False, debug=False, bootstrap_iter=100):
        """ Estimates probability of returning a solution with rho_prime fraction of data
            in candidate selection.

        :param rho_prime: fraction of data in candidate selection that we want to estimate
                        the probabiilty of returning a solution for
        :type rho_prime: float
        :param candidate_dataset: a dataset object containing candidate solution dataset
        :type candidate_dataset: :py:class:`.DataSet` object
        :param safety_dataset: a dataset object containing safety dataset
        :type safety_dataset: :py:class:`.DataSet` object
        :param n_safety: size of candidate dataset
        :type n_safety: int
        :param n_safety: size of safety dataset
        :type n_safety: int
        """
        frac_data_in_safety_prime = 1 - rho_prime

        # Size of bootstrapped datasets according to rho'.
        total_data = len(self.dataset.df) 
        bs_n_candidate = int(total_data * rho_prime)
        bs_n_safety = total_data - bs_n_candidate

        pass_count = 0
        for i in range(bootstrap_iter):

            # Split candidate_dataset into pools for bootstraping samplinhg.
            candidate_pool, safety_pool, pool_n_candidate, pool_n_safety = self.create_dataset(
                    candidate_dataset, frac_data_in_safety_prime, shuffle=True)

            # Bootstrap sample datasets approximating candidate and safety datasets.
            bs_candidate_dataset = self.bootstrap_sample_dataset(candidate_pool, bs_n_candidate)
            bs_safety_dataset = self.bootstrap_sample_dataset(safety_pool, bs_n_safety)

            # Compute a candidate solution using bootstrapped candidate dataset.
            passed_safety, _ = self.run_core(bs_candidate_dataset, bs_safety_dataset, bs_n_safety, 
                    frac_data_in_safety_prime, write_cs_logfile, debug=False)

            if passed_safety:
                pass_count +=1

        bs_prob_pass = pass_count / bootstrap_iter

        return bs_prob_pass
