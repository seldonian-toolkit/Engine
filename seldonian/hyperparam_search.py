""" Wrapper module for hyperparameter selection for Seldonian algorithms """

import autograd.numpy as np
import pandas as pd
import cma
from scipy.optimize import minimize

import os
import glob
import copy
import time
import scipy
import pickle
import warnings
import itertools
import multiprocessing as mp
from collections import OrderedDict
from tqdm import tqdm
from functools import partial
from concurrent.futures import ProcessPoolExecutor

from seldonian.seldonian_algorithm import SeldonianAlgorithm
from seldonian.dataset import SupervisedDataSet, RLDataSet
from seldonian.candidate_selection.candidate_selection import CandidateSelection
from seldonian.safety_test.safety_test import SafetyTest
from seldonian.models import objectives
from seldonian.utils.io_utils import load_pickle,save_pickle,cmaes_logger
from seldonian.utils.stats_utils import tinv
import seldonian.utils.hyperparam_utils as hp_utils 


class HyperSchema(object):
    def __init__(self,hyper_dict):
        """ Container for all hyperparameters one wants to tune
        and their possible values.
        
        :param hyper_dict: Hyperparameter dictionary adhering to the following format:
            keys: names of hyperparameters
            values: dictionary whose format depends on how you want this hyperparameter tuned. 
            If you want to do a grid search over this hyperparameter, the required keys are: 
                ["values","hyper_type","tuning_method"], where
                "values" is the list of values to search,
                "hyper_type" is one of ["optimization","model","SA"], specifying 
                the type of hyperparameter, and 
                "tuning_method" is "grid_search"
            If you want to do CMA-ES over this hyperparameter, the required keys are: 
                ["initial_value","min_val","max_val","hyper_type","search distribution"], where
                "initial_value" is the starting value for this hyperparameter in CMA-ES,
                "min_val" is the minimum value you want to search for this hyperparameter,
                "max_val" is the maximum value you want to search for this hyperparameter,
                "hyper_type" is one of ["optimization","model","SA","tuning_method"], specifying 
                the type of hyperparameter, 
                "search_distribution" is either "uniform" or "log-uniform". "uniform" searches 
                over a uniform distribution between "min_val" and "max_val", where "log-uniform" 
                searchs over a log-uniform distribution betwen "min_val" and "max_val". This is common
                for step size hyperparameters, for example. 
                ""tuning_method" is "CMA-ES".

            
        Here is an example for tuning the number of iterations "num_iters" using grid search 
        and the step size "alpha_theta" using CMA-ES:

        hyper_dict = {
            "num_iters": {
                "values":[100,500,1000,1500],
                "hyper_type":"optimization",
                "tuning_method": "grid_search"
            },
            "alpha_theta": {
                "initial_value":0.005,
                "min_val": 0.0001,
                "max_val": 0.1,
                "hyper_type":"optimization",
                "search_distribution": "log-uniform"
                "tuning_method": "CMA-ES"
            }
        }

        """
        self.allowed_optimization_hyperparams = [
            "alpha_theta",
            "alpha_lamb",
            "beta_velocity",
            "beta_rmsprop",
            "batch_size",
            "n_epochs",
            "num_iters"
        ]
        self.allowed_SA_hyperparams = [
            "bound_inflation_factor",
            "frac_data_in_safety",
            "delta_split_vector"
        ]
        self.allowable_tuning_methods = ["grid_search","CMA-ES"]
        self.allowable_hyper_types = ["optimization","model","SA"]
        self.hyper_dict = OrderedDict(self._validate(hyper_dict)) 
        self.hyper_param_names = list(self.hyper_dict.keys())
        
    def _validate(self,hyper_dict):
        """ Check that the hyperparameter dictionary is formatted properly
        and contains valid hyperparameters. Model hyperparameters are specific 
        to the model so we can't know what they might be ahead of time. Errors 
        regarding model hyperparameters will be caught elsewhere.
        """
        for hyper_name,hyper_info in hyper_dict.items():
            if not isinstance(hyper_info,dict):
                raise RuntimeError(f"hyper_dict['{hyper_name}'] is not a dictionary. ")
            
            # Check for tuning method
            if "tuning_method" not in hyper_info:
                raise KeyError(f"hyper_dict['{hyper_name}'] must have the key: 'tuning_method'. ")

            if hyper_info["tuning_method"] not in self.allowable_tuning_methods:
                raise ValueError(f"hyper_dict['{hyper_name}']['tuning_method'] must be one of: {self.allowable_tuning_methods}. ")

            if hyper_info["tuning_method"] == "grid_search":
                required_keys = ["values","hyper_type","tuning_method"]
            elif hyper_info["tuning_method"] == "CMA-ES":
                required_keys = ["initial_value","min_val","max_val","hyper_type",
                "search_distribution","tuning_method"]
            
            for w in required_keys:
                if w not in hyper_info:
                    tuning_method = hyper_info['tuning_method']
                    raise KeyError(f"hyper_dict['{hyper_name}'] must have the key: '{w}' when tuning method is: {tuning_method}.")
            
            if hyper_info["tuning_method"] == "grid_search":
                if len(hyper_info["values"]) < 1:
                    raise ValueError(f"hyper_dict['{hyper_name}']['values'] must have at least one value")
            elif hyper_info["tuning_method"] == "CMA-ES":
                if not isinstance(hyper_info["initial_value"],(int, float)):
                    raise ValueError(f"hyper_dict['{hyper_name}']['initial_value'] must be a real number")

            if hyper_info["hyper_type"] not in self.allowable_hyper_types:
                raise ValueError(f"hyper_dict['{hyper_name}']['hyper_type'] must be one of {self.allowable_hyper_types}")
            
            if hyper_info["hyper_type"] == "optimization" and hyper_name not in self.allowed_optimization_hyperparams:
                raise ValueError(f"{hyper_name} is not an allowed optimization hyperparameter")
            
            if hyper_info["hyper_type"] == "SA" and hyper_name not in self.allowed_SA_hyperparams:
                raise ValueError(f"{hyper_name} is not an allowed hyperparameter of the Seldonian algorithm")
        return hyper_dict   


class HyperparamSearch:
    def __init__(
            self, 
            spec, 
            hyperparam_spec,
            results_dir,
            write_logfile=False,
    ):
        """Object for finding the best hyperparameters to use to optimize for probability
        of returning a safe solution for Seldonian algorithms. 
        
        Note: currently only implemented for finding optimal train/safety data split.

        List of hyperparameters to optimize:
        - Percentage of data in candidate and safety datasets

        :param spec: The specification object with the complete
                set of parameters for running the Seldonian algorithm
        :type spec: :py:class:`.Spec` object
        :param hyperparam_spec: The specification object with the complete
                set of parameters for doing hyparpameter selection
        :type hyperparam_spec: :py:class:`.HyperparameterSelectionSpec` object
        :param results_dir: The directory where results will be saved
        :type results_dir: str
        :param write_logfile: Whether to write out logs from hyperparameter optimization
        :type write_logfile: Bool
        """
        self.spec = spec
        self.hyperparam_spec = hyperparam_spec 
        self.hyper_dict = self.hyperparam_spec.hyper_schema.hyper_dict
        self.hyper_param_names = self.hyperparam_spec.hyper_schema.hyper_param_names
        self.results_dir = results_dir
        self.write_logfile = write_logfile

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
        self.meta = self.dataset.meta

        self.model = self.spec.model
        if self.regime == "supervised_learning": self.sub_regime = self.spec.sub_regime

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


    def get_safety_size(
            self,
            n_total,
            frac_data_in_safety
    ):
        """Determine the number of data points in the safety dataset.

        :param n_total: the size of the total dataset
        :type n_total: int
        :param frac_data_in_safety: fraction of data used in safety test,
                the remaining fraction will be used in candidate selection
        :type frac_data_in_safety: float

        :return: n_safety, the desired size of the safety dataset
        :rtype: int
        """
        n_safety = int(frac_data_in_safety * n_total)

        # If have at least 4 datapoints, make sure that each set gets at least 2.
        if n_total >= 4:
            n_safety = max(n_safety, 2)  # >=2 point in safety
            n_safety = min(n_safety, n_total - 2)  # >=2 point in selection

        return n_safety


    def candidate_safety_combine(
            self,
            candidate_dataset,
            safety_dataset
    ):
        """Combine candidate_dataset and safety_dataset into a full dataset.
        The data will be joined so that the candidate data comes before the 
        safety data.

        :param candidate_dataset: a dataset object containing data
        :type candiddate_dataset: :py:class:`.DataSet` object
        :param safety_dataset: a dataset object containing data
        :type safety_dataset: :py:class:`.DataSet` object

        :return: combinded_dataset, a dataset containing candidate and safety dataset
        :rtype: :py:class:`.DataSet` object
        """
        if self.regime == "supervised_learning":
            combined_num_datapoints = (candidate_dataset.num_datapoints +
                    safety_dataset.num_datapoints)
            # Combine features.
            if ((type(candidate_dataset.features) == list) and 
                    (type(safety_dataset.features) == list)):
                combined_features = [f_c + f_s for (f_c, f_s) in 
                        zip(candidate_dataset.features, safety_dataset.features)]
            else:
                combined_features = np.concatenate((candidate_dataset.features, safety_dataset.features),
                        axis=0)
            # Compute labels - must be a numpy array.
            combined_labels = np.concatenate((candidate_dataset.labels, safety_dataset.labels), axis=0)

            # Combine sensitive attributes. Must be a numpy array.
            combined_sensitive_attrs = np.concatenate((candidate_dataset.sensitive_attrs, 
                safety_dataset.sensitive_attrs), axis=0)

            # Create a dataset.
            combined_dataset = SupervisedDataSet(
                    features=combined_features,
                    labels=combined_labels,
                    sensitive_attrs=combined_sensitive_attrs,
                    num_datapoints=combined_num_datapoints,
                    meta=candidate_dataset.meta)


        elif self.regime == "reinforcement_learning":
            # TODO: Finish implementing this.
            raise NotImplementedError(
                    "Creating bootstrap sampled datasets not yet implemented for "
                    "reinforcement_learning regime")

        return combined_dataset


    def candidate_safety_split(
            self,
            dataset,
            frac_data_in_safety
    ):
        """Split features, labels and sensitive attributes
        into candidate and safety sets according to frac_data_in_safety

        :param dataset: a dataset object containing data
        :type dataset: :py:class:`.DataSet` object
        :param frac_data_in_safety: Fraction of data used in safety test.
                The remaining fraction will be used in candidate selection
        :type frac_data_in_safety: float

        :return: F_c,F_s,L_c,L_s,S_c,S_s
                where F=features, L=labels, S=sensitive attributes
        :rtype: Tuple
        """
        n_points_tot = dataset.num_datapoints
        n_safety = self.get_safety_size(n_points_tot, frac_data_in_safety)
        n_candidate = n_points_tot - n_safety

        if self.regime == "supervised_learning":
            # Split features
            if type(dataset.features) == list:
                F_c = [x[:n_candidate] for x in dataset.features]
                F_s = [x[n_candidate:] for x in dataset.features]
            else:
                F_c = dataset.features[:n_candidate]
                F_s = dataset.features[n_candidate:]
            # Split labels - must be numpy array
            L_c = dataset.labels[:n_candidate]
            L_s = dataset.labels[n_candidate:]

            # Split sensitive attributes - must be numpy array
            S_c = dataset.sensitive_attrs[:n_candidate]
            S_s = dataset.sensitive_attrs[n_candidate:]
            return F_c, F_s, L_c, L_s, S_c, S_s, n_candidate, n_safety

        elif self.regime == "reinforcement_learning":
            # Split episodes
            E_c = dataset.episodes[0:n_candidate]
            E_s = dataset.episodes[n_candidate:]

            # Split sensitive attributes - must be numpy array
            S_c = dataset.sensitive_attrs[:n_candidate]
            S_s = dataset.sensitive_attrs[n_candidate:]
            return E_c, E_s, S_c, S_s, n_candidate, n_safety

    def create_dataset(
            self,
            dataset,
            frac_data_in_safety,
            shuffle=False
    ):
        """Partition data to create candidate and safety dataset according to
            frac_data_in_safety. 

        :param dataset: a dataset object containing data
        :type dataset: :py:class:`.DataSet` object
        :param frac_data_in_safety: fraction of data used in safety test,
                the remaining fraction will be used in candidate selection
        :type frac_data_in_safety: float
        :param shuffle: bool indicating if we should shuffle the dataset before 
                splitting it into candidate and safety datasets
        :type shuffle: bool 

        :return: (candidate_dataset, safety_dataset). candidate_dataset
                and safety_datasets are the resulting datasets after partitioning
                the dataset.
        :rtype: Tuple containing two `.DataSet` objects.
        """
        if shuffle:
            dataset = hp_utils.create_shuffled_dataset(dataset)

        if self.regime == "supervised_learning":

            if dataset.num_datapoints < 4:
                warning_msg = (
                    "Warning: not enough data to " "run the Seldonian algorithm."
                )
                warnings.warn(warning_msg)

            # Split the data.
            (   candidate_features,
                safety_features,
                candidate_labels,
                safety_labels,
                candidate_sensitive_attrs,
                safety_sensitive_attrs,
                n_candidate,
                n_safety
            ) = self.candidate_safety_split(dataset, frac_data_in_safety)

            candidate_dataset = SupervisedDataSet(
                features=candidate_features,
                labels=candidate_labels,
                sensitive_attrs=candidate_sensitive_attrs,
                num_datapoints=n_candidate,
                meta=dataset.meta,
            )

            safety_dataset = SupervisedDataSet(
                features=safety_features,
                labels=safety_labels,
                sensitive_attrs=safety_sensitive_attrs,
                num_datapoints=n_safety,
                meta=dataset.meta,
            )

            if candidate_dataset.num_datapoints < 2 or safety_dataset.num_datapoints < 2:
                warning_msg = (
                    "Warning: not enough data to " "run the Seldonian algorithm."
                )
                warnings.warn(warning_msg)
            if self.spec.verbose:
                print(f"Safety dataset has {safety_dataset.num_datapoints} datapoints")
                print(f"Candidate dataset has {candidate_dataset.num_datapoints} datapoints")

        elif self.regime == "reinforcement_learning":

            # Split the data.
            (   candidate_episodes,
                safety_episodes,
                candidate_sensitive_attrs,
                safety_sensitive_attrs,
                n_candidate,
                n_safety,
            ) = self.candidate_safety_split(dataset, frac_data_in_safety)

            candidate_dataset = RLDataSet(
                episodes=candidate_episodes,
                sensitive_attrs=candidate_sensitive_attrs,
                meta=self.meta,
            )

            safety_dataset = RLDataSet(
                episodes=safety_episodes,
                sensitive_attrs=safety_sensitive_attrs,
                meta=self.meta,
            )

            print(f"Safety dataset has {safety_dataset.num_datapoints} episodes")
            print(f"Candidate dataset has {candidate_dataset.num_datapoints} episodes")

        return candidate_dataset, safety_dataset


    def generate_all_bootstrap_datasets(
            self,
            candidate_dataset,
            frac_data_in_safety,
            n_bootstrap_samples_candidate,
            n_bootstrap_samples_safety,
            bootstrap_savedir,
    ):
        """Utility function for supervised learning to generate the
        resampled datasets to use in each bootstrap trial. Resamples (with replacement)
        features, labels and sensitive attributes to create 
        self.hyperparam_spec.n_bootstrap_trials versions of these 

        :param candidate_dataset: Dataset object containing candidate solution dataset.
                This is the dataset we will be bootstrap sampling from.
        :type candidate_dataset: :py:class:`.DataSet` object
        :param frac_data_in_safety: fraction of data in safety set that we want to 
                        estimate the probabiilty of returning a solution for
        :type frac_data_in_safety: float
        :param n_bootstrap_samples_candidate: The size of the candidate selection 
                bootstrapped dataset
        :type n_bootstrap_samples_candidate: int
        :param n_bootstrap_samples_safety: The size of the safety bootstrapped dataset
        :type n_bootstrap_safety: int
        :param bootstrap_savedir: The root diretory to save all the bootstrapped datasets.
        :type bootstrap_savedir: str
        """

        # If not enough datapoints
        if candidate_dataset.num_datapoints < 4:
            return created_trials 

        dataset_save_subdir = os.path.join(bootstrap_savedir, 
                f"frac_data_in_safety_{frac_data_in_safety:.2f}")
        
        os.makedirs(dataset_save_subdir, exist_ok=True) 

        for bootstrap_trial_i in range(self.hyperparam_spec.n_bootstrap_trials):
            # Where to save bootstrapped dataset.
            bootstrap_datasets_savename = os.path.join(dataset_save_subdir, 
                    f"bootstrap_datasets_trial_{bootstrap_trial_i}.pkl")

            # Only create datasets if dataset not already existing.
            if os.path.exists(bootstrap_datasets_savename):
                continue

            bootstrap_datasets_dict = dict() # Will store all the datasets.

            # Bootstrap sample candidate selection and safety datasets.
            if self.hyperparam_spec.use_bs_pools:
                # Partition candidate_dataset into pools to bootstrap 
                (bootstrap_pool_candidate, bootstrap_pool_safety) = self.create_dataset(
                        candidate_dataset, frac_data_in_safety, shuffle=True)

                # Sample from pools.
                bootstrap_datasets_dict["candidate"] = hp_utils.bootstrap_sample_dataset(
                        bootstrap_pool_candidate, n_bootstrap_samples_candidate,self.regime)
                bootstrap_datasets_dict["safety"] = hp_utils.bootstrap_sample_dataset(
                        bootstrap_pool_safety, n_bootstrap_samples_safety,self.regime)

            else: # Sample directly from candidate datset.
                bootstrap_datasets_dict["candidate"] = hp_utils.bootstrap_sample_dataset(
                        candidate_dataset, n_bootstrap_samples_candidate,self.regime)
                bootstrap_datasets_dict["safety"] = hp_utils.bootstrap_sample_dataset(
                        candidate_dataset, n_bootstrap_samples_safety,self.regime)

            # Save datasets.
            save_pickle(bootstrap_datasets_savename, bootstrap_datasets_dict, verbose=self.spec.verbose)

        return 


    def create_bootstrap_trial_spec(
            self,
            bootstrap_trial_i,
            frac_data_in_safety, 
            bootstrap_savedir,
            hyperparam_setting=None,
    ):
        """Create the spec to run this iteration of the bootstrap trial.

        :param bootstrap_trial_i: Indicates which trial we are currently running
        :type bootstrap_trial_i: int
        :param frac_data_in_safety: fraction of data used in safety test to split the 
            datasets for the trial.
        :type frac_data_in_safety: float
        :param bootstrap_savedir: The root diretory to save all the bootstrapped datasets.
        :type bootstrap_savedir: str
        """
        spec_for_bootstrap_trial = copy.deepcopy(self.spec)

        # Load datasets associated with the trial.
        bootstrap_datasets_savename = os.path.join(
            self.results_dir,
            "bootstrapped_datasets",
            f"frac_data_in_safety_{frac_data_in_safety:.2f}", 
            f"bootstrap_datasets_trial_{bootstrap_trial_i}.pkl"
        )

        bootstrap_datasets_dict = load_pickle(bootstrap_datasets_savename)

        bs_candidate_dataset = bootstrap_datasets_dict["candidate"]
        bs_safety_dataset = bootstrap_datasets_dict["safety"]

        # Combine loaded candidate and safety dataset to create the full dataset.
        combined_dataset = self.candidate_safety_combine(bs_candidate_dataset, bs_safety_dataset)

        # Set the datasets associated with the trial.
        spec_for_bootstrap_trial.dataset = combined_dataset 
        spec_for_bootstrap_trial.candidate_dataset = bs_candidate_dataset
        spec_for_bootstrap_trial.safety_dataset = bs_safety_dataset
        spec_for_bootstrap_trial.frac_data_in_safety = frac_data_in_safety

        # Update spec with hyperparam_setting.
        spec_for_bootstrap_trial = hp_utils.set_spec_with_hyperparam_setting(
                spec_for_bootstrap_trial, hyperparam_setting)

        return spec_for_bootstrap_trial


    def run_bootstrap_trial(
            self,
            bootstrap_trial_i,
            frac_data_in_safety,
            parent_savedir,
            hyperparam_setting=None,
    ):
        """Run bootstrap train bootstrap_trial_i to estimate the probability of passing
        with frac_data_in_safety.

        Returns a boolean indicating if the bootstrap trial was actually run. If the 
            bootstrap has been already run, will return False.

        :param bootstrap_trial_i: integer indicating which trial of the bootstrap 
            experiment we are currently running. Allows us to identify which bootstrapped
            dataset to load adn run
        :type bootstrap_trial_i: int
        :param frac_data_in_safety: fraction of data in safety set that we want to
            estimate the probabiilty of returning a solution for
        :type frac_data_in_safety: float
        :param bootstrap_savedir: The root diretory to load bootstrapped dataset and save 
            the result of this bootstrap trial
        :type bootstrap_savedir: str
        """
        # TODO: Update this with the other kwargs, should be a spec.

        bs_result_savename = os.path.join(parent_savedir, 
                f"trial_{bootstrap_trial_i}_result.pkl")

        # If this bootstrap trial has already been run, skip.
        if os.path.exists(bs_result_savename):
            if self.spec.verbose:
                print(f"Bootstrap trial {bootstrap_trial_i} has already been run. Skipping.")
            return False

        # Create spec for the bootstrap trial. The bootstrapped candidate and safety
        # datasets are created here.
        spec_for_bootstrap_trial = self.create_bootstrap_trial_spec(bootstrap_trial_i,
                frac_data_in_safety, parent_savedir, hyperparam_setting)
        # Run Seldonian Algorithm on the bootstrapped data. Load the datasets here.
        SA = SeldonianAlgorithm(spec_for_bootstrap_trial)
        # try:
        passed_safety, solution = SA.run(write_cs_logfile=self.spec.verbose, 
                debug=self.spec.verbose)
        # except (ValueError, ZeroDivisionError): # Now, all experiemnts should return.
        #     passed_safety = False
        #     solution = "NSF"

        # Save the results.
        # os.makedirs(bs_result_subdir, exist_ok=True) 
        trial_result_dict = {
                "bootstrap_trial_i" : bootstrap_trial_i,
                "passed_safety" : passed_safety,
                "solution" : solution
        }
        save_pickle(bs_result_savename, trial_result_dict, verbose=self.spec.verbose)

        return True


    def aggregate_est_prob_pass(
            self,
            est_frac_data_in_safety,
            bootstrap_savedir
    ):
        """Compute the estimated probability of passing using the result files in 
        bootstrap_savedir.

        :param est_frac_data_in_safety: fraction of data in safety set that we want to 
                        estimate the probabiilty of returning a solution for
        :type est_frac_data_in_safety: float
        :param bootstrap_savedir: root diretory to load results from bootstrap trial, and
            write aggregated result
        :type bootstrap_savedir: str
        """
        # TODO: Update this to allow aggregating first n, for given n.
        # bs_frac_subdir = os.path.join(bootstrap_savedir,
        #         f"frac_data_in_safety_{est_frac_data_in_safety:.2f}")
        # bs_result_subdir = os.path.join(bs_frac_subdir, "bootstrap_results")

        # Load the result for each trial.
        bs_trials_index = []
        bs_trials_pass = []
        bs_trials_solution = []
        result_trial_filenames = glob.glob(bootstrap_savedir + "/trial_*_result.pkl")
        assert len(result_trial_filenames) > 0
        for result_trial_savename in result_trial_filenames:
            result_trial_dict = load_pickle(result_trial_savename)
            bs_trials_index.append(result_trial_dict["bootstrap_trial_i"])
            bs_trials_pass.append(result_trial_dict["passed_safety"])
            bs_trials_solution.append(result_trial_dict["solution"])

        # Create dataframe containing data.
        results_df = pd.DataFrame(data = {
            "passed_safety" : bs_trials_pass,
            "solution" : bs_trials_solution
        })
        results_df.index = bs_trials_index
        results_df.sort_index(inplace=True)
        results_csv_savename = os.path.join(bootstrap_savedir, "all_bs_trials_results.csv")
        results_df.to_csv(results_csv_savename)

        # Compute the probability of passing.
        # TODO: When is this nan? Should we change to nan mean?
        est_prob_pass = np.mean(bs_trials_pass)
        num_trials_passed = np.sum(bs_trials_pass)

        # TODO: Update so delta is passed through to CIs.
        if self.hyperparam_spec.confidence_interval_type == "ttest":
            lower_bound, upper_bound = hp_utils.ttest_bound(self.hyperparam_spec,bs_trials_pass)
        elif self.hyperparam_spec.confidence_interval_type == "clopper-pearson":
            lower_bound, upper_bound = hp_utils.clopper_pearson_bound(self.hyperparam_spec,num_trials_passed)
        else:
            lower_bound, upper_bound = None, None

        # TODO: Update tests to have these returns.
        return est_prob_pass, lower_bound, upper_bound, results_df


    def get_bootstrap_dataset_size(
            self,
            frac_data_in_safety
    ):
        """Computes the number of datapoints that should go into the bootstrapped 
                candidate and safety datasets according to frac_data_in_safety.

        :param frac_data_in_safety: fraction of data in safety set that we want to estimate
                        the probabiilty of returning a solution for
        :type frac_data_in_safety: float
        """
        total_data = self.dataset.num_datapoints # Original dataset size.
        n_bootstrap_samples_safety = int(total_data * frac_data_in_safety) 
        n_bootstrap_samples_candidate = total_data - n_bootstrap_samples_safety

        return n_bootstrap_samples_candidate, n_bootstrap_samples_safety



    def get_est_prob_pass(
        self,
        frac_data_in_safety,
        bootstrap_savedir,
        hyperparam_setting=None
    ):
        """Estimates probability of returning a solution with rho_prime fraction of data
            in candidate selection.

        :param frac_data_in_safety: fraction of data in safety set that we want to
            estimate the probabiilty of returning a solution for
        :type frac_data_in_safety: float
        :param n_bootstrap_samples_candidate: size of candidate dataset sampled in bootstrap
        :type n_boostrap_samples_candidate: int
        :param n_bootstrap_samples_safety: size of safety dataset sampled in bootstrap
        :type n_bootstrap_samples_safety: int
        :param bootstrap_savedir: root diretory to store bootstrap datasets and results
        :type bootstrap_savedir: str
        :type hyperparam_setting: tuple containing hyperparameter values that should be
            set for this bootstrap experiment (if not given will use default from self.spec)
        :type hyperparam_setting: tuple of tuples, where each inner tuple takes the form
            (hyperparameter name, hyperparameter type, hyperparameter value)
                Example:
                (("alpha_theta", "optimization", 0.001), ("num_iters", "optimization", 500))
        """
        # Generate the bootstrapped datsets to use across all trials.
        # TODO: Do we need to think about bootstrap dataset generation in any way for hyperparameters?

        # Create a partial function for run_bootstrap_trial.
       
        partial_kwargs = { 
                "frac_data_in_safety": frac_data_in_safety,
                "parent_savedir": bootstrap_savedir,
                "hyperparam_setting": hyperparam_setting
                }
        helper = partial(self.run_bootstrap_trial, **partial_kwargs)

        # Run the trials.
        bs_trials_ran = [] # List indicating if the bootstrap trial was run.
        start_time = time.time()
        if self.hyperparam_spec.n_bootstrap_workers == 1:
            for bootstrap_trial_i in tqdm(range(self.hyperparam_spec.n_bootstrap_trials), leave=False):
                # TODO: Log bs_trials_run.
                bs_trials_ran.append(helper(bootstrap_trial_i))
        elif self.hyperparam_spec.n_bootstrap_workers > 1: 
            with ProcessPoolExecutor(
                    max_workers=self.hyperparam_spec.n_bootstrap_workers, mp_context=mp.get_context("fork")
            ) as ex:
                for ran_trial in tqdm(
                        ex.map(helper, np.arange(self.hyperparam_spec.n_bootstrap_trials)),
                        total=self.hyperparam_spec.n_bootstrap_trials, leave=False):
                    bs_trials_ran.append(ran_trial)
        else:
            raise ValueError(f"n_workers value of {self.hyperparam_spec.n_bootstrap_workers} must be >=1")
        elapsed_time = time.time() - start_time

        # If trial was run, we want to indicate that at least one trial was run.
        ran_new_bs_trials = any(bs_trials_ran)

        # Accumulate results from bootstrap trials get estimate.
        est_prob_pass, lower_bound, upper_bound, results_df = \
                self.aggregate_est_prob_pass(
                        frac_data_in_safety, bootstrap_savedir)

        # TODO: Update the test for including lower and upper bounds.
        return (est_prob_pass, lower_bound, upper_bound, results_df, elapsed_time, 
                ran_new_bs_trials)



    def get_all_greater_est_prob_pass(
            self,
        ):
        """Compute the estimated probability of passing for all safety fractions in  
            self.all_frac_data_in_safety.
        """
        start_time = time.time()
        all_estimates = {}
        for frac_data_in_safety in self.all_frac_data_in_safety:  
            print("rho:", frac_data_in_safety)
            all_estimates[frac_data_in_safety] = {}
            bootstrap_savedir = os.path.join(self.results_dir,
                    f"bootstrap_safety_frac_{frac_data_in_safety:.2f}")

            # Partition data according to frac_data_in_safety.
            (   candidate_dataset,
                safety_dataset,
            ) = self.create_dataset(self.dataset, frac_data_in_safety, shuffle=False)

            # Need at least 4 points in candidate to bootstrap estimate.
            if candidate_dataset.num_datapoints < 4: 
                continue

            # Estimate the probability of passing for datasplits with more in selection.
            for frac_data_in_safety_prime in self.all_frac_data_in_safety:
                if frac_data_in_safety_prime > frac_data_in_safety:  # Only consider more data in cs, less in safety.
                    continue
                print(" rho':", frac_data_in_safety_prime)

                # Compute desired sizes of the bootstrapped candidate and safety datasets.
                n_bootstrap_samples_candidate, n_bootstrap_samples_safety = \
                    self.get_bootstrap_dataset_size(frac_data_in_safety_prime)

                # Copmute probability of passing.
                prime_prob_pass, _, _, _, _, ran_new_bs_trials = self.get_est_prob_pass(
                    frac_data_in_safety_prime,
                    candidate_dataset,
                    n_bootstrap_samples_candidate,
                    n_bootstrap_samples_safety,
                    bootstrap_savedir,
                )
                print("     prob pass:", prime_prob_pass)
                all_estimates[frac_data_in_safety][frac_data_in_safety_prime] = prime_prob_pass

        elapsed_time = time.time() - start_time
        print("elapsed_time:", elapsed_time)
        return all_estimates, elapsed_time


    def get_gridsearchable_hyperparameter_iterator(
            self
    ):
        """
        Create iterator for every combination of grid-searchable hyperparameter values that we want to
            optimize for.

        """
        all_hyper_iterables = []

        for (hyper_name, hyper_info) in self.hyper_dict.items():
            if hyper_info["tuning_method"] == "grid_search":
                hyper_values, hyper_type = hyper_info["values"], hyper_info["hyper_type"]
                if hyper_name == "frac_data_in_safety":
                    hyper_values = sorted(hyper_values,reverse=True)
                all_hyper_iterables.append([(hyper_name, hyper_type, value) for value in hyper_values])

        return itertools.product(*all_hyper_iterables)

    def create_hyperparam_bootstrap_savedir(
            self,
            hyperparam_setting
    ):
        # TODO: Do we want to generalize to make work for safety_frac as well?
        bootstrap_savedir = "bootstrap"
        for (hyper_name, hyper_type, hyper_value) in hyperparam_setting:
            bootstrap_savedir += f"__{hyper_name}_{hyper_value:.2e}"

        return bootstrap_savedir


    def find_best_hyperparameters(
            self,
            frac_data_in_safety,
            **kwargs
    ):
        """
        Does hyperparameter tuning for all hyperparameters in HyperSchema.hyper_dict.
        Figures out which ones are to be grid-searched and which are to be optimized with 
        CMA-ES, constructs the grid, then runs the tuning. 

        """

        # Make a single directory where bootstrapped datasets will live
        bootstrap_savedir = os.path.join(self.results_dir,'bootstrapped_datasets')
        os.makedirs(bootstrap_savedir, exist_ok=True)

        # Partition data according to frac_data_in_safety
        (   candidate_dataset,
            safety_dataset,
        ) = self.create_dataset(self.dataset, frac_data_in_safety, shuffle=False)

        # Compute desired sizes of the bootstrapped candiate and safety datasets.
        n_bootstrap_samples_candidate, n_bootstrap_samples_safety = \
                self.get_bootstrap_dataset_size(frac_data_in_safety)

        # Create the bootstrapped datasets
        created_trial_datasets = self.generate_all_bootstrap_datasets(
                candidate_dataset, frac_data_in_safety, n_bootstrap_samples_candidate, 
                n_bootstrap_samples_safety, bootstrap_savedir)
        
        # Dictionary mapping hyperparameter setting to the estimated probability.
        all_est_prob_pass = {}

        # Figure out if there are grid searchable hps
        do_grid_search = False
        grid_search_hps = [name for name,hyper_info in self.hyper_dict.items() if hyper_info['tuning_method']=="grid_search"]
        if len(grid_search_hps) > 0:
            do_grid_search=True

        # Figure out if there are CMA-ES searchable hps
        do_cmaes = False
        do_powell = False
        cmaes_hps = [name for name,hyper_info in self.hyper_dict.items() if hyper_info['tuning_method']=="CMA-ES"]
        
        if len(cmaes_hps) == 1:
            # CMA-ES won't work with 1 param, so use powell
            do_powell = True
        if len(cmaes_hps) > 1:
            do_cmaes = True

        if do_grid_search:
            print(f"doing grid search over: {grid_search_hps}")
            for hyperparam_setting in self.get_gridsearchable_hyperparameter_iterator():

                print("hyperparam_setting:")
                print(hyperparam_setting)
                if do_cmaes:
                    print("Running CMA-ES...")
                    # Run CMA-ES where the parameters are the CMA-ES tunable hyperparameters,
                    # keep the grid search ones fixed for this CMA-ES run.
                    curr_prob_pass, best_cmaes_hyperparams = self.run_cmaes(
                        frac_data_in_safety=frac_data_in_safety,
                        fixed_hyperparam_setting=hyperparam_setting,
                        **kwargs)
                    print("Done.")
                    # combine best cmaes params with existing hyperparam setting
                    full_hyperparam_setting = hyperparam_setting + tuple(best_cmaes_hyperparams)
                    print("full_hyperparam_setting:")
                    print(full_hyperparam_setting)
                    all_est_prob_pass[full_hyperparam_setting] = curr_prob_pass
                elif do_powell:
                    print("Running Powell...")
                    # Run Powell with a single hyperparameter to tune,
                    # keeping the grid search parameters fixed.
                    curr_prob_pass, best_cmaes_hyperparams = self.run_powell(
                        frac_data_in_safety=frac_data_in_safety,
                        fixed_hyperparam_setting=hyperparam_setting,
                        **kwargs)
                    print("Done.")
                    full_hyperparam_setting = hyperparam_setting + tuple(best_cmaes_hyperparams)
                    all_est_prob_pass[full_hyperparam_setting] = curr_prob_pass


                else:
                    # No CMA-ES or Powell, just use this combo of hyperparameters
                    # to estimate probability of passing.
                    curr_prob_pass, _, _, _, _, curr_ran_new_bs_trials = self.get_est_prob_pass(
                            frac_data_in_safety,
                            candidate_dataset,
                            n_bootstrap_samples_candidate,
                            n_bootstrap_samples_safety,
                            bootstrap_savedir,
                            hyperparam_setting
                    )
                    all_est_prob_pass[hyperparam_setting] = curr_prob_pass

            # Select the hyperparameter with the highest predicited probability of passing.
            best_hyperparam_setting = max(all_est_prob_pass, key=lambda k: all_est_prob_pass[k])
        
            
        # Set spec with best hyperparameter setting.
        best_hyperparam_spec = hp_utils.set_spec_with_hyperparam_setting(
                copy.deepcopy(self.spec), best_hyperparam_setting)

        return best_hyperparam_setting, best_hyperparam_spec

    def run_cmaes(self,frac_data_in_safety,fixed_hyperparam_setting,**kwargs):
        """ Run CMA-ES over the hyperparameters that 
        we specified in hyper_dict to have tuning_method = "CMA-ES". 
        Use fixed values for all other hyperparams. 
        """  
        if self.write_logfile:
            log_counter = 0
            logdir = os.path.join(os.getcwd(), "cmaes_logs")
            os.makedirs(logdir, exist_ok=True)
            filename = os.path.join(
                logdir, f"cmaes_log{log_counter}.csv"
            )

            while os.path.exists(filename):
                filename = filename.replace(
                    f"log{log_counter}", f"log{log_counter+1}"
                )
                log_counter += 1

            logger = partial(cmaes_logger,filename=filename)
        else:
            logger = None

        opts = {}
        if "maxiter" in kwargs:
            opts["maxiter"] = kwargs["maxiter"]

        if "seed" in kwargs:
            opts["seed"] = kwargs["seed"]

        if "sigma0" in kwargs:
            sigma0 = kwargs["sigma0"]
        else:
            sigma0 = 0.2

        # Map initial hyperparameter values to theta vector
        theta_init = self._get_theta_init_from_hyper_dict()
        print(f"Running CMA-ES with initial theta: {theta_init}")
        es = cma.CMAEvolutionStrategy(theta_init, sigma0, opts)
        partial_objective = partial(
            self.cmaes_objective,
            frac_data_in_safety=frac_data_in_safety,
            fixed_hyperparam_setting=fixed_hyperparam_setting
        )
        es.optimize(partial_objective,callback=logger)
        
        es.disp()
        if self.write_logfile and kwargs["verbose"]:
            print(f"Wrote {filename} with CMA-ES log info")

        best_prob_pass = es.result.fbest

        theta_best = es.result.xbest
        best_hyperparam_setting = self._unpack_theta_to_hyperparam_values(theta_best)
        return best_prob_pass,best_hyperparam_setting


    def run_powell(self,frac_data_in_safety,fixed_hyperparam_setting,**kwargs):
        """ Run Powell minimization over a single hyperparameter. 
        This is the fallback optimizer we use when we only have 1 hyperparameter
        and CMA-ES tuning is specified. CMA-ES is not intended for use in 1D.
        Use fixed values for all other hyperparams. 
        """  
        
        opts = {}
        if "maxiter" in kwargs:
            opts["maxiter"] = kwargs["maxiter"]

        if "seed" in kwargs:
            opts["seed"] = kwargs["seed"]

        if "sigma0" in kwargs:
            sigma0 = kwargs["sigma0"]
        else:
            sigma0 = 0.2

        # Map initial hyperparameter values to theta vector
        theta_init = self._get_theta_init_from_hyper_dict()
        print(f"Running Powell with initial theta: {theta_init}")
        partial_objective = partial(
            self.powell_objective,
            frac_data_in_safety=frac_data_in_safety,
            fixed_hyperparam_setting=fixed_hyperparam_setting
        )

        res = minimize(fun=partial_objective,x0=theta_init,method="Powell")
        theta_best = res.x        
        best_prob_pass = res.fun        
        # if self.write_logfile and kwargs["verbose"]:
        #     print(f"Wrote {filename} with CMA-ES log info")

        print(f"theta_best: {theta_best}")
        best_hyperparam_setting = self._unpack_theta_to_hyperparam_values(theta_best)
        return best_prob_pass,best_hyperparam_setting


    def _get_theta_init_from_hyper_dict(self):
        """ Utility function for packing hyperparam initial values
        into a 1D vector for CMA-ES.
        """
        # self.hyper_dict is an ordered dict 
        theta_init = []
        for (hyper_name, hyper_info) in self.hyper_dict.items():
            if hyper_info["tuning_method"] == "CMA-ES":
                hyper_val = hyper_info["initial_value"]
                x_min,x_max = hyper_info['min_val'],hyper_info['max_val']
                if hyper_info["search_distribution"] == "log-uniform":
                    hyper_val = np.log(hyper_val) 
                    x_min,x_max = np.log(x_min),np.log(x_max) 
                # reverse the sigmoiding process for encoding parameter gets encoded to theta (see _unpack_theta_to_hyperparam_values())
                G = (x_max-x_min)/(hyper_val-x_min)
                theta_ii = np.log(1/(G-1))
                if hyper_info["search_distribution"] == 'log-uniform':
                    theta_ii = np.exp(theta_ii)
                theta_init.append(theta_ii)
        return theta_init

    def _unpack_theta_to_hyperparam_values(self,theta):
        """ Utility function for unpacking hyperparam values 
        from a 1D vector used in CMA-ES to values we can inject into a Seldonian
        Spec object.

        :param theta: Vector of hyperparameters
        """
        hyperparam_setting = []
        for ii,(hyper_name,hyper_info) in enumerate(self.hyper_dict.items()):
            if hyper_info["tuning_method"] != "CMA-ES":
                continue
            x_min,x_max = hyper_info['min_val'],hyper_info['max_val']
            if hyper_info["search_distribution"] == "log-uniform":
                # this ensures that search is done uniformly in log space
                # this is especially important for step size where we want to 
                # explore small step sizes just as often as large step sizes
                x_min,x_max = np.log(x_min),np.log(x_max)
                log_hyper_val = self.sigmoid(theta[ii])*(x_max-x_min) + x_min # squashes to be between [-10,0] for example
                hyper_val = np.exp(log_hyper_val)
            else:
                hyper_val = self.sigmoid(theta[ii])*(x_max-x_min) + x_min  
            
            if hyper_info["dtype"] == "int":    
                hyper_val = int(round(hyper_val))

            hyperparam_setting.append((hyper_name,hyper_info['hyper_type'],hyper_val))

        return hyperparam_setting
    
    def sigmoid(self,x):
        return 1/(1+np.exp(-x))

    def cmaes_objective(
        self,
        theta,
        frac_data_in_safety,
        fixed_hyperparam_setting,
    ):
        """ The objective function that CMA-ES tries to minimize. 
        We want to minimize (1-prob_pass) in order to maximize prob_pass. 
        Need to return the thing we are trying to minimze. 

        :param theta: Vector of hyperparameters
        :param frac_data_in_safety: Fraction of data going to safety test
        :param fixed_hyperparam_setting: The hyperparameters from grid search
            that are frozen for this CMA-ES run.
        """
        # Unpack theta to get hyperparam setting
        hyperparam_setting = self._unpack_theta_to_hyperparam_values(theta)
        # combine the CMA-ES hyperparams with the fixed hyperparams
        full_hyperparam_setting = fixed_hyperparam_setting + tuple(hyperparam_setting)
        print("hyperparams for this iteration of CMAES:")
        print(full_hyperparam_setting)
        if hasattr(self,"cmaes_iteration"):
            self.cmaes_iteration += 1
        else:
            self.cmaes_iteration = 0
        
        iteration_savedir = os.path.join(
            self.results_dir,
            f"frac_data_in_safety_{frac_data_in_safety:.2f}",
            f"cmaes_iteration{self.cmaes_iteration}"
        )

        os.makedirs(iteration_savedir,exist_ok=True)
        # TODO: Log created_trial_datasets

        partial_kwargs = { 
                "frac_data_in_safety": frac_data_in_safety,
                "parent_savedir": iteration_savedir,
                "hyperparam_setting": full_hyperparam_setting
                }
        helper = partial(self.run_bootstrap_trial, **partial_kwargs)

        # Run the trials.
        bs_trials_ran = [] # List indicating if the bootstrap trial was run.
        start_time = time.time()
        if self.hyperparam_spec.n_bootstrap_workers == 1:
            for bootstrap_trial_i in tqdm(range(self.hyperparam_spec.n_bootstrap_trials), leave=False):
                # TODO: Log bs_trials_run.
                bs_trials_ran.append(helper(bootstrap_trial_i))
        elif self.hyperparam_spec.n_bootstrap_workers > 1: 
            with ProcessPoolExecutor(
                    max_workers=self.hyperparam_spec.n_bootstrap_workers, mp_context=mp.get_context("fork")
            ) as ex:
                for ran_trial in tqdm(
                        ex.map(helper, np.arange(self.hyperparam_spec.n_bootstrap_trials)),
                        total=self.hyperparam_spec.n_bootstrap_trials, leave=False):
                    bs_trials_ran.append(ran_trial)
        else:
            raise ValueError(f"n_workers value of {self.hyperparam_spec.n_bootstrap_workers} must be >=1")
        # elapsed_time = time.time() - start_time

        # If trial was run, we want to indicate that at least one trial was run.
        ran_new_bs_trials = any(bs_trials_ran)

        # Accumulate results from bootstrap trials get estimate.
        est_prob_pass, lower_bound, upper_bound, results_df = \
                self.aggregate_est_prob_pass(
                        frac_data_in_safety, iteration_savedir)
        print("est_prob_pass:")
        print(est_prob_pass)
        return 1-est_prob_pass

    def powell_objective(
        self,
        theta,
        frac_data_in_safety,
        fixed_hyperparam_setting,
    ):
        """ The objective function that Powell tries to minimize. 
        We want to minimize (1-prob_pass) in order to maximize prob_pass. 
        Need to return the thing we are trying to minimze. 

        :param theta: Vector of hyperparameters
        :param frac_data_in_safety: Fraction of data going to safety test
        :param fixed_hyperparam_setting: The hyperparameters from grid search
            that are frozen for this run.
        """
        # Unpack theta to get hyperparam setting
        hyperparam_setting = self._unpack_theta_to_hyperparam_values(theta)
        full_hyperparam_setting = fixed_hyperparam_setting + tuple(hyperparam_setting)
        print("hyperparams for this iteration of Powell:")
        print(full_hyperparam_setting)
        print(f"theta: {theta}")
        if hasattr(self,"powell_iteration"):
            self.powell_iteration += 1
        else:
            self.powell_iteration = 0

        iteration_savedir = os.path.join(
            self.results_dir,
            f"frac_data_in_safety_{frac_data_in_safety:.2f}",
            f"powell_iteration{self.powell_iteration}"
        )
        os.makedirs(iteration_savedir,exist_ok=True)
        # TODO: Log created_trial_datasets

        partial_kwargs = { 
                "frac_data_in_safety": frac_data_in_safety,
                "parent_savedir": iteration_savedir,
                "hyperparam_setting": full_hyperparam_setting
                }
        
        helper = partial(self.run_bootstrap_trial, **partial_kwargs)

        # Run the trials.
        bs_trials_ran = [] # List indicating if the bootstrap trial was run.
        start_time = time.time()
        if self.hyperparam_spec.n_bootstrap_workers == 1:
            for bootstrap_trial_i in tqdm(range(self.hyperparam_spec.n_bootstrap_trials), leave=False):
                # TODO: Log bs_trials_run.
                bs_trials_ran.append(helper(bootstrap_trial_i))
        elif self.hyperparam_spec.n_bootstrap_workers > 1: 
            with ProcessPoolExecutor(
                    max_workers=self.hyperparam_spec.n_bootstrap_workers, mp_context=mp.get_context("fork")
            ) as ex:
                for ran_trial in tqdm(
                        ex.map(helper, np.arange(self.hyperparam_spec.n_bootstrap_trials)),
                        total=self.hyperparam_spec.n_bootstrap_trials, leave=False):
                    bs_trials_ran.append(ran_trial)
        else:
            raise ValueError(f"n_workers value of {self.hyperparam_spec.n_bootstrap_workers} must be >=1")
        # elapsed_time = time.time() - start_time

        # If trial was run, we want to indicate that at least one trial was run.
        # ran_new_bs_trials = any(bs_trials_ran)

        # Accumulate results from bootstrap trials get estimate.
        est_prob_pass, lower_bound, upper_bound, results_df = \
                self.aggregate_est_prob_pass(
                        frac_data_in_safety, iteration_savedir)
        print("est_prob_pass:")
        print(est_prob_pass)
        print()
        return 1-est_prob_pass


    def find_best_frac_data_in_safety(
            self,
            threshold=0.01 # TODO: Come up with a better name than this.
    ):
        """Find the best frac_data_in_safety to use for the Seldonian algorithm.

        :return: (frac_data_in_safety, candidate_dataset, safety_dataset). frac_data_in_safety
                indicates the percentage of total data that is included in the safety dataset.
                candidate_dataset and safety_dataset are dataset objects containing data from
                elf.dataset split according to frac_data_in_safety
        :rtyle: Tuple
        """
        # Sort frac data in safety 
        self.all_frac_data_in_safety = \
                self.hyperparam_spec.hyper_schema.hyper_dict["frac_data_in_safety"]["values"]
        self.all_frac_data_in_safety.sort(reverse=True) # Start with most data in safety.

        # TODO: Can we pass in frac_data_in_safety now as a hyperparam_setting? Just a single one?
        # TODO: Update test now that use CI.
        all_est_dict_list = [] # Store dicionaries for dataframe.
        ran_new_bs_trials = False 

        # Move data from safety to cs dataset, high high to low. self.all_frac_data_in_safety
        #   was sorted in init.
        for frac_data_in_safety in self.all_frac_data_in_safety:  
            bootstrap_savedir = os.path.join(self.results_dir, 
                    f"bootstrap_safety_frac_{frac_data_in_safety:.2f}")
            os.makedirs(bootstrap_savedir, exist_ok=True) 

            # Partition data according to frac_data_in_safety.
            (   candidate_dataset,
                safety_dataset,
            ) = self.create_dataset(self.dataset, frac_data_in_safety, shuffle=False)

            # Dictionary mapping est_frac_data_in_safety to the estimated probability.
            all_est_prob_pass = {}

            # Need at least 4 points in candidate to estimate probability of other splits.
            if candidate_dataset.num_datapoints < 4: 
                continue

            # Compute desired sizes of the bootstrapped candidate and safety datasets.
            n_bootstrap_samples_candidate, n_bootstrap_samples_safety = \
                self.get_bootstrap_dataset_size(frac_data_in_safety)

            # Estimate probability of passing.
            curr_prob_pass, curr_lower_bound, curr_upper_bound, _, _, curr_ran_new_bs_trials \
                    = self.get_est_prob_pass(
                            frac_data_in_safety,
                            candidate_dataset,
                            n_bootstrap_samples_candidate,
                            n_bootstrap_samples_safety,
                            bootstrap_savedir,
                        )  
            ran_new_bs_trials = ran_new_bs_trials or curr_ran_new_bs_trials
            all_est_prob_pass[frac_data_in_safety] = curr_prob_pass
            all_est_dict_list.append({
                "frac_data_in_safety": frac_data_in_safety, 
                "est_frac_data_in_safety": frac_data_in_safety,
                "est_prob_pass": curr_prob_pass,
                "est_lower_bound": curr_lower_bound,
                "est_upper_bound": curr_upper_bound,
            })

            # Estimate if any of the future splits of data lead to higher P(pass)
            prime_better = False
            all_prime_below_threshold = False
            for est_frac_data_in_safety in self.all_frac_data_in_safety:
                if est_frac_data_in_safety >= frac_data_in_safety:  # Est if more data in cs.
                    continue

                # Compute desired sizes of the bootstrapped candidate and safety datasets.
                n_bootstrap_samples_candidate, n_bootstrap_samples_safety = \
                    self.get_bootstrap_dataset_size(est_frac_data_in_safety)

                # Compute probability of passing.
                prime_prob_pass, prime_lower_bound, prime_upper_bound, _, _, curr_ran_new_bs_trials \
                        = self.get_est_prob_pass(
                                est_frac_data_in_safety,
                                candidate_dataset,
                                n_bootstrap_samples_candidate,
                                n_bootstrap_samples_safety,
                                bootstrap_savedir,
                            )
                ran_new_bs_trials = ran_new_bs_trials or curr_ran_new_bs_trials
                all_est_prob_pass[est_frac_data_in_safety] = prime_prob_pass
                all_est_dict_list.append({
                    "frac_data_in_safety": frac_data_in_safety, 
                    "est_frac_data_in_safety": est_frac_data_in_safety,
                    "est_prob_pass": prime_prob_pass,
                    "est_lower_bound": prime_lower_bound,
                    "est_upper_bound": prime_upper_bound,
                })

                # Check if estimate is below threshold.
                all_prime_below_threshold = all_prime_below_threshold and (prime_prob_pass <= threshold)

                # Check if ound a future split that we predict is better.
                if self.hyperparam_spec.confidence_interval_type is not None: # Compare confidence intervals.
                    if prime_upper_bound >= curr_lower_bound:
                        prime_better = True
                        break
                else: # Use point estimate to compare.
                    if prime_prob_pass >= curr_prob_pass:
                        prime_better = True
                        break

            # We do not predict that moving more data into the candidate selection is better,
            # so use current rho.
            if not prime_better and not all_prime_below_threshold:
                break

        # Write out all the estimates to a dataframe.
        all_est_csv_savename = os.path.join(self.results_dir, "all_bootstrap_est.csv")
        all_est_df = pd.DataFrame(all_est_dict_list)
        all_est_df.to_csv(all_est_csv_savename)

        # TODO: frac_data_in_safety should be saved... but perhaps at the top level
        return (frac_data_in_safety, candidate_dataset, safety_dataset, ran_new_bs_trials)

