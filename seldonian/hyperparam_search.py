""" Wrapper module for hyperparameter selection for Seldonian algorithms """

import autograd.numpy as np
import pandas as pd

import os
import copy
import time
import pickle
import warnings
import multiprocessing as mp
from tqdm import tqdm
from functools import partial
from concurrent.futures import ProcessPoolExecutor

from seldonian.seldonian_algorithm import SeldonianAlgorithm
from seldonian.dataset import SupervisedDataSet, RLDataSet
from seldonian.candidate_selection.candidate_selection import CandidateSelection
from seldonian.safety_test.safety_test import SafetyTest
from seldonian.models import objectives
from seldonian.utils.io_utils import load_pickle

class HyperparamSearch:
    def __init__(
            self, 
            spec, 
            all_frac_data_in_safety,
            results_dir
    ):
        """Object for finding the best hyperparameters to use to optimize for probability
        of returning a safe solution for Seldonian algorithms. Note: currently only used to
        find optimal data split.

        List of hyperparameters to optimize:
        - Percentage of data in candidate and safety datasets

        :param spec: The specification object with the complete
                set of parameters for running the Seldonian algorithm
        :type spec: :py:class:`.Spec` object
        :param all_frac_data_in_safety: Array containing the values of fraction of data in
                the safety set that are being considered
        :type all_frac_data_in_safety: numpy.ndarray
        """
        self.spec = spec
        self.results_dir = results_dir

        # Sort frac data in safety 
        self.all_frac_data_in_safety = all_frac_data_in_safety
        self.all_frac_data_in_safety.sort(reverse=True) # Start with most data in safety.

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
        """
        n_safety = int(frac_data_in_safety * n_total)
        n_safety = max(n_safety, 2)  # >=2 point in safety
        n_safety = min(n_safety, n_total - 2)  # >=2 point in selection

        return n_safety


    def candidate_safety_combine(
            self,
            candidate_dataset,
            safety_dataset
    ):
        """Combine candidate_dataset and safety_dataset into a full dataset.
        The data will be joined so that the candidate data combes before the 
        safety data.

        :param candidate_dataset: a dataset object containing data
        :type candiddate_dataset: :py:class:`.DataSet` object
        :param safety_dataset: a dataset object containing data
        :type safety_dataset: :py:class:`.DataSet` object
        """
        if self.regime == "supervised_learning":
            combined_num_datapoints = (candidate_dataset.num_datapoints +
                    safety_dataset.num_datapoints)
            # Combine features.
            if ((type(candidate_dataset.features) == list) and 
                    (type(safety_dataset.features) == list)):
                # TODO: Double check this. A little unsure how this data is stored.
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
                    meta_information=candidate_dataset.meta_information)


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


    def create_shuffled_dataset(
            self,
            dataset
    ):
        """Create new dataset containing the same data as the given original dataset,
            but with the data shuffled in a new order.

        :param dataset: a dataset object containing data
        :type dataset: :py:class:`.DataSet` object
        """
        ix_shuffle = np.arange(dataset.num_datapoints)
        np.random.shuffle(ix_shuffle)

        # features can be list of arrays or a single array
        if type(dataset.features) == list:
            resamp_features = [x[ix_shuffle] for x in flist]
        else:
            resamp_features = dataset.features[ix_shuffle]

        # labels and sensitive attributes must be arrays
        resamp_labels = dataset.labels[ix_shuffle]
        if isinstance(dataset.sensitive_attrs, np.ndarray):
            resamp_sensitive_attrs = dataset.sensitive_attrs[ix_shuffle]
        else:
            resamp_sensitive_attrs = []

        shuffled_dataset = SupervisedDataSet(
            features=resamp_features,
            labels=resamp_labels,
            sensitive_attrs=resamp_sensitive_attrs,
            num_datapoints=dataset.num_datapoints,
            meta_information=dataset.meta_information,
        )

        return shuffled_dataset


    def create_dataset(
            self,
            dataset,
            frac_data_in_safety,
            shuffle=False
    ):
        """Partition data to create candidate and safety dataset according to
            frac_data_in_safety. Information about the split can be found in 
            datasplit_info.

        :param dataset: a dataset object containing data
        :type dataset: :py:class:`.DataSet` object
        :param frac_data_in_safety: fraction of data used in safety test,
                the remaining fraction will be used in candidate selection
        :type frac_data_in_safety: float
        :param shuffle: bool indicating if we should shuffle the dataset before 
                splitting it into candidate and safety datasets
        :type shuffle: bool 
        """
        if shuffle:
            dataset = self.create_shuffled_dataset(dataset)

        if self.regime == "supervised_learning":

            if dataset.num_datapoints < 4:
                warning_msg = (
                    "Warning: not enough data to " "run the Seldonian algorithm."
                )
                warnings.warn(warning_msg)

            # Split the data.
            datasplit_info = dict() # Store information about the data split.
            (   datasplit_info["candidate_features"],
                datasplit_info["safety_features"],
                datasplit_info["candidate_labels"],
                datasplit_info["safety_labels"],
                datasplit_info["candidate_sensitive_attrs"],
                datasplit_info["safety_sensitive_attrs"],
                datasplit_info["n_candidate"],
                datasplit_info["n_safety"],
            ) = self.candidate_safety_split(dataset, frac_data_in_safety)

            candidate_dataset = SupervisedDataSet(
                features=datasplit_info["candidate_features"],
                labels=datasplit_info["candidate_labels"],
                sensitive_attrs=datasplit_info["candidate_sensitive_attrs"],
                num_datapoints=datasplit_info["n_candidate"],
                meta_information=dataset.meta_information,
            )

            safety_dataset = SupervisedDataSet(
                features=datasplit_info["safety_features"],
                labels=datasplit_info["safety_labels"],
                sensitive_attrs=datasplit_info["safety_sensitive_attrs"],
                num_datapoints=datasplit_info["n_safety"],
                meta_information=dataset.meta_information,
            )

            if datasplit_info["n_candidate"] < 2 or datasplit_info["n_safety"] < 2:
                warning_msg = (
                    "Warning: not enough data to " "run the Seldonian algorithm."
                )
                warnings.warn(warning_msg)
            if self.spec.verbose:
                print(f"Safety dataset has {datasplit_info['n_safety']} datapoints")
                print(f"Candidate dataset has {datasplit_info['n_candidate']} datapoints")

        elif self.regime == "reinforcement_learning":

            # Split the data.
            datasplit_info = dict() # Store information about the data split.
            (   datasplit_info["candidate_episodes"],
                datasplit_info["safety_episodes"],
                datasplit_info["candidate_sensitive_attrs"],
                datasplit_info["safety_sensitive_attrs"],
                datasplit_info["n_candidate"],
                datasplit_info["n_safety"],
            ) = self.candidate_safety_split(dataset, frac_data_in_safety)

            candidate_dataset = RLDataSet(
                episodes=datasplit_info["candidate_episodes"],
                sensitive_attrs=datasplit_info["candidate_sensitive_attrs"],
                meta_information=datasplit_info["column_names"],
            )

            safety_dataset = RLDataSet(
                episodes=datasplit_info["safety_episodes"],
                sensitive_attrs=datasplit_info["safety_sensitive_attrs"],
                meta_information=datasplit_info["column_names"],
            )

            print(f"Safety dataset has {datasplit_info['n_safety']} episodes")
            print(f"Candidate dataset has {datasplit_info['n_candidate']} episodes")

        return candidate_dataset, safety_dataset, datasplit_info


    def bootstrap_sample_dataset(
            self,
            dataset,
            n_bootstrap_samples,
            savename
    ):
        """Bootstrap sample a dataset of size n_bootstrap_samples from the data points
            in dataset.

        :param dataset: The original dataset from which to resample
        :type dataset: pandas DataFrame
        :param n_bootstrap_samples: The size of the bootstrapped dataset
        :type n_bootstrap_samples: int
        :param savename: Path to save the bootstrapped dataset.
        :type savename: str
        """ 
        if self.regime == "supervised_learning":
            ix_resamp = np.random.choice(
                range(dataset.num_datapoints), n_bootstrap_samples, replace=True
            )
            # features can be list of arrays or a single array
            if type(dataset.features) == list:
                resamp_features = [x[ix_resamp] for x in flist]
            else:
                resamp_features = dataset.features[ix_resamp]

            # labels and sensitive attributes must be arrays
            resamp_labels = dataset.labels[ix_resamp]
            if isinstance(dataset.sensitive_attrs, np.ndarray):
                resamp_sensitive_attrs = dataset.sensitive_attrs[ix_resamp]
            else:
                resamp_sensitive_attrs = []

            bootstrap_dataset = SupervisedDataSet(
                features=resamp_features,
                labels=resamp_labels,
                sensitive_attrs=resamp_sensitive_attrs,
                num_datapoints=n_bootstrap_samples,
                meta_information=dataset.meta_information,
            )

            with open(savename, "wb") as outfile:
                pickle.dump(bootstrap_dataset, outfile)
                if self.spec.verbose:
                    print(f"Saved {savename}")

        elif self.regime == "reinforcement_learning":
            # TODO: Finish implementing this.
            raise NotImplementedError(
                    "Creating bootstrap sampled datasets not yet implemented for "
                    "reinforcement_learning regime")


    def generate_all_bootstrap_datasets(
            self,
            candidate_dataset,
            est_frac_data_in_safety,
            n_bootstrap_trials,
            n_bootstrap_samples_candidate,
            n_bootstrap_samples_safety,
            bootstrap_savedir
    ):
        """Utility function for supervised learning to generate the
        resampled datasets to use in each bootstrap trial. Resamples (with replacement)
        features, labels and sensitive attributes to create n_bootstrap_trials versions 
        of these 

        :param candidate_dataset: Dataset object containing candidate solution dataset.
                This is the dataset we will be bootstrap sampling from.
        :type candidate_dataset: :py:class:`.DataSet` object
        :param est_frac_data_in_safety: fraction of data in safety set that we want to 
                        estimate the probabiilty of returning a solution for
        :type est_frac_data_in_safety: float
        :param n_bootstrap_trials: The number of bootstrap trials, i.e. the number of
                resampled datasets to make
        :type n_bootstrap_trials: int
        :param n_bootstrap_samples_candidate: The size of the candidate selection 
                bootstrapped dataset
        :type n_bootstrap_samples_candidate: int
        :param n_bootstrap_samples_safety: The size of the safety bootstrapped dataset
        :type n_bootstrap_safety: int
        :param bootstrap_savedir: The root diretory to save all the bootstrapped datasets.
        :type bootstrap_savedir: str
        """
        save_subdir = os.path.join(bootstrap_savedir, 
                f"future_safety_frac_{est_frac_data_in_safety:.2f}", "bootstrap_datasets")
        os.makedirs(save_subdir, exist_ok=True)

        for bootstrap_trial_i in range(n_bootstrap_trials):
            # Where to save bootstrapped dataset.
            bootstrap_savename_candidate = os.path.join(save_subdir, 
                    f"bootstrap_trial_{bootstrap_trial_i}_candidate.pkl")
            bootstrap_savename_safety = os.path.join(save_subdir, 
                    f"bootstrap_trial_{bootstrap_trial_i}_safety.pkl")
            bootstrap_savename_datasplit_info = os.path.join(save_subdir,
                    f"bootstrap_trial_{bootstrap_trial_i}_datasplit_info.pkl")

            # Only create datasets if not already existing.
            if not (os.path.exists(bootstrap_savename_safety) and 
                    os.path.exists(bootstrap_savename_candidate) and
                    os.path.exists(bootstrap_savename_datasplit_info)):

                # Partition candidate_dataset into pools to bootstrap 
                (bootstrap_pool_candidate, bootstrap_pool_safety, 
                        bootstrap_datasplit_info) = self.create_dataset(
                                candidate_dataset, est_frac_data_in_safety, shuffle=True)

                # Bootstrap sample candidate selection and safety datasets.
                self.bootstrap_sample_dataset(bootstrap_pool_candidate, 
                        n_bootstrap_samples_candidate, bootstrap_savename_candidate)
                self.bootstrap_sample_dataset(bootstrap_pool_safety,
                        n_bootstrap_samples_safety, bootstrap_savename_safety)

                # Save datasplit_info.
                with open(bootstrap_savename_datasplit_info, "wb") as outfile:
                    pickle.dump(bootstrap_datasplit_info, outfile)
                    if self.spec.verbose:
                        print(f"Saved {savename}")


    def create_bootstrap_trial_spec(
            self,
            bootstrap_trial_i,
            frac_data_in_safety, 
            bootstrap_savedir
    ):
        """Create the spec to run this iteration of the bootstrap triail.

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
        save_subdir = os.path.join(bootstrap_savedir,
                f"future_safety_frac_{frac_data_in_safety:.2f}", "bootstrap_datasets")
        bs_candidate_dataset = load_pickle(os.path.join(save_subdir,
            f"bootstrap_trial_{bootstrap_trial_i}_candidate.pkl"))
        bs_safety_dataset = load_pickle(os.path.join(save_subdir,
            f"bootstrap_trial_{bootstrap_trial_i}_safety.pkl"))
        bs_datasplit_info = load_pickle(os.path.join(save_subdir,
            f"bootstrap_trial_{bootstrap_trial_i}_datasplit_info.pkl"))

        # Combine loaded candidate and safety dataset to create the full dataset.
        combined_dataset = self.candidate_safety_combine(bs_candidate_dataset, bs_safety_dataset)

        # Set the datasets assocaited with the trial.
        spec_for_bootstrap_trial.dataset = combined_dataset 
        spec_for_bootstrap_trial.candidate_dataset = bs_candidate_dataset
        spec_for_bootstrap_trial.safety_dataset = bs_safety_dataset
        spec_for_bootstrap_trial.datasplit_info = bs_datasplit_info
        spec_for_bootstrap_trial.frac_data_in_safety = frac_data_in_safety

        return spec_for_bootstrap_trial


    def run_bootstrap_trial(
            self,
            bootstrap_trial_i,
            **kwargs
    ):
        """Run bootstrap train bootstrap_trial_i to estimate the probability of passing
        with est_frac_data_in_safety.

        :param bootstrap_trial_i: integer indicating which trial of the bootstrap 
            experiment we are currently running. Allows us to identify which bootstrapped
            dataset to load adn run
        :type bootstrap_trial_i: int
        :param est_frac_data_in_safety: fraction of data in safety set that we want to
            estimate the probabiilty of returning a solution for
        :type est_frac_data_in_safety: float
        :param bootstrap_savedir: The root diretory to load bootstrapped dataset and save 
            the result of this bootstrap trial
        :type bootstrap_savedir: str
        """
        est_frac_data_in_safety = kwargs["est_frac_data_in_safety"]
        bootstrap_savedir = kwargs["bootstrap_savedir"]

        # Paths to load datasetsa dnd store results.
        bs_result_subdir = os.path.join(bootstrap_savedir,
                f"future_safety_frac_{est_frac_data_in_safety:.2f}", "bootstrap_results")
        os.makedirs(bs_result_subdir, exist_ok=True)
        bs_result_savename = os.path.join(bs_result_subdir, 
                f"trial_{bootstrap_trial_i}_result.pkl")

        # If this bootstrap trial has already been run,
        if os.path.exists(bs_result_savename):
            # TODO: Make sure verbose gets passed down here.
            if self.spec.verbose:
                print(f"Bootstrap trial {bootstrap_trial_i} has already been run. Skipping.")
            return None

        # Create spec for the bootstrap trial. The bootstrapped candidate and safety
        # datasets are created here.
        spec_for_bootstrap_trial = self.create_bootstrap_trial_spec(bootstrap_trial_i,
                est_frac_data_in_safety, bootstrap_savedir)

        # Run Seldonian Algorithm on the bootstrapped data. Load the datasets here.
        SA = SeldonianAlgorithm(spec_for_bootstrap_trial)
        passed_safety, solution = SA.run(write_cs_logfile=self.spec.verbose, 
                debug=self.spec.verbose)
        trial_result_dict = {"passed_safety" : passed_safety,
                             "solution" : solution}

        # Save the results.
        with open(bs_result_savename, "wb") as outfile:
            pickle.dump(trial_result_dict, outfile)
            if self.spec.verbose:
                print(f"Saved results for bootstrap trial {bootstrap_trial_i} for rho' "
                        "{est_frac_data_in_safety:.2f}")

        return passed_safety, solution


    def aggregate_est_prob_pass(
            self,
            est_frac_data_in_safety,
            bootstrap_savedir, 
            n_bootstrap_trials=None
    ):
        """Compute the estimated probability of passing using the first given
            n_bootstrap_trials. If n_bootstrap_trials is not given, will compute without
            all the result files in bootstrap_savedir.

        :param est_frac_data_in_safety: fraction of data in safety set that we want to 
                        estimate the probabiilty of returning a solution for
        :type est_frac_data_in_safety: float
        :param bootstrap_savedir: root diretory to load results from bootstrap trial, and
            write aggregated result
        :type bootstrap_savedir: str
        :param n_bootstrap_trials: the number of bootstrap trials we are aggregating over.
            If given, we only aggregate up to give number. If not given, will aggregate 
            all trial results in directly
        """
        bs_frac_subdir = os.path.join(bootstrap_savedir,
                f"future_safety_frac_{est_frac_data_in_safety:.2f}")
        bs_result_subdir = os.path.join(bs_frac_subdir, "bootstrap_results")

        # Compute the number of trial results to compute over.
        if n_bootstrap_trials is None:
            n_bootstrap_trials = len(os.listdir(bs_result_subdir))
 
        # Load the result for each trial.
        bs_trials_pass = []
        bs_trials_solution = []
        for bootstrap_trial_i in range(n_bootstrap_trials):
            result_trial_savename = os.path.join(bs_result_subdir,
                f"trial_{bootstrap_trial_i}_result.pkl")
            result_trial_dict = load_pickle(result_trial_savename)
            bs_trials_pass.append(result_trial_dict["passed_safety"])
            bs_trials_solution.append(result_trial_dict["solution"])

        # Create dataframe containing data.
        results_df = pd.DataFrame(data = {"passed_safety" : bs_trials_pass,
                                          "solution" : bs_trials_solution})
        results_df_savename = os.path.join(bs_frac_subdir, "all_results_df.pkl")
        with open(results_df_savename, "wb") as outfile:
            pickle.dump(results_df, outfile)

        # Compute the probability of passing.
        est_prob_pass = np.mean(bs_trials_pass)
        return est_prob_pass, results_df


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
        total_data = self.dataset.num_datapoints 
        n_bootstrap_samples_candidate = int(total_data * frac_data_in_safety) 
        n_bootstrap_samples_safety = total_data - n_bootstrap_samples_candidate

        return n_bootstrap_samples_candidate, n_bootstrap_samples_safety


    def get_est_prob_pass(
        self,
        est_frac_data_in_safety,
        candidate_dataset,
        n_candidate,
        n_safety,
        bootstrap_savedir,
        n_bootstrap_trials,
        n_workers,
    ):
        """Estimates probability of returning a solution with rho_prime fraction of data
            in candidate selection.

        :param est_frac_data_in_safety: fraction of data in safety set that we want to
            estimate the probabiilty of returning a solution for
        :type est_frac_data_in_safety: float
        :param candidate_dataset: a dataset object containing candidate solution dataset.
            This is the dataset we will bootstrap sample from to compute estimate.
        :type candidate_dataset: :py:class:`.DataSet` object
        :param n_candidate: size of true candidate dataset
        :type n_safety: int
        :param n_safety: size of true safety dataset
        :type n_safety: int
        :param bootstrap_savedir: root diretory to store bootstrap datasets and results
        :type bootstrap_savedir: str
        :param n_bootstrap_trials: number of trials to run to get bootstrap estimate
        :type n_bootstrap_trials: int
        :param n_workers: the number of workers to use to run the experiment
        :type n_workers: int
        """
        # Desired size of bootstrapped datasets according to est_frac_data_in_safety.
        n_bootstrap_samples_candidate, n_bootstrap_samples_safety = self.get_bootstrap_dataset_size(
                est_frac_data_in_safety)

        # Generate the bootstrapped datsets to use across all trials.
        self.generate_all_bootstrap_datasets(candidate_dataset, est_frac_data_in_safety,
                n_bootstrap_trials, n_bootstrap_samples_candidate, n_bootstrap_samples_safety,
                bootstrap_savedir)

        # Create a partial function for run_bootstrap_trial.
        partial_kwargs = { 
                "est_frac_data_in_safety": est_frac_data_in_safety,
                "bootstrap_savedir": bootstrap_savedir}
        helper = partial(self.run_bootstrap_trial, **partial_kwargs)

        # Run the trials.
        start_time = time.time()
        if n_workers == 1:
            for bootstrap_trial_i in tqdm(range(n_bootstrap_trials)):
                helper(bootstrap_trial_i)
        elif n_workers > 1: 
            with ProcessPoolExecutor(
                    max_workers=n_workers, mp_context=mp.get_context("fork")
            ) as ex:
                tqdm(ex.map(helper, np.arange(n_bootstrap_trials)),
                        total=n_bootstrap_trials)
        else:
            raise ValueError(f"n_workers value of {n_workers} must be >=1")
        elapsed_time = time.time() - start_time

        # Accumulate results from bootstrap trials get estimate.
        est_prob_pass, results_df = self.aggregate_est_prob_pass(
            est_frac_data_in_safety, bootstrap_savedir, n_bootstrap_trials)

        return est_prob_pass, results_df, elapsed_time


    def find_best_hyperparams(
            self,
            n_bootstrap_trials=100,
            n_workers=1
    ):
        """Find the best hyperparameter values to use for the Seldonian algorithm.
        Note: currently only implemented for frac_data_in_safety.

        :param n_bootstrap_trials: number of trials to run to get bootstrap estimate
        :type n_bootstrap_trials: int
        :param n_workers: the number of workers to use to run the experiment
        :type n_workers: int

        :return: (frac_data_in_safety, candidate_dataset, safety_dataset). frac_data_in_safety
                indicates the percentage of total data that is included in the safety dataset.
                candidate_dataset and safety_dataset are dataset objects containing data from
                self.dataset split according to frac_data_in_safety
        :rtyle: Tuple
        """
        for frac_data_in_safety in self.all_frac_data_in_safety:  # Move data from safety to cs dataset, high high to low.
            bootstrap_savedir = os.path.join(self.results_dir,
                    f"bootstrap_safety_frac_{frac_data_in_safety:.2f}")

            # TODO: Ensure that this is being done the same
            # Partition data according to frac_data_in_safety.
            (   candidate_dataset,
                safety_dataset,
                datasplit_info
            ) = self.create_dataset(self.dataset, frac_data_in_safety, shuffle=False)

            # Estimate probability of passing.
            curr_prob_pass, _, _  = self.get_est_prob_pass(
                frac_data_in_safety,
                candidate_dataset,
                datasplit_info["n_candidate"],
                datasplit_info["n_safety"],
                bootstrap_savedir,
                n_bootstrap_trials,
                n_workers
            )  #

            # Estimate if any of the future splits of data lead to higher P(pass)
            prime_better = False
            for frac_data_in_safety_prime in self.all_frac_data_in_safety:
                if frac_data_in_safety_prime >= frac_data_in_safety:  # Only consider more data in cs.
                    continue

                prime_prob_pass, _, _ = self.get_est_prob_pass(
                    frac_data_in_safety_prime,
                    candidate_dataset,
                    datasplit_info["n_candidate"],
                    datasplit_info["n_safety"],
                    bootstrap_savedir,
                    n_bootstrap_trials,
                    n_workers
                )
                if (
                    prime_prob_pass > curr_prob_pass 
                ):  # Found a future split that we predict is better.
                    prime_better = True
                    break

            # We do not predict that moving more data into the candidate selection is better.
            if prime_better is False:
                break

        return (frac_data_in_safety, candidate_dataset, safety_dataset)


