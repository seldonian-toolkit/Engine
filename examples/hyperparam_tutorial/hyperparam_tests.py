# TODO: Integrate these tests to main test.

import os
import shutil
import pickle
import autograd.numpy as np   # Thinly-wrapped version of Numpy
import random
import time
import pandas as pd
from tqdm import tqdm
from seldonian.utils.io_utils import load_pickle
from seldonian.models.models import LinearRegressionModel
from seldonian.spec import SupervisedSpec
from seldonian.spec import HyperparameterSelectionSpec
from seldonian.seldonian_algorithm import SeldonianAlgorithm
from seldonian.hyperparam_search import HyperparamSearch, HyperSchema
from seldonian.utils.tutorial_utils import (
    make_synthetic_regression_dataset)
from seldonian.parse_tree.parse_tree import (
    make_parse_trees_from_constraints)


# ================================== Set-Up ============================================
def create_test_SA_spec(num_points=1000, frac_data_in_safety=0.6, seed=0):
    if seed is not None:
        np.random.seed(seed)

    # 1. Define the data - X ~ N(0,1), Y ~ X + N(0,1)
    dataset = make_synthetic_regression_dataset(num_points=num_points)

    # 2. Create parse trees from the behavioral constraints constraint strings:
    constraint_strs = ['Mean_Squared_Error >= 1.25','Mean_Squared_Error <= 2.0']
    # confidence levels: 
    deltas = [0.1,0.1] 

    parse_trees = make_parse_trees_from_constraints(
        constraint_strs,deltas)

    # 3. Define underlying machine learning model.
    model = LinearRegressionModel()

    # 4. Create specs object.
    SA_spec = SupervisedSpec(
        dataset=dataset,
        model=model,
        parse_trees=parse_trees,
        sub_regime='regression',
        frac_data_in_safety=frac_data_in_safety,
    )

    return SA_spec

def create_HS_spec(all_frac_data_in_safety, n_bootstrap_trials=100, n_bootstrap_workers=30,
        use_bs_pools=False):
    hyper_schema = HyperSchema(
            {
                "frac_data_in_safety": {
                    "values": all_frac_data_in_safety,
                    "hyper_type": "SA"
                    },
                }
            )

    HS_spec = HyperparameterSelectionSpec(
            hyper_schema=hyper_schema,
            n_bootstrap_trials=n_bootstrap_trials,
            n_bootstrap_workers=n_bootstrap_workers,
            use_bs_pools=use_bs_pools,
            confidence_interval_type=None
    )

    return HS_spec
    

def get_true_prob_pass(all_frac_data_in_safety, num_datapoints=1000):
    """
    Using test_spec, computes the probability of passing for all_frac_data_in_safety.
    """ 
    num_trials = 100
    all_prob_pass = {} # Map fraction to the estimated probability of passing.

    for frac_data_in_safety in all_frac_data_in_safety:

        pass_count = 0
        for trial in tqdm(range(num_trials), leave=False):
            SA_spec = create_test_SA_spec(frac_data_in_safety=frac_data_in_safety, seed=None)
            SA = SeldonianAlgorithm(SA_spec)
            passed_safety, solution = SA.run()
            if passed_safety: pass_count += 1

        all_prob_pass[frac_data_in_safety] = pass_count / num_trials
        print(frac_data_in_safety, pass_count / num_trials)

    # Expected for 1000 datapoints:
    # {0.1:0.26, 0.2:0.57, 0.3:0.78, 0.4:0.89, 0.5:0.91, 0.6:0.92, 0.7:0.97, 0.8:0.97, 0.9:0.95}
    # Expected for 2000 datapoints:
    # {0.1:0.24, 0.2:0.57, 0.3:0.69, 0.4:0.84, 0.5:0.9, 0.6: 0.97, 0.7: 0.98, 0.8: 0.96, 0.9: 0.92}
    return all_prob_pass


def create_test_aggregate_est_prob_pass_files(bootstrap_savedir, est_frac_data_in_safety):
    results_dir = os.path.join(bootstrap_savedir, 
            f"future_safety_frac_{est_frac_data_in_safety:.2f}/bootstrap_results")
    os.makedirs(results_dir, exist_ok=True)
    num_trials = 20
    all_trial_i = np.concatenate((np.arange(10) , np.arange(20, 30)))
    all_trial_pass = [True, False, True, True, False, True, True, False, False, False, True,
                     False, True, False, True, False, True, False, True, False] 
    all_trial_solutions = np.random.rand(num_trials)

    # Write out the trial results.
    for index in range(num_trials):
        bootstrap_trial_i = all_trial_i[index]
        bs_trial_savename = os.path.join(results_dir, f"trial_{bootstrap_trial_i}_result.pkl")
        with open(bs_trial_savename, "wb") as outfile:
            pickle.dump({
                "bootstrap_trial_i" : bootstrap_trial_i,
                "passed_safety": all_trial_pass[index],
                "solution": all_trial_solutions[index]},
            outfile)

    # Return the synthetic data for comparing.
    return all_trial_i, all_trial_pass, all_trial_solutions


def create_test_save_all_bootrap_est_files(results_dir):
    shutil.copytree("test/test_save_all_bootstrap_est_info", results_dir)


# ================================== Tests ============================================

def test_frac_sort():
    """
    Test that the safety frac are being traversed in the correct order.
    Should be sorted from high to low, so we can start with the most data in safety, and then
        move into candidate selection.
    """
    all_frac_data_in_safety = [0.5, 0.1, 0.2, 0.9, 0.77]
    SA_spec = create_test_SA_spec()
    HS_spec = create_HS_spec(all_frac_data_in_safety)
    HS = HyperparamSearch(SA_spec, HS_spec, "test")
    assert(np.allclose(HS.all_frac_data_in_safety, [0.9, 0.77, 0.5, 0.2, 0.1]))
    print("test_frac_sort passed")


def test_get_safety_size():
    all_frac_data_in_safety = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    SA_spec = create_test_SA_spec()
    HS_spec = create_HS_spec(all_frac_data_in_safety)
    HS = HyperparamSearch(SA_spec, HS_spec, "test")

    # Test for less than 4 datapoints.
    n_total = 1
    for frac in all_frac_data_in_safety:
        assert(HS.get_safety_size(n_total, frac) == 0)

    n_total = 2
    assert(HS.get_safety_size(n_total, 0.1) == 0)
    assert(HS.get_safety_size(n_total, 0.2) == 0)
    assert(HS.get_safety_size(n_total, 0.3) == 0)
    assert(HS.get_safety_size(n_total, 0.4) == 0)
    assert(HS.get_safety_size(n_total, 0.5) == 1)
    assert(HS.get_safety_size(n_total, 0.6) == 1)
    assert(HS.get_safety_size(n_total, 0.7) == 1)
    assert(HS.get_safety_size(n_total, 0.8) == 1)
    assert(HS.get_safety_size(n_total, 0.9) == 1)

    n_total = 3
    assert(HS.get_safety_size(n_total, 0.1) == 0)
    assert(HS.get_safety_size(n_total, 0.2) == 0)
    assert(HS.get_safety_size(n_total, 0.3) == 0)
    assert(HS.get_safety_size(n_total, 0.4) == 1)
    assert(HS.get_safety_size(n_total, 0.5) == 1)
    assert(HS.get_safety_size(n_total, 0.6) == 1)
    assert(HS.get_safety_size(n_total, 0.7) == 2)
    assert(HS.get_safety_size(n_total, 0.8) == 2)
    assert(HS.get_safety_size(n_total, 0.9) == 2)

    # Test for 4 datapoints, should always have 2.
    n_total = 4
    for frac in all_frac_data_in_safety:
        assert(HS.get_safety_size(n_total, frac) == 2)

    # Test for 10 datapoints.
    n_total = 10
    assert(HS.get_safety_size(n_total, 0.1) == 2)
    assert(HS.get_safety_size(n_total, 0.2) == 2)
    assert(HS.get_safety_size(n_total, 0.3) == 3)
    assert(HS.get_safety_size(n_total, 0.4) == 4)
    assert(HS.get_safety_size(n_total, 0.5) == 5)
    assert(HS.get_safety_size(n_total, 0.6) == 6)
    assert(HS.get_safety_size(n_total, 0.7) == 7)
    assert(HS.get_safety_size(n_total, 0.8) == 8)
    assert(HS.get_safety_size(n_total, 0.9) == 8)

    print("test_get_safety_size passed")


def test_candidate_safety_split():
    # 1. Single point.
    num_points = 1
    all_frac_data_in_safety = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    SA_spec = create_test_SA_spec(num_points)
    HS_spec = create_HS_spec(all_frac_data_in_safety)
    HS = HyperparamSearch(SA_spec, HS_spec, "test")
    for frac in all_frac_data_in_safety:
        F_c, F_s, L_c, L_s, S_c, S_s, n_candidate, n_safety = HS.candidate_safety_split(
                HS.dataset, frac)
        assert(n_candidate == 1)
        assert(n_safety == 0)
        assert(F_c.shape[0] == n_candidate)
        assert(L_c.shape[0] == n_candidate)
        assert(F_s.shape[0] == n_safety)
        assert(L_s.shape[0] == n_safety)

    # 2. No points.
    num_points = 0
    all_frac_data_in_safety = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    SA_spec = create_test_SA_spec(num_points)
    HS_spec = create_HS_spec(all_frac_data_in_safety)
    HS = HyperparamSearch(SA_spec, HS_spec, "test")
    for frac in all_frac_data_in_safety:
        F_c, F_s, L_c, L_s, S_c, S_s, n_candidate, n_safety = HS.candidate_safety_split(
                HS.dataset, frac)
        assert(n_candidate == 0)
        assert(n_safety == 0)
        assert(F_c.shape[0] == n_candidate)
        assert(L_c.shape[0] == n_candidate)
        assert(F_s.shape[0] == n_safety)
        assert(L_s.shape[0] == n_safety)

    # 3. Normal amount of points.
    num_points = 100
    all_frac_data_in_safety = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    SA_spec = create_test_SA_spec(num_points)
    HS_spec = create_HS_spec(all_frac_data_in_safety)
    HS = HyperparamSearch(SA_spec, HS_spec, "test")
    for frac in all_frac_data_in_safety:
        F_c, F_s, L_c, L_s, S_c, S_s, n_candidate, n_safety = HS.candidate_safety_split(
                HS.dataset, frac)
        expected_n_safety = int(frac * num_points)
        expected_n_candidate = num_points - expected_n_safety
        assert(n_candidate == expected_n_candidate)
        assert(n_safety == expected_n_safety)
        assert(F_c.shape[0] == n_candidate)
        assert(L_c.shape[0] == n_candidate)
        assert(F_s.shape[0] == n_safety)
        assert(L_s.shape[0] == n_safety)

    print("test_candidate_safety_split passed")


def test_create_dataset():
    num_points = 100
    all_frac_data_in_safety = [0.9, 0.77, 0.5, 0.2, 0.1]
    SA_spec = create_test_SA_spec(num_points)
    HS_spec = create_HS_spec(all_frac_data_in_safety)
    HS = HyperparamSearch(SA_spec, HS_spec, "test")

    for frac_data_in_safety in all_frac_data_in_safety:
        (candidate_dataset, safety_dataset) = HS.create_dataset(
                HS.dataset, frac_data_in_safety, shuffle=False)
        expected_n_safety = int(num_points * frac_data_in_safety)
        expected_n_candidate = num_points - expected_n_safety

        # Check that the saved dataset sizes are what we expect.
        assert(safety_dataset.num_datapoints == expected_n_safety)
        assert(candidate_dataset.num_datapoints == expected_n_candidate)
        assert(safety_dataset.features.shape[0] == expected_n_safety)
        assert(candidate_dataset.features.shape[0] == expected_n_candidate)
        assert(safety_dataset.labels.shape[0] == expected_n_safety)
        assert(candidate_dataset.labels.shape[0] == expected_n_candidate)

    print("test_create_dataset passed")


def test_candidate_safety_combine():
    frac_data_in_safety = 0.7
    SA_spec = create_test_SA_spec()
    HS_spec = create_HS_spec([frac_data_in_safety])
    HS = HyperparamSearch(SA_spec, HS_spec, "test")
    candidate_dataset, safety_dataset = HS.create_dataset(
            HS.dataset, frac_data_in_safety)

    # Re-join candidate and safety dataset and compare to the original dataset.
    combined_dataset = HS.candidate_safety_combine(candidate_dataset, safety_dataset)
    assert(np.allclose(SA_spec.dataset.features, combined_dataset.features))
    assert(np.allclose(SA_spec.dataset.labels, combined_dataset.labels))
    assert(np.allclose(SA_spec.dataset.sensitive_attrs, combined_dataset.sensitive_attrs))
    assert(SA_spec.dataset.num_datapoints == combined_dataset.num_datapoints)
    assert(SA_spec.dataset.meta== combined_dataset.meta)

    print("test_candidate_safety_combine passed")


def test_bootstrap_sample_dataset():
    num_points = 100 
    curr_frac_data_in_safety = 0.4
    all_frac_data_in_safety = [0.9, 0.77, 0.5, 0.2, 0.1]
    SA_spec = create_test_SA_spec(num_points)
    HS_spec = create_HS_spec(all_frac_data_in_safety)
    HS = HyperparamSearch(SA_spec, HS_spec, "test")

    # Bootstrap sampling from candidate_dataset.
    n_bootstrap_samples = 50 # Sample more data.
    candidate_dataset, safety_dataset = HS.create_dataset(
            HS.dataset, curr_frac_data_in_safety, shuffle=False)
    bootstrap_dataset = HS.bootstrap_sample_dataset(candidate_dataset, n_bootstrap_samples)

    # Check the stored number of points match with the bootstrap sample.
    assert(bootstrap_dataset.num_datapoints == n_bootstrap_samples)
    assert(len(bootstrap_dataset.features) == n_bootstrap_samples)
    assert(len(bootstrap_dataset.labels) == n_bootstrap_samples)
    if isinstance(HS.dataset.sensitive_attrs, np.ndarray):
        assert(len(bootstrap_dataset.sensitive_attrs) == n_bootstrap_samples)

    # Check that data in bootstrap_dataset is only from candidate_dataset.
    # Note that in our case, features is 1d so we can do this.
    unique_bootstrap_features = set(np.unique(bootstrap_dataset.features))
    unique_candidate_features = set(np.unique(candidate_dataset.features))
    unique_safety_features = set(np.unique(safety_dataset.features))
    assert(len(unique_bootstrap_features) <= len(unique_candidate_features))
    assert(unique_bootstrap_features.issubset(unique_candidate_features))
    assert(len(unique_bootstrap_features.intersection(unique_candidate_features)) ==
            len(unique_bootstrap_features))
    assert(len(unique_bootstrap_features.intersection(unique_safety_features)) == 0)

    print("test_bootstrap_sample_dataset passed")


def test_create_shuffled_dataset():
    # 1. Normal amount of datapoints.
    num_points = 100 
    all_frac_data_in_safety = [0.9, 0.77, 0.5, 0.2, 0.1]
    SA_spec = create_test_SA_spec(num_points)
    HS_spec = create_HS_spec(all_frac_data_in_safety)
    HS = HyperparamSearch(SA_spec, HS_spec, "test")

    shuffled_dataset = HS.create_shuffled_dataset(HS.dataset)

    orig_sorted_features = np.sort(HS.dataset.features.flatten())
    orig_sorted_labels = np.sort(HS.dataset.labels.flatten())
    shuffled_sorted_features = np.sort(shuffled_dataset.features.flatten())
    shuffled_sorted_labels = np.sort(shuffled_dataset.labels.flatten())

    # Check that same data in shuffled dataset.
    assert(np.allclose(orig_sorted_features, shuffled_sorted_features))
    assert(np.allclose(orig_sorted_labels, shuffled_sorted_labels))

    # Check that data is shuffled.
    assert(not(np.allclose(HS.dataset.features, shuffled_dataset.features)))
    assert(not(np.allclose(HS.dataset.labels, shuffled_dataset.labels)))

    # TODO: Check that after shuffing, things still match.

    # 2. Test for edge cases of single data point.
    num_points = 1
    all_frac_data_in_safety = [0.9, 0.77, 0.5, 0.2, 0.1]
    SA_spec = create_test_SA_spec(num_points)
    HS_spec = create_HS_spec(all_frac_data_in_safety)
    HS = HyperparamSearch(SA_spec, HS_spec, "test")

    shuffled_dataset = HS.create_shuffled_dataset(HS.dataset)

    # Just chek nothing got messed up.
    assert(np.allclose(HS.dataset.features, shuffled_dataset.features))
    assert(np.allclose(HS.dataset.labels, shuffled_dataset.labels))

    # 3. Test for edge cases of no data points.
    num_points = 0
    all_frac_data_in_safety = [0.9, 0.77, 0.5, 0.2, 0.1]
    SA_spec = create_test_SA_spec(num_points)
    HS_spec = create_HS_spec(all_frac_data_in_safety)
    HS = HyperparamSearch(SA_spec, HS_spec, "test")

    shuffled_dataset = HS.create_shuffled_dataset(HS.dataset)

    # Just chek nothing got messed up.
    assert(np.allclose(shuffled_dataset.features, []))
    assert(np.allclose(shuffled_dataset.labels, []))

    print("test_create_shuffled_dataset passed")


def test_get_bootstrap_dataset_size():
    num_points = 99
    all_frac_data_in_safety = [0.9, 0.77, 0.5, 0.2, 0.1]
    SA_spec = create_test_SA_spec(num_points)
    HS_spec = create_HS_spec(all_frac_data_in_safety)
    HS = HyperparamSearch(SA_spec, HS_spec, "test")

    for frac_data_in_safety in all_frac_data_in_safety:
        n_bs_candidate, n_bs_safety = HS.get_bootstrap_dataset_size(frac_data_in_safety)
        assert(n_bs_candidate + n_bs_safety == num_points)

    n_bs_candidate, n_bs_safety = HS.get_bootstrap_dataset_size(0.9)
    assert(n_bs_candidate == 10)
    assert(n_bs_safety == 89)

    print("test_get_bootstrap_dataset_size passed")


def test_generate_all_bootstrap_datasets():
    """Test that bootstrapped datasets are generated for each trial.
    
    Note that all tests relating to checking the actual data in the generated bootstrapped  
        datasets is done in test_bootstrap_sample_dataset.
    """

    # ==================== Normal test, normal amount of data ==========================
    # Remove existing test data directory.
    test_dir = "test/test_generate_all_bootstrap_datasets"
    if os.path.exists(test_dir): shutil.rmtree(test_dir)

    num_points = 100 
    bs_all_frac_data_in_safety = [0.9, 0.77, 0.5, 0.2, 0.1]
    SA_spec = create_test_SA_spec(num_points)
    HS_spec = create_HS_spec(bs_all_frac_data_in_safety)
    HS = HyperparamSearch(SA_spec, HS_spec, test_dir)

    curr_frac_data_in_safety = 0.4
    candidate_dataset, safety_dataset  = HS.create_dataset(
            HS.dataset, curr_frac_data_in_safety, shuffle=False)

    n_bootstrap_trials = 10
    n_bootstrap_candidate = 110
    n_bootstrap_safety = 50

    all_created_trials = {}
    for frac_data_in_safety in bs_all_frac_data_in_safety:
        all_created_trials[frac_data_in_safety] = HS.generate_all_bootstrap_datasets(
                candidate_dataset, frac_data_in_safety, n_bootstrap_trials,
                n_bootstrap_candidate, n_bootstrap_safety, test_dir)

    # Test all fractions of data are created.
    all_data_dir_expected = sorted(["future_safety_frac_0.90", "future_safety_frac_0.77",
        "future_safety_frac_0.50", "future_safety_frac_0.20", "future_safety_frac_0.10"])
    all_data_dir = sorted(os.listdir(test_dir))
    assert(all_data_dir == all_data_dir_expected)

    for frac_data_in_safety in bs_all_frac_data_in_safety:
        # Check that the correct trials are run.
        assert(all_created_trials[frac_data_in_safety] == list(range(n_bootstrap_trials)))

        # Check that all the expected folders are created.
        expected_datasets = sorted([f"bootstrap_datasets_trial_{i}.pkl" for i 
            in range(n_bootstrap_trials)])
        existing_datasets = sorted(os.listdir(os.path.join(test_dir, 
            f"future_safety_frac_{frac_data_in_safety:.2f}", "bootstrap_datasets")))
        assert(expected_datasets == existing_datasets)


    # ======================= Adding additonal trials ===============================
    new_n_bootstrap_trials = 50

    all_created_trials = {}
    for frac_data_in_safety in bs_all_frac_data_in_safety:
        all_created_trials[frac_data_in_safety] = HS.generate_all_bootstrap_datasets(
                candidate_dataset, frac_data_in_safety, new_n_bootstrap_trials, 
                n_bootstrap_candidate, n_bootstrap_safety, test_dir)

    # Test that the correct number of each folder is created, now that we added trials.
    for frac_data_in_safety in bs_all_frac_data_in_safety:
        # Check that the correct trials are run.
        assert(all_created_trials[frac_data_in_safety] == list(range(
            n_bootstrap_trials, new_n_bootstrap_trials)))

        # Check that all the expected folders are created.
        expected_datasets = sorted([f"bootstrap_datasets_trial_{i}.pkl" for i 
            in range(new_n_bootstrap_trials)])
        existing_datasets = sorted(os.listdir(os.path.join(test_dir, 
            f"future_safety_frac_{frac_data_in_safety:.2f}", "bootstrap_datasets")))
        assert(expected_datasets == existing_datasets)

    # ========================= Edge case, not enough data ===========================

    # Remove existing test data directory.
    test_dir = "test/test_generate_all_bootstrap_datasets"
    if os.path.exists(test_dir): shutil.rmtree(test_dir)

    num_points = 3 
    bs_all_frac_data_in_safety = [0.9, 0.77, 0.5, 0.2, 0.1]
    SA_spec = create_test_SA_spec(num_points)
    HS_spec = create_HS_spec(bs_all_frac_data_in_safety)
    HS = HyperparamSearch(SA_spec, HS_spec, test_dir)

    curr_frac_data_in_safety = 0.4
    candidate_dataset, safety_dataset = HS.create_dataset(
            HS.dataset, curr_frac_data_in_safety, shuffle=False)

    n_bootstrap_trials = 10
    n_bootstrap_candidate = 110
    n_bootstrap_safety = 50
    for frac_data_in_safety in bs_all_frac_data_in_safety:
        HS.generate_all_bootstrap_datasets(candidate_dataset, frac_data_in_safety,
                n_bootstrap_trials, n_bootstrap_candidate, n_bootstrap_safety, test_dir)

    # Check that data is not even created.
    assert(os.path.exists(test_dir) is False)

    print("test_generate_all_bootstrap_datasets passed")


def test_SA_datasplit_loading():
    """Check that when give SA pre-split candidate_dataset and safety_dataset in the spec,
        produces the same answer.
    """
    # Use the same spec for both.
    SA_spec = create_test_SA_spec()

    # Run the seldonian algorithm using the spec object.
    SA = SeldonianAlgorithm(SA_spec)
    passed_safety, solution = SA.run()

    # Use HS to get split dataset.
    HS_spec = create_HS_spec([SA_spec.frac_data_in_safety])
    HS = HyperparamSearch(SA_spec, HS_spec, "test")
    candidate_dataset, safety_dataset = HS.create_dataset(
            SA_spec.dataset, SA_spec.frac_data_in_safety, shuffle=False)

    # Create a spec that contains the split candidate and safety datasets again.
    SA_spec_split = SupervisedSpec(
            dataset=SA_spec.dataset,
            candidate_dataset=candidate_dataset,
            safety_dataset=safety_dataset,
            model=SA_spec.model,
            parse_trees=SA_spec.parse_trees,
            sub_regime='regression',
    )

    SA_split = SeldonianAlgorithm(SA_spec_split)
    passed_safety_split, solution_split = SA_split.run()

    assert(passed_safety_split == passed_safety)
    assert(np.allclose(solution, solution_split, atol=1e-2))
    # Note the slight difference in answer above is from the different order that we call run.


def test_run_bootstrap_trial():
    # TODO: Test exception catching here... Catch exceptions when doing SA.run.
    """
    Test that loads (1) correct data by name, (2) successfully runs seldonian (compare with 
        true seldonian result), (3) saves data correctly in the right place.
    """
    bootstrap_savedir = "test/test_run_bootstrap_trial"
    if os.path.exists(bootstrap_savedir): shutil.rmtree(bootstrap_savedir)

    SA_spec = create_test_SA_spec()
    bootstrap_trial_i = 13
    est_frac_data_in_safety = SA_spec.frac_data_in_safety

    # Run the seldonian algorihtm using the spec object.
    SA = SeldonianAlgorithm(SA_spec)
    SA_passed_safety, SA_solution = SA.run() # This is what we compare the solution to.

    # Use HS to get split dataset.
    bootstrap_datasets_dict = dict()
    HS_spec = create_HS_spec([SA_spec.frac_data_in_safety])
    HS = HyperparamSearch(SA_spec, HS_spec, bootstrap_savedir)
    (bootstrap_datasets_dict["candidate"], bootstrap_datasets_dict["safety"]) = \
            HS.create_dataset(SA_spec.dataset, SA_spec.frac_data_in_safety, shuffle=False)

    # Create test data to use in this trial. Just store the candidate and safety dataset
    # from a normal split, so that we can compare with SA.
    datasets_subdir = os.path.join(bootstrap_savedir,
            f"future_safety_frac_{est_frac_data_in_safety:.2f}", "bootstrap_datasets")
    os.makedirs(datasets_subdir, exist_ok=True)
    bootstrap_datasets_savename = os.path.join(datasets_subdir,
            f"bootstrap_datasets_trial_{bootstrap_trial_i}.pkl")
    with open(bootstrap_datasets_savename, "wb") as outfile:
        pickle.dump(bootstrap_datasets_dict, outfile)

    # Check that run_bootstrap_trial from HyperparamSearch gives you the same solution.
    HS_spec = create_HS_spec([SA_spec.frac_data_in_safety])
    HS = HyperparamSearch(SA_spec, HS_spec, "test")
    HS_run = HS.run_bootstrap_trial(bootstrap_trial_i, 
            est_frac_data_in_safety=est_frac_data_in_safety,
            bootstrap_savedir=bootstrap_savedir)
    results_subdir = os.path.join(bootstrap_savedir,
            f"future_safety_frac_{est_frac_data_in_safety:.2f}", "bootstrap_results")
    results_savename = os.path.join(results_subdir, f"trial_{bootstrap_trial_i}_result.pkl")
    HS_result_dict = load_pickle(results_savename)
    HS_passed_safety = HS_result_dict["passed_safety"]
    HS_solution = HS_result_dict["solution"]
    assert(HS_run is True)
    assert(SA_passed_safety == HS_passed_safety)
    assert(np.allclose(SA_solution, HS_solution, atol=1e-2))

    # Check that will not re-run if trial already run, and correctly return False.
    HS_run = HS.run_bootstrap_trial(bootstrap_trial_i,
            est_frac_data_in_safety=est_frac_data_in_safety,
            bootstrap_savedir=bootstrap_savedir)
    assert(HS_run is False)

    print("test_run_bootstrap_trial passed")


def test_aggregate_est_prob_pass():
    # Create synthetic results.
    est_frac_data_in_safety = 0.4
    n_bootstrap_trials = 50
    bootstrap_savedir = "test/test_aggregate_est_prob_pass"
    if os.path.exists(bootstrap_savedir): shutil.rmtree(bootstrap_savedir)
    os.makedirs(bootstrap_savedir, exist_ok=True)
    all_trial_i, all_trial_pass, all_trial_solutions = create_test_aggregate_est_prob_pass_files(
            bootstrap_savedir, est_frac_data_in_safety)

    SA_spec = create_test_SA_spec()
    HS_spec = create_HS_spec([SA_spec.frac_data_in_safety], n_bootstrap_trials=n_bootstrap_trials)
    HS = HyperparamSearch(SA_spec, HS_spec, "test")

    # Aggregate the results, when the indices are not all there because some imcomplete
    # trials.
    est_prob_pass, lower_bound, upper_bound, results_df = HS.aggregate_est_prob_pass(
            est_frac_data_in_safety,  n_bootstrap_trials, bootstrap_savedir)
    assert(np.allclose(est_prob_pass, np.mean(all_trial_pass)))
    assert(np.array_equal(results_df["passed_safety"].values, all_trial_pass))
    assert(np.array_equal(results_df["solution"].values, all_trial_solutions))
    assert(lower_bound is None)
    assert(upper_bound is None)

    # TODO: Update tests to compute bounds.

    print("test_aggregate_est_prob_pass passed")


def test_get_est_prob_pass():

    # ============== Test with est_frac_data_in_safety same as true ==============
    bootstrap_savedir = "test/test_get_est_prob_pass/iter"
    if os.path.exists(bootstrap_savedir): shutil.rmtree(bootstrap_savedir)

    n_bootstrap_trials = 100
    true_frac_data_in_safety = 0.6
    num_points_dataset = 2000 # Need quite large dataset to get bootstrap samplines.
    SA_spec = create_test_SA_spec(
            num_points=num_points_dataset,
            frac_data_in_safety=true_frac_data_in_safety)
    est_frac_data_in_safety = SA_spec.frac_data_in_safety

    # Use HS to get split dataset.
    HS_spec = create_HS_spec([SA_spec.frac_data_in_safety])
    HS = HyperparamSearch(SA_spec, HS_spec, bootstrap_savedir)
    candidate_dataset, safety_dataset = HS.create_dataset(
            SA_spec.dataset, SA_spec.frac_data_in_safety, shuffle=False)
    n_candidate, n_safety = candidate_dataset.num_datapoints, safety_dataset.num_datapoints

    # Compute estimated probability of passing.
    (est_prob_pass, lower_bound, upper_bound, results_df, iter_time, ran_new_bs_trials) \
            = HS.get_est_prob_pass(
                    est_frac_data_in_safety, candidate_dataset, n_candidate, n_safety,
                    bootstrap_savedir, n_bootstrap_trials, n_workers=1)
    assert(0.95 <= est_prob_pass <= 1.0) # In reality is 1.
    assert(ran_new_bs_trials is True)
    assert(lower_bound is None)
    assert(upper_bound is None)

    # ======= Run test with same conditions as above, but with multiple workers ========
    bootstrap_savedir_parallel = "test/test_get_est_prob_pass/parallel"
    if os.path.exists(bootstrap_savedir_parallel): shutil.rmtree(bootstrap_savedir_parallel)

    n_workers = 20
    est_prob_pass_parallel, _, _, results_df_parallel, parallel_time, ran_new_bs_trials = HS.get_est_prob_pass(
            est_frac_data_in_safety, candidate_dataset, n_candidate, n_safety,
            bootstrap_savedir_parallel, n_bootstrap_trials, n_workers=n_workers)

    assert(ran_new_bs_trials)
    speedup = iter_time / parallel_time
    print(f"    With {n_workers} workers, {speedup:.2f}x speedup for get_est_prob_pass!")
    assert(np.allclose(est_prob_pass, est_prob_pass_parallel))


    # ==================== Re-run with same amount of trials ===========================

    # First, check that if re-run with current amount of trials, that we correctly do not
    # run any additional trials.
    rerun_est_prob_pass, _, _, rerun_results_df, _, ran_new_bs_trials = HS.get_est_prob_pass(
            est_frac_data_in_safety, candidate_dataset, n_candidate, n_safety,
            bootstrap_savedir_parallel, n_bootstrap_trials, n_workers=n_workers)

    # Check that new trials are not actually run.
    assert(ran_new_bs_trials is False)
    assert(np.allclose(rerun_est_prob_pass, est_prob_pass))

    # Check that no new trials are run, by seeing if trial information in dataframe similar.
    assert(np.array_equal(rerun_results_df["passed_safety"], 
        results_df_parallel["passed_safety"]))
    assert(np.allclose(np.stack(rerun_results_df["solution"].values), 
        np.stack(results_df_parallel["solution"].values)))

    # ====================== Re-run with additional trials =============================

    # Now let's run an additonal 20 trials on top.
    new_n_bootstrap_trials = 120
    add_est_prob_pass, _, _, add_results_df, _, ran_new_bs_trials = HS.get_est_prob_pass(
            est_frac_data_in_safety, candidate_dataset, n_candidate, n_safety,
            bootstrap_savedir_parallel, new_n_bootstrap_trials, n_workers=n_workers)

    # Check that new trials were actually run.
    assert(ran_new_bs_trials)
    assert(add_results_df.shape == (new_n_bootstrap_trials, 2))

    # Check that the initial trial information are untouched.
    assert(np.array_equal(add_results_df["passed_safety"][:n_bootstrap_trials], 
        results_df_parallel["passed_safety"][:n_bootstrap_trials]))
    assert(np.allclose(np.stack(add_results_df["solution"][:n_bootstrap_trials].values, axis=0),
        np.stack(results_df_parallel["solution"][:n_bootstrap_trials].values, axis=0)))

    print("test_get_est_prob_pass passed")


def test_get_all_greater_est_prob_pass():
    # ================================== Normal Test ===================================
    results_dir = "test/test_get_all_greater_est_prob_pass"
    if os.path.exists(results_dir): shutil.rmtree(results_dir)

    # Check that all the outputs are being written out.
    num_points_dataset = 2000 # Need quite large dataset to get bootstrap samplines.
    all_frac_data_in_safety = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    n_bootstrap_trials = 300
    n_workers = 60
    SA_spec = create_test_SA_spec(num_points=num_points_dataset)
    HS_spec = create_HS_spec(all_frac_data_in_safety, n_bootstrap_trials=n_bootstrap_trials,
            n_bootstrap_workers=n_workers, use_bs_pools=False)

    # Use HS to get split dataset.
    HS = HyperparamSearch(SA_spec, HS_spec, results_dir)
    all_estimates, elapsed_time = HS.get_all_greater_est_prob_pass(
            n_bootstrap_trials, n_workers=n_workers)
    print("elapsed time:", elapsed_time)

    print(all_estimates)

    # so little data in the candidate dataset, that it is probably not enough to do the
    # bootstrap sampling... and this identify that 0.9 and 0.8 are good enoguh estimators.
    expected_estimates = {
            0.9: { # Most data in safety, not enough to estimate well..
                0.9: 0.73, 
                0.8: 0.87, 
                0.7: 0.8533333333333334, 
                0.6: 0.83, 
                0.5: 0.7, 
                0.4: 0.67, 
                0.3: 0.59, 
                0.2: 0.47333333333333333, 
                0.1: 0.27666666666666667
                }, 
            0.8: {
                0.8: 0.97, 
                0.7: 0.9733333333333334, 
                0.6: 0.9733333333333334, 
                0.5: 0.95, 
                0.4: 0.94, 
                0.3: 0.8433333333333334, 
                0.2: 0.6966666666666667, 
                0.1: 0.3933333333333333
                }, 
            0.7: {
                0.7: 0.99, 
                0.6: 0.9866666666666667, 
                0.5: 0.97, 
                0.4: 0.9533333333333334, 
                0.3: 0.9033333333333333, 
                0.2: 0.81, 
                0.1: 0.4633333333333333
                }, 
            0.6: {
                0.6: 0.9933333333333333, 
                0.5: 0.9933333333333333, 
                0.4: 0.9766666666666667, 
                0.3: 0.9233333333333333, 
                0.2: 0.7733333333333333, 
                0.1: 0.48333333333333334
                }, 
            0.5: {
                0.5: 0.9866666666666667, 
                0.4: 0.97, 
                0.3: 0.8933333333333333, 
                0.2: 0.74, 
                0.1: 0.47333333333333333
                },
            0.4: {
                0.4: 0.98, 
                0.3: 0.9166666666666666, 
                0.2: 0.7766666666666666, 
                0.1: 0.47333333333333333
            }, 
            0.3: {
                0.3: 0.9466666666666667,
                0.2: 0.88,
                0.1: 0.5
            }, 
            0.2: {
                0.2: 0.89, 
                0.1: 0.52
            },
            0.1: {
                0.1: 0.5433333333333333
            }
        }

    for frac_data_in_safety in all_frac_data_in_safety:
        for frac_data_in_safety_prime in all_frac_data_in_safety:
            if frac_data_in_safety_prime > frac_data_in_safety: continue 

            print()
            print(frac_data_in_safety, frac_data_in_safety_prime)
            print(all_estimates[frac_data_in_safety][frac_data_in_safety_prime])
            print(expected_estimates[frac_data_in_safety][frac_data_in_safety_prime])
            assert(np.allclose(
                all_estimates[frac_data_in_safety][frac_data_in_safety_prime],
                expected_estimates[frac_data_in_safety][frac_data_in_safety_prime],
                atol=1e-2))

    # ============================ Test, not enough data ===============================
    results_dir = "test/test_get_all_greater_est_prob_pass"
    if os.path.exists(results_dir): shutil.rmtree(results_dir)

    # Not enough data... none of the ests should be run.
    num_points_dataset = 3 # Need quite large dataset to get bootstrap samplines.
    all_frac_data_in_safety = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    n_bootstrap_trials = 300
    n_workers = 31
    SA_spec = create_test_SA_spec(num_points=num_points_dataset)
    HS_spec = create_HS_spec(all_frac_data_in_safety, n_bootstrap_trials=n_bootstrap_trials,
            n_bootstrap_workers=n_workers)

    HS = HyperparamSearch(SA_spec, HS_spec, results_dir)
    all_estimates, _ = HS.get_all_greater_est_prob_pass(
            n_bootstrap_trials, n_workers=n_workers)
    for frac_data_in_safety in all_frac_data_in_safety:
        assert(all_estimates[frac_data_in_safety] == {}) # Nothing was estimated.

    print("test_get_all_greater_est_prob_pass passed")


def test_find_best_hyperparams():
    # ====================== Test if not enough data ===================================
    results_dir = "test/test_find_best_hyperparams"
    if os.path.exists(results_dir): shutil.rmtree(results_dir)

    # With this little data, should not even try to compute bootstrap estimates.
    num_points_dataset = 3 
    all_frac_data_in_safety = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    n_bootstrap_trials = 300
    n_workers = 60
    SA_spec = create_test_SA_spec(num_points=num_points_dataset)
    HS_spec = create_HS_spec(all_frac_data_in_safety, n_bootstrap_trials=n_bootstrap_trials,
            n_bootstrap_workers=n_workers)

    # Check that, passess through all the bootstrapping and selects to put most data in cs.
    HS = HyperparamSearch(SA_spec, HS_spec, results_dir)
    frac_data_in_safety, candidate_dataset, safety_dataset, ran_new_bs_trials = \
            HS.find_best_hyperparams(n_bootstrap_trials, n_workers=n_workers)
    assert(frac_data_in_safety == min(all_frac_data_in_safety))

    # ============================= Regular test =======================================
    np.random.seed(0)
    results_dir = "test/test_find_best_hyperparams"
    if os.path.exists(results_dir): shutil.rmtree(results_dir)

    # Check that all the outputs are being written out.
    num_points_dataset = 2000 # Need quite large dataset to get bootstrap samplines.
    all_frac_data_in_safety = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    SA_spec = create_test_SA_spec(num_points=num_points_dataset)
    HS_spec = create_HS_spec(all_frac_data_in_safety)

    # Use HS to get split dataset.
    n_bootstrap_trials = 300
    n_workers = 60
    HS = HyperparamSearch(SA_spec, HS_spec, results_dir)
    frac_data_in_safety, candidate_dataset, safety_dataset, ran_new_bs_trials = \
            HS.find_best_hyperparams(
                    n_bootstrap_trials, n_workers=n_workers)
    assert(frac_data_in_safety == 0.6)

    # Check that the correct folders for the bootstrap trials created with respect to fact
    # that stopped at rho=0.6.
    frac_09_path = os.path.join(results_dir, "bootstrap_safety_frac_0.90")
    frac_08_path = os.path.join(results_dir, "bootstrap_safety_frac_0.80")
    frac_07_path = os.path.join(results_dir, "bootstrap_safety_frac_0.70")
    frac_06_path = os.path.join(results_dir, "bootstrap_safety_frac_0.60")
    frac_05_path = os.path.join(results_dir, "bootstrap_safety_frac_0.50")
    frac_04_path = os.path.join(results_dir, "bootstrap_safety_frac_0.40")
    frac_03_path = os.path.join(results_dir, "bootstrap_safety_frac_0.30")
    frac_02_path = os.path.join(results_dir, "bootstrap_safety_frac_0.20")
    frac_01_path = os.path.join(results_dir, "bootstrap_safety_frac_0.10")
    existing_rho_paths = [frac_09_path, frac_08_path, frac_07_path, frac_06_path]
    assert(os.path.exists(frac_09_path))
    assert(os.path.exists(frac_08_path))
    assert(os.path.exists(frac_07_path))
    assert(os.path.exists(frac_06_path))
    assert(os.path.exists(frac_05_path) is False)
    assert(os.path.exists(frac_04_path) is False)
    assert(os.path.exists(frac_03_path) is False)
    assert(os.path.exists(frac_02_path) is False)
    assert(os.path.exists(frac_01_path) is False)

    # Check that the correct rho' are computed for each rho.
    assert(sorted(os.listdir(frac_09_path)) == sorted([
        "future_safety_frac_0.90",
        "future_safety_frac_0.80",
    ]))
    assert(sorted(os.listdir(frac_08_path)) == sorted([
        "future_safety_frac_0.80",
        "future_safety_frac_0.70",
    ]))
    assert(sorted(os.listdir(frac_07_path)) == sorted([
        "future_safety_frac_0.70",
        "future_safety_frac_0.60",
    ]))
    assert(sorted(os.listdir(frac_06_path)) == sorted([
        "future_safety_frac_0.60",
        "future_safety_frac_0.50",
        "future_safety_frac_0.40",
        "future_safety_frac_0.30",
        "future_safety_frac_0.20",
        "future_safety_frac_0.10",
    ]))

    # Check that all the results are created.
    all_result_filenames = sorted(["trial_%d_result.pkl" % trial for trial in range(n_bootstrap_trials)])
    for rho_path in existing_rho_paths:
        for rho_prime_path in os.listdir(rho_path):
            assert(sorted(os.listdir(os.path.join(rho_path, rho_prime_path, 
                "bootstrap_results"))) == all_result_filenames)

    # Check that the all_bootstrap_est.csv contains all the correct estimates.
    expected_df = pd.DataFrame([
        {"frac_data_in_safety": 0.9, "est_frac_data_in_safety": 0.9, 
            "est_prob_pass": 0.73},
        {"frac_data_in_safety": 0.9, "est_frac_data_in_safety": 0.8, 
            "est_prob_pass": 0.87},
        {"frac_data_in_safety": 0.8, "est_frac_data_in_safety": 0.8, 
            "est_prob_pass": 0.96333333},
        {"frac_data_in_safety": 0.8, "est_frac_data_in_safety": 0.7, 
            "est_prob_pass": 0.97},
        {"frac_data_in_safety": 0.7, "est_frac_data_in_safety": 0.7, 
            "est_prob_pass": 0.98666667},
        {"frac_data_in_safety": 0.7, "est_frac_data_in_safety": 0.6, 
            "est_prob_pass": 0.98666667},
        {"frac_data_in_safety": 0.6, "est_frac_data_in_safety": 0.6, 
            "est_prob_pass": 1.0},
        {"frac_data_in_safety": 0.6, "est_frac_data_in_safety": 0.5, 
            "est_prob_pass": 0.99333333},
        {"frac_data_in_safety": 0.6, "est_frac_data_in_safety": 0.4, 
            "est_prob_pass":  0.97666667},
        {"frac_data_in_safety": 0.6, "est_frac_data_in_safety": 0.3, 
            "est_prob_pass": 0.9166666},
        {"frac_data_in_safety": 0.6, "est_frac_data_in_safety": 0.2, 
            "est_prob_pass": 0.8},
        {"frac_data_in_safety": 0.6, "est_frac_data_in_safety": 0.1, 
            "est_prob_pass": 0.43333333}
    ])
    all_est_df = pd.read_csv(os.path.join(results_dir, "all_bootstrap_est.csv"))
    assert(np.allclose(expected_df["frac_data_in_safety"], all_est_df["frac_data_in_safety"]))
    assert(np.allclose(expected_df["est_frac_data_in_safety"], 
        all_est_df["est_frac_data_in_safety"]))
    assert(np.allclose(expected_df["est_prob_pass"], all_est_df["est_prob_pass"]))

    # ====================== Run again, without new trials ================================
    new_frac_data_in_safety, new_candidate_dataset, new_safety_dataset, ran_new_bs_trials = \
            HS.find_best_hyperparams(
                    n_bootstrap_trials, n_workers=n_workers)
    assert(ran_new_bs_trials is False)
    assert(new_frac_data_in_safety == frac_data_in_safety)

    # ========================== Adding new trials ====================================
    new_n_bootstrap_trials = 350
    frac_data_in_safety, candidate_dataset, safety_dataset, ran_new_bs_trials = \
            HS.find_best_hyperparams(
                    new_n_bootstrap_trials, n_workers=n_workers)
    assert(ran_new_bs_trials)

    print("test_find_best_hyperparams passed")


def test_size_integration_test():
    # TODO: Not sure how to convert this into an actual test... but can use this to see if things are ever changing just through print statements.... just my own sanity.

    np.random.seed(0)
    results_dir = "test/size_integration_test"
    if os.path.exists(results_dir): shutil.rmtree(results_dir) 

    # Check that all the outputs are being written out.
    num_points_dataset = 1000 # Need quite large dataset to get bootstrap samplines.
    n_bootstrap_trials = 1
    n_workers = 1
    all_frac_data_in_safety = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    SA_spec = create_test_SA_spec(num_points=num_points_dataset)
    HS_spec = create_HS_spec(all_frac_data_in_safety, n_bootstrap_trials=n_bootstrap_trials,
            n_bootstrap_workers=n_workers)

    # Use HS to get split dataset.
    HS = HyperparamSearch(SA_spec, HS_spec, results_dir)
    frac_data_in_safety, candidate_dataset, safety_dataset, ran_new_bs_trials = \
            HS.find_best_hyperparams(
                    n_bootstrap_trials, n_workers=n_workers)


if __name__ == "__main__":
    test_frac_sort()
    test_get_safety_size()
    test_candidate_safety_split()
    test_create_dataset()
    test_candidate_safety_combine()
    test_bootstrap_sample_dataset()
    test_create_shuffled_dataset()
    test_get_bootstrap_dataset_size()
    test_generate_all_bootstrap_datasets()
    test_SA_datasplit_loading()
    test_run_bootstrap_trial()
    test_aggregate_est_prob_pass()
    test_get_est_prob_pass()
    test_get_all_greater_est_prob_pass() 
    test_find_best_hyperparams()
    test_size_integration_test()

    print("all tests passed")

