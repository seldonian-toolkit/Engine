# TODO: Eventually, will move these tests into the library testing framework

import os
import shutil
import pickle
import tqdm
import autograd.numpy as np   # Thinly-wrapped version of Numpy
import random
import time
from seldonian.utils.io_utils import load_pickle
from seldonian.models.models import LinearRegressionModel
from seldonian.spec import SupervisedSpec
from seldonian.seldonian_algorithm import SeldonianAlgorithm
from seldonian.hyperparam_search import HyperparamSearch
from seldonian.utils.tutorial_utils import (
    make_synthetic_regression_dataset)
from seldonian.parse_tree.parse_tree import (
    make_parse_trees_from_constraints)


# ================================== Set-Up ============================================
def create_test_spec(num_points=1000, frac_data_in_safety=0.6):
    np.random.seed(0)

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
    spec = SupervisedSpec(
        dataset=dataset,
        model=model,
        parse_trees=parse_trees,
        sub_regime='regression',
        frac_data_in_safety=frac_data_in_safety,
    )

    return spec

def create_test_aggregate_est_prob_pass_files(bootstrap_savedir, est_frac_data_in_safety):
    result_dir = os.path.join(bootstrap_savedir, 
            f"future_safety_frac_{est_frac_data_in_safety:.2f}/bootstrap_results")
    os.makedirs(result_dir, exist_ok=True)
    trial_pass = [True, False, True, True, False, True, True, False, False, False, True,
                     False, True, False, True, False, True, False, True, False, True]
    trial_solutions = np.arange(len(trial_pass))

    # Write out the trial results.
    for bootstrap_trial_i in range(len(trial_pass)):
        bs_trial_savename = os.path.join(result_dir, f"trial_{bootstrap_trial_i}_result.pkl")
        with open(bs_trial_savename, "wb") as outfile:
            pickle.dump({"passed_safety": trial_pass[bootstrap_trial_i],
                         "solution": trial_solutions[bootstrap_trial_i]}, outfile)

    # Return the synthetic data for comparing.
    return trial_pass, trial_solutions


# ================================== Tests ============================================

def test_frac_sort():
    """
    Test that the safety frac are being traversed in the correct order.
    """
    spec = create_test_spec()
    all_frac_data_in_safety = [0.5, 0.1, 0.2, 0.9, 0.77]
    HS = HyperparamSearch(spec, all_frac_data_in_safety, "test")
    assert(np.allclose(HS.all_frac_data_in_safety, [0.9, 0.77, 0.5, 0.2, 0.1]))
    print("test_frac_sort passed")


def test_create_dataset():
    num_points = 100
    spec = create_test_spec(num_points)
    all_frac_data_in_safety = [0.9, 0.77, 0.5, 0.2, 0.1]
    HS = HyperparamSearch(spec, all_frac_data_in_safety, "test")

    for frac_data_in_safety in all_frac_data_in_safety:
        (candidate_dataset, safety_dataset, datasplit_info) = HS.create_dataset(
                HS.dataset, frac_data_in_safety, shuffle=False)

        # Test the sizeo of the created datasets is the same.
        assert(candidate_dataset.num_datapoints + safety_dataset.num_datapoints == num_points)
        assert(safety_dataset.num_datapoints == frac_data_in_safety * num_points)
        assert(candidate_dataset.num_datapoints == num_points - frac_data_in_safety * num_points)

    print("test_create_dataset passed")
    

def test_candidate_safety_combine():
    spec = create_test_spec()
    frac_data_in_safety = 0.7
    HS = HyperparamSearch(spec, [frac_data_in_safety], "test")
    candidate_dataset, safety_dataset, datasplit_info = HS.create_dataset(
            HS.dataset, frac_data_in_safety)

    # Re-join candidate and safety dataset and compare to the original dataset.
    combined_dataset = HS.candidate_safety_combine(candidate_dataset, safety_dataset)
    assert(np.allclose(spec.dataset.features, combined_dataset.features))
    assert(np.allclose(spec.dataset.labels, combined_dataset.labels))
    assert(np.allclose(spec.dataset.sensitive_attrs, combined_dataset.sensitive_attrs))
    assert(spec.dataset.num_datapoints == combined_dataset.num_datapoints)
    assert(spec.dataset.meta_information == combined_dataset.meta_information)

    print("test_candidate_safety_combine passed")



def test_bootstrap_sample_dataset():
    num_points = 100 
    spec = create_test_spec(num_points)
    curr_frac_data_in_safety = 0.4
    all_frac_data_in_safety = [0.9, 0.77, 0.5, 0.2, 0.1]
    HS = HyperparamSearch(spec, all_frac_data_in_safety, "test")

    # Generate bootstrapped dataset.
    savedir = "test/test_bootstrap_sample_dataset"
    os.makedirs(savedir, exist_ok=True)
    savename = os.path.join(savedir, "bootstrap_test.pkl")
    n_bootstrap_samples = 50 # Sample more data.

    # Bootstrap sampling from candidate_dataset.
    candidate_dataset, safety_dataset, _ = HS.create_dataset(HS.dataset, curr_frac_data_in_safety, shuffle=False)
    HS.bootstrap_sample_dataset(candidate_dataset, n_bootstrap_samples, savename)

    # Load bootstrapped dataset.
    bootstrap_dataset = load_pickle(savename)

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
    num_points = 100 
    spec = create_test_spec(num_points)
    all_frac_data_in_safety = [0.9, 0.77, 0.5, 0.2, 0.1]
    HS = HyperparamSearch(spec, all_frac_data_in_safety, "test")

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
    print("test_create_shuffled_dataset passed")


def test_get_bootstrap_dataset_size():
    num_points = 99
    spec = create_test_spec(num_points)
    all_frac_data_in_safety = [0.9, 0.77, 0.5, 0.2, 0.1]
    HS = HyperparamSearch(spec, all_frac_data_in_safety, "test")

    for frac_data_in_safety in all_frac_data_in_safety:
        n_bs_candidate, n_bs_safety  = HS.get_bootstrap_dataset_size(frac_data_in_safety)
        assert(n_bs_candidate + n_bs_safety == num_points)

    print("test_get_bootstrap_dataset_size passed")


def test_generate_all_bootstrap_datasets():
    """Test that bootstrapped datasets are generated for each trial.
    
    Note that all tests relating to checking the actual data in the generated bootstrapped  
        datasets is done in test_bootstrap_sample_dataset.
    """
    # Remove existing test data directory.
    test_dir = "test/test_generate_all_bootstrap_datasets"
    if os.path.exists(test_dir): shutil.rmtree(test_dir)

    num_points = 100 
    spec = create_test_spec(num_points)
    bs_all_frac_data_in_safety = [0.9, 0.77, 0.5, 0.2, 0.1]
    HS = HyperparamSearch(spec, bs_all_frac_data_in_safety, test_dir)

    curr_frac_data_in_safety = 0.4
    candidate_dataset, safety_dataset, _ = HS.create_dataset(
            HS.dataset, curr_frac_data_in_safety, shuffle=False)

    n_bootstrap_trials = 10
    n_bootstrap_candidate = 110
    n_bootstrap_safety = 50

    for frac_data_in_safety in bs_all_frac_data_in_safety:
        HS.generate_all_bootstrap_datasets(candidate_dataset, frac_data_in_safety,
                n_bootstrap_trials, n_bootstrap_candidate, n_bootstrap_safety, test_dir)

    # Test all fractions of data are created.
    all_data_dir_expected = sorted(["future_safety_frac_0.90", "future_safety_frac_0.77",
        "future_safety_frac_0.50", "future_safety_frac_0.20", "future_safety_frac_0.10"])
    all_data_dir = sorted(os.listdir(test_dir))
    assert(all_data_dir == all_data_dir_expected)

    # Test that the correct number of
    for frac_data_in_safety in bs_all_frac_data_in_safety:
        expected_candidate = [f"bootstrap_trial_{i}_candidate.pkl" for i 
            in range(n_bootstrap_trials)] 
        expected_safety = [f"bootstrap_trial_{i}_safety.pkl" for i
            in range(n_bootstrap_trials)]
        expected_datasplit = [f"bootstrap_trial_{i}_datasplit_info.pkl" for i
            in range(n_bootstrap_trials)]
        all_frac_data_dir_expected = sorted(expected_candidate + expected_safety +
                expected_datasplit)

        all_frac_data_dir = sorted(os.listdir(os.path.join(test_dir, 
            f"future_safety_frac_{frac_data_in_safety:.2f}", "bootstrap_datasets")))
        assert(all_frac_data_dir_expected == all_frac_data_dir)

    # TODO: Test that we skip if already created.

    print("test_generate_all_bootstrap_datasets passed")


def test_SA_datasplit_loading():
    """Check that when give SA pre-split candidate_dataset and safety_dataset in the spec,
        produces the same answer.
    """
    # Use the same spec for both.
    spec = create_test_spec()

    # Run the seldonian algorihtm using the spec object.
    SA = SeldonianAlgorithm(spec)
    passed_safety, solution = SA.run()

    # Use HS to get split dataset.
    HS = HyperparamSearch(spec, [spec.frac_data_in_safety], "test")
    candidate_dataset, safety_dataset, datasplit_info = HS.create_dataset(
            spec.dataset, spec.frac_data_in_safety, shuffle=False)

    # Create a spec that contains the split candidate and safety datasets again.
    spec_split = SupervisedSpec(
            dataset=spec.dataset,
            candidate_dataset=candidate_dataset,
            safety_dataset=safety_dataset,
            datasplit_info=datasplit_info,
            model=spec.model,
            parse_trees=spec.parse_trees,
            sub_regime='regression',
    )

    SA_split = SeldonianAlgorithm(spec_split)
    passed_safety_split, solution_split = SA_split.run()

    assert(passed_safety_split == passed_safety)
    assert(np.allclose(solution, solution_split, atol=1e-2))
    # Note the slight difference in answer above is from the different order that we call run.


def test_run_bootstrap_trial():
    # TODO: Is there anything else that I would like to type?
    """
    Test that loads (1) correct data by name, (2) successfully runs seldonian (compare with 
        true seldonian result), (3) saves data correctly in the right place.
    """
    bootstrap_savedir = "test/test_run_bootstrap_trial"
    if os.path.exists(bootstrap_savedir): shutil.rmtree(bootstrap_savedir)

    spec = create_test_spec()
    bootstrap_trial_i = 13
    est_frac_data_in_safety = spec.frac_data_in_safety

    # Run the seldonian algorihtm using the spec object.
    SA = SeldonianAlgorithm(spec)
    SA_passed_safety, SA_solution = SA.run() # This is what we compare the solution to.

    # Use HS to get split dataset.
    HS = HyperparamSearch(spec, [spec.frac_data_in_safety], bootstrap_savedir)
    candidate_dataset, safety_dataset, datasplit_info = HS.create_dataset(
            spec.dataset, spec.frac_data_in_safety, shuffle=False)

    # Create test data to use in this trial. Just store the candidate and safety dataset
    # form a normal split, so that we can compare with SA.
    datasets_subdir = os.path.join(bootstrap_savedir,
            f"future_safety_frac_{est_frac_data_in_safety:.2f}", "bootstrap_datasets")
    os.makedirs(datasets_subdir, exist_ok=True)
    candidate_dataset_savename = os.path.join(datasets_subdir,
            f"bootstrap_trial_{bootstrap_trial_i}_candidate.pkl")
    safety_dataset_savename = os.path.join(datasets_subdir,
            f"bootstrap_trial_{bootstrap_trial_i}_safety.pkl")
    datasplit_info_savename = os.path.join(datasets_subdir,
            f"bootstrap_trial_{bootstrap_trial_i}_datasplit_info.pkl")
    with open(candidate_dataset_savename, "wb") as outfile:
        pickle.dump(candidate_dataset, outfile)
    with open(safety_dataset_savename, "wb") as outfile:
        pickle.dump(safety_dataset, outfile)
    with open(datasplit_info_savename, "wb") as outfile:
        pickle.dump(datasplit_info, outfile)

    # Check that run_bootstrap_trial from HyperparamSearch gives you the same solution.
    HS = HyperparamSearch(spec, [spec.frac_data_in_safety], "test")
    HS_passed_safety, HS_solution = HS.run_bootstrap_trial(bootstrap_trial_i, 
            est_frac_data_in_safety=est_frac_data_in_safety,
            bootstrap_savedir=bootstrap_savedir)

    assert(SA_passed_safety == HS_passed_safety)
    assert(np.allclose(SA_solution, HS_solution, atol=1e-2))

    # Check that the results are written in the correct place.
    results_subdir = os.path.join(bootstrap_savedir,
            f"future_safety_frac_{est_frac_data_in_safety:.2f}", "bootstrap_results")
    os.makedirs(results_subdir, exist_ok=True)
    results_savename = os.path.join(results_subdir, f"trial_{bootstrap_trial_i}_result.pkl")
    result = load_pickle(results_savename)
    assert(HS_passed_safety == result["passed_safety"])
    assert(np.allclose(HS_solution, result["solution"]))

    # Check that will not re-run if trial already run.
    output = HS.run_bootstrap_trial(bootstrap_trial_i,
            est_frac_data_in_safety=est_frac_data_in_safety,
            bootstrap_savedir=bootstrap_savedir)
    assert(output is None)

    print("test_run_bootstrap_trial passed")


def test_aggregate_est_prob_pass():
    # Create synthetic results.
    est_frac_data_in_safety = 0.4
    bootstrap_savedir = "test/test_aggregate_est_prob_pass"
    if os.path.exists(bootstrap_savedir): shutil.rmtree(bootstrap_savedir)
    os.makedirs(bootstrap_savedir, exist_ok=True)
    trial_pass, trial_solutions = create_test_aggregate_est_prob_pass_files(bootstrap_savedir,
            est_frac_data_in_safety)

    spec = create_test_spec()
    HS = HyperparamSearch(spec, [spec.frac_data_in_safety], "test")

    # First aggregate all results, without giving the number of trials.
    est_prob_pass, results_df = HS.aggregate_est_prob_pass(
            est_frac_data_in_safety, bootstrap_savedir)
    assert(np.allclose(est_prob_pass, np.mean(trial_pass)))
    assert(np.array_equal(results_df["passed_safety"], trial_pass))
    assert(np.array_equal(results_df["solution"], trial_solutions))

    # Now give the true number of trials.
    est_prob_pass, results_dir = HS.aggregate_est_prob_pass(
            est_frac_data_in_safety, bootstrap_savedir, len(trial_pass))
    assert(np.allclose(est_prob_pass, np.mean(trial_pass)))
    assert(np.array_equal(results_df["passed_safety"], trial_pass))
    assert(np.array_equal(results_df["solution"], trial_solutions))

    # Now only give a subset of trials.
    N = 5
    est_prob_pass, results_df = HS.aggregate_est_prob_pass(
            est_frac_data_in_safety, bootstrap_savedir, N)
    assert(np.allclose(est_prob_pass, np.mean(trial_pass[:N])))
    assert(np.array_equal(results_df["passed_safety"], trial_pass[:N]))
    assert(np.array_equal(results_df["solution"], trial_solutions[:N]))

    print("test_aggregate_est_prob_pass passed")


def test_get_est_prob_pass():

    # ============== Test with est_frac_data_in_safety same as true ==============
    bootstrap_savedir = "test/test_get_est_prob_pass/iter"
    if os.path.exists(bootstrap_savedir): shutil.rmtree(bootstrap_savedir)

    n_bootstrap_trials = 100
    true_frac_data_in_safety = 0.6
    num_points_dataset = 2000 # Need quite large dataset to get bootstrap samplines.
    spec = create_test_spec(
            num_points=num_points_dataset,
            frac_data_in_safety=true_frac_data_in_safety)
    est_frac_data_in_safety = spec.frac_data_in_safety

    # Use HS to get split dataset.
    HS = HyperparamSearch(spec, [spec.frac_data_in_safety], bootstrap_savedir)
    candidate_dataset, safety_dataset, datasplit_info = HS.create_dataset(
            spec.dataset, spec.frac_data_in_safety, shuffle=False)
    n_candidate, n_safety = candidate_dataset.num_datapoints, safety_dataset.num_datapoints

    # Compute estimated probability of passing.
    est_prob_pass, results_df, iter_time = HS.get_est_prob_pass(
            est_frac_data_in_safety, candidate_dataset, n_candidate, n_safety,
            bootstrap_savedir, n_bootstrap_trials, n_workers=1)
    assert(0.95 <= est_prob_pass <= 1.0) # In reality is 1.

    # ======= Run test with same conditions as above, but with multiple workers ========
    bootstrap_savedir_parallel = "test/test_get_est_prob_pass/parallel"
    if os.path.exists(bootstrap_savedir_parallel): shutil.rmtree(bootstrap_savedir_parallel)

    n_workers = 20
    est_prob_pass_parallel, results_df_parallel, parallel_time = HS.get_est_prob_pass(
            est_frac_data_in_safety, candidate_dataset, n_candidate, n_safety,
            bootstrap_savedir_parallel, n_bootstrap_trials, n_workers=n_workers)
    speedup = iter_time / parallel_time
    print(f"    With {n_workers} workers, {speedup:.2f}x speedup for get_est_prob_pass!")

    assert(np.allclose(est_prob_pass, est_prob_pass_parallel))

    print("test_get_est_prob_pass passed")


def test_find_best_hyperparams():
    results_dir = "test/test_find_best_hyperparams"
    if os.path.exists(results_dir): shutil.rmtree(results_dir)

    # Check that all the outputs are being written out.
    num_points_dataset = 2000 # Need quite large dataset to get bootstrap samplines.
    all_frac_data_in_safety = [0.2, 0.4, 0.6, 0.8]
    spec = create_test_spec(num_points=num_points_dataset)

    # Use HS to get split dataset.
    n_bootstrap_trials = 100
    n_workers = 31
    HS = HyperparamSearch(spec, all_frac_data_in_safety, results_dir)
    frac_data_in_safety, candidate_dataset, safety_dataset = HS.find_best_hyperparams(
            n_bootstrap_trials, n_workers=n_workers)
    assert(frac_data_in_safety == 0.4)

    # TODO: Check that all the correct files are created.
    # TODO: Check from the files that nothing is over-run.
    # TODO: Check that the returned datasets have the correct proportions.
    print("test_find_best_hyperparams passed")


if __name__ == "__main__":
    test_frac_sort()
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
    test_find_best_hyperparams()


