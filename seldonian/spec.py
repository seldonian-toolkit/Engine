""" Module for building the specification object needed to run Seldonian algorithms """
import os
import importlib

from seldonian.utils.io_utils import save_pickle
from seldonian.dataset import load_supervised_metadata
from seldonian.models.models import *
from seldonian.models import objectives
from seldonian.parse_tree.parse_tree import make_parse_trees_from_constraints


class Spec(object):
    """Base class for specification object required to
    run the Seldonian algorithm

    :param dataset: The dataset object containing safety data
    :type dataset: :py:class:`.DataSet` object
    :param model: The :py:class:`.SeldonianModel` object
    :param frac_data_in_safety: Fraction of data used in safety test.
            The remaining fraction will be used in candidate selection
    :type frac_data_in_safety: float
    :param primary_objective: The objective function that would
            be solely optimized in the absence of behavioral constraints,
            i.e., the loss function
    :type primary_objective: function or class method
    :param initial_solution_fn: Function to provide
            initial model weights in candidate selection
    :type initial_solution_fn: function
    :param parse_trees: List of parse tree objects containing the
                    behavioral constraints
    :type parse_trees: List(:py:class:`.ParseTree` objects)
    :param base_node_bound_method_dict: A dictionary specifying the
            bounding method to use for each base node
    :type base_node_bound_method_dict: dict, defaults to {}
    :param use_builtin_primary_gradient_fn: Whether to use the built-in
            function for the gradient of the primary objective,
            if one exists. If False, uses autograd
    :type use_builtin_primary_gradient_fn: bool, defaults to True
    :param custom_primary_gradient_fn: A function for computing
            the gradient of the primary objective. If None,
            falls back on builtin function or autograd
    :type custom_primary_gradient_fn: function, defaults to None
    :param optimization_technique: The method for optimization during
            candidate selection. E.g. 'gradient_descent', 'barrier_function'
    :type optimization_technique: str, defaults to 'gradient_descent'
    :param optimizer: The string name of the optimizer used
            during candidate selection
    :type optimizer: str, defaults to 'adam'
    :param optimization_hyperparams: Hyperparameters for
            optimization during candidate selection. See :ref:`candidate_selection`.
    :type optimization_hyperparams: dict
    :param regularization_hyperparams: Hyperparameters for
            regularization during candidate selection. See :ref:`candidate_selection`.
    :type regularization_hyperparams: dict
    """

    def __init__(
        self,
        dataset,
        model,
        frac_data_in_safety,
        primary_objective,
        initial_solution_fn,
        parse_trees,
        base_node_bound_method_dict={},
        use_builtin_primary_gradient_fn=True,
        custom_primary_gradient_fn=None,
        optimization_technique="gradient_descent",
        optimizer="adam",
        optimization_hyperparams={
            "lambda_init": 0.5,
            "alpha_theta": 0.005,
            "alpha_lamb": 0.005,
            "beta_velocity": 0.9,
            "beta_rmsprop": 0.95,
            "num_iters": 200,
            "gradient_library": "autograd",
            "hyper_search": None,
            "verbose": False,
        },
        regularization_hyperparams={},
        batch_size_safety=None,
        candidate_dataset=None,
        safety_dataset=None,
        verbose=False,
    ):
        self.dataset = dataset
        self.candidate_dataset = candidate_dataset
        self.safety_dataset = safety_dataset
        self.model = model
        self.frac_data_in_safety = frac_data_in_safety
        self.primary_objective = primary_objective
        self.initial_solution_fn = initial_solution_fn
        self.use_builtin_primary_gradient_fn = use_builtin_primary_gradient_fn
        self.custom_primary_gradient_fn = custom_primary_gradient_fn
        self.parse_trees = validate_parse_trees(parse_trees)
        self.base_node_bound_method_dict = base_node_bound_method_dict
        self.optimization_technique = optimization_technique
        self.optimizer = optimizer
        self.optimization_hyperparams = optimization_hyperparams
        self.regularization_hyperparams = regularization_hyperparams
        self.batch_size_safety = batch_size_safety
        self.verbose = verbose


class SupervisedSpec(Spec):
    """Specification object for running Supervised learning
    Seldonian algorithms

    :param dataset: The dataset object containing safety data
    :type dataset: :py:class:`.DataSet` object
    :param model: The SeldonianModel object
    :param parse_trees: List of parse tree objects containing the
                    behavioral constraints
    :param sub_regime: "classification" or "regression"
    :param frac_data_in_safety: Fraction of data used in safety test.
            The remaining fraction will be used in candidate selection
    :type frac_data_in_safety: float
    :param primary_objective: The objective function that would
            be solely optimized in the absence of behavioral constraints,
            i.e. the loss function
    :param initial_solution_fn: Function to provide
            initial model weights in candidate selection
    :param base_node_bound_method_dict: A dictionary specifying the
            bounding method to use for each base node
    :type base_node_bound_method_dict: dict, defaults to {}
    :param use_builtin_primary_gradient_fn: Whether to use the built-in
            function for the gradient of the primary objective,
            if one exists. If False, uses autograd
    :type use_builtin_primary_gradient_fn: bool, defaults to True
    :param custom_primary_gradient_fn: A function for computing
            the gradient of the primary objective. If None,
            falls back on builtin function or autograd
    :type custom_primary_gradient_fn: function, defaults to None
    :param optimization_technique: The method for optimization during
            candidate selection. E.g. 'gradient_descent', 'barrier_function'
    :type optimization_technique: str, defaults to 'gradient_descent'
    :param optimizer: The string name of the optimizer used
            during candidate selection
    :type optimizer: str, defaults to 'adam'
    :param optimization_hyperparams: Hyperparameters for
            optimization during candidate selection. See :ref:`candidate_selection`.
    :type optimization_hyperparams: dict
    :param regularization_hyperparams: Hyperparameters for
            regularization during candidate selection. See :ref:`candidate_selection`.
    :type regularization_hyperparams: dict
    """

    def __init__(
        self,
        dataset,
        model,
        parse_trees,
        sub_regime,
        frac_data_in_safety=0.6,
        primary_objective=None,
        initial_solution_fn=None,
        base_node_bound_method_dict={},
        use_builtin_primary_gradient_fn=True,
        custom_primary_gradient_fn=None,
        optimization_technique="gradient_descent",
        optimizer="adam",
        optimization_hyperparams={
            "lambda_init": 0.5,
            "alpha_theta": 0.005,
            "alpha_lamb": 0.005,
            "beta_velocity": 0.9,
            "beta_rmsprop": 0.95,
            "num_iters": 200,
            "gradient_library": "autograd",
            "use_batches": False,
            "hyper_search": None,
            "verbose": False,
        },
        regularization_hyperparams={},
        batch_size_safety=None,
        candidate_dataset=None,
        safety_dataset=None,
        verbose=False,
    ):
        super().__init__(
            dataset=dataset,
            model=model,
            parse_trees=parse_trees,
            primary_objective=primary_objective,
            initial_solution_fn=initial_solution_fn,
            frac_data_in_safety=frac_data_in_safety,
            base_node_bound_method_dict=base_node_bound_method_dict,
            use_builtin_primary_gradient_fn=use_builtin_primary_gradient_fn,
            custom_primary_gradient_fn=custom_primary_gradient_fn,
            optimization_technique=optimization_technique,
            optimizer=optimizer,
            optimization_hyperparams=optimization_hyperparams,
            regularization_hyperparams=regularization_hyperparams,
            batch_size_safety=batch_size_safety,
            candidate_dataset=candidate_dataset,
            safety_dataset=safety_dataset,
            verbose=verbose,
        )
        self.sub_regime = sub_regime


class RLSpec(Spec):
    """Specification object for running RL Seldonian algorithms

    :param dataset: The dataset object containing safety data
    :type dataset: :py:class:`.DataSet` object

    :param model: The :py:class:`.RL_Model` object

    :param parse_trees: List of parse tree objects containing the
                    behavioral constraints
    :type parse_trees: List(:py:class:`.ParseTree` objects)

    :param frac_data_in_safety: Fraction of data used in safety test.
            The remaining fraction will be used in candidate selection
    :type frac_data_in_safety: float

    :param primary_objective: The objective function that would
            be solely optimized in the absence of behavioral constraints,
            i.e. the loss function
    :type primary_objective: function or class method

    :param initial_solution_fn: Function to provide
            initial model weights in candidate selection
    :type initial_solution_fn: function

    :param base_node_bound_method_dict: A dictionary specifying the
            bounding method to use for each base node
    :type base_node_bound_method_dict: dict, defaults to {}

    :param use_builtin_primary_gradient_fn: Whether to use the built-in
            function for the gradient of the primary objective,
            if one exists. If False, uses autograd
    :type use_builtin_primary_gradient_fn: bool, defaults to True

    :param custom_primary_gradient_fn: A function for computing
            the gradient of the primary objective. If None,
            falls back on builtin function or autograd
    :type custom_primary_gradient_fn: function, defaults to None

    :param optimization_technique: The method for optimization during
            candidate selection. E.g. 'gradient_descent', 'barrier_function'
    :type optimization_technique: str, defaults to 'gradient_descent'

    :param optimizer: The string name of the optimizer used
            during candidate selection
    :type optimizer: str, defaults to 'adam'

    :param optimization_hyperparams: Hyperparameters for
            optimization during candidate selection. See
            :ref:`candidate_selection`.
    :type optimization_hyperparams: dict

    :param regularization_hyperparams: Hyperparameters for
            regularization during candidate selection. See
            :ref:`candidate_selection`.
    :type regularization_hyperparams: dict
    """

    def __init__(
        self,
        dataset,
        model,
        parse_trees,
        frac_data_in_safety=0.6,
        primary_objective=None,
        initial_solution_fn=None,
        base_node_bound_method_dict={},
        use_builtin_primary_gradient_fn=True,
        custom_primary_gradient_fn=None,
        optimization_technique="gradient_descent",
        optimizer="adam",
        optimization_hyperparams={
            "lambda_init": 0.5,
            "alpha_theta": 0.005,
            "alpha_lamb": 0.005,
            "beta_velocity": 0.9,
            "beta_rmsprop": 0.95,
            "num_iters": 200,
            "use_batches": False,
            "gradient_library": "autograd",
            "hyper_search": None,
            "verbose": False,
        },
        regularization_hyperparams={},
        batch_size_safety=None,
        candidate_dataset=None,
        safety_dataset=None,
        verbose=False,
    ):
        super().__init__(
            dataset=dataset,
            model=model,
            frac_data_in_safety=frac_data_in_safety,
            primary_objective=primary_objective,
            initial_solution_fn=initial_solution_fn,
            parse_trees=parse_trees,
            base_node_bound_method_dict=base_node_bound_method_dict,
            use_builtin_primary_gradient_fn=use_builtin_primary_gradient_fn,
            custom_primary_gradient_fn=custom_primary_gradient_fn,
            optimization_technique=optimization_technique,
            optimizer=optimizer,
            optimization_hyperparams=optimization_hyperparams,
            regularization_hyperparams=regularization_hyperparams,
            batch_size_safety=batch_size_safety,
            candidate_dataset=candidate_dataset,
            safety_dataset=safety_dataset,
            verbose=verbose,
        )

class ProposedHyperparameterSelectionSpec(object):
    """Class for the specification object for selecting hyperparameters

    :param hyper_schema: A hyperparameter schema specifying which hyperparameters to tune
        and the values to sweep over.
    :type hyper_schema: seldonian.hyperparam_search.HyperSchema
    :param n_bootstrap_trials: The number of bootstrap trials to run
    :type n_bootstrap_trials: int
    :param n_bootstrap_workers: The number of workers (parallel processes) to use when running bootstrap trials
    :type n_bootstrap_workers: int
    :param use_bs_pools: Whether to use sampling pools during bootstrapping
    :type use_bs_pools: bool
    :param confidence_interval_type: "ttest" or "clopper-pearson"
    :type confidence_interval_type: str
    """

    def __init__(
        self,
        hyper_schema,
        n_bootstrap_trials,
        n_bootstrap_workers,
        use_bs_pools,
        confidence_interval_type=None
    ):
        self.hyper_schema = hyper_schema
        self.n_bootstrap_trials = n_bootstrap_trials
        self.n_bootstrap_workers = n_bootstrap_workers
        self.use_bs_pools = use_bs_pools,
        self.confidence_interval_type = confidence_interval_type

class HyperparameterSelectionSpec(object):
    """Base class for specification object for selecting Hyperparameters

    :param n_bootstrap_trials: The number of bootstrap trials to use when estimating
        Hyperparameters to use
    :type n_bootstrap_trials: int
    :param all_frac_data_in_safety: List of all values of fraction of data used in 
        safety to select over 
    :type frac_data_in_safety: List(float)
    """

    def __init__(
        self,
        n_bootstrap_trials,
        all_frac_data_in_safety,
        n_bootstrap_workers,
        use_bs_pools,
        confidence_interval_type=None
    ):
        self.n_bootstrap_trials = n_bootstrap_trials
        self.all_frac_data_in_safety = all_frac_data_in_safety
        self.n_bootstrap_workers = n_bootstrap_workers
        self.use_bs_pools = use_bs_pools,
        self.confidence_interval_type = confidence_interval_type

def createSimpleSupervisedSpec(
    dataset,
    constraint_strs,
    deltas,
    sub_regime='regression',
    sensitive_col_names=[],
    frac_data_in_safety=0.6,
    save=True,
    save_dir=".",
    verbose=False,
):
    """Convenience function for creating SupervisedSpec object
    without a metadata file.
    
    Saves spec.pkl file in save_dir

    :param dataset: The dataset object containing data and metadata
    :type dataset: :py:class:`.DataSet`
    :param constraint_strs: Constraint strings
    :type constraint_strs: List(str)
    :param deltas: Confidence thresholds
    :type deltas: List(float)
    :param sub_regime: "classification" or "regression"
    :param sensitive_col_names: List of sensitive column names
    :type sensitive_col_names: List(str)
    :param frac_data_in_safety: Fraction of data used in safety test.
            The remaining fraction will be used in candidate selection
    :type frac_data_in_safety: float
    :param save: Boolean flag determining whether to save to a file
    :param save_dir: Directory where to save the spec.pkl file
    :type save_dir: str
    :param verbose: Boolean glag to control verbosity
    """

    assert dataset.regime == "supervised_learning"

    if sub_regime == "regression":
        model = LinearRegressionModel()
        primary_objective = objectives.Mean_Squared_Error
    elif sub_regime in ["classification", "binary_classification"]:
        model = BinaryLogisticRegressionModel()
        primary_objective = objectives.binary_logistic_loss
    elif sub_regime == "multiclass_classification":
        model = MultiClassLogisticRegressionModel()
        primary_objective = objectives.multiclass_logistic_loss

    parse_trees = make_parse_trees_from_constraints(
        constraint_strs,
        deltas,
        regime="supervised_learning",
        sub_regime=sub_regime,
        columns=sensitive_col_names,
        delta_weight_method="equal",
    )

    # Save spec object, using defaults where necessary
    spec = SupervisedSpec(
        dataset=dataset,
        model=model,
        frac_data_in_safety=frac_data_in_safety,
        primary_objective=primary_objective,
        use_builtin_primary_gradient_fn=True,
        parse_trees=parse_trees,
        sub_regime=sub_regime,
        initial_solution_fn=None,
        optimization_technique="gradient_descent",
        optimizer="adam",
        optimization_hyperparams={
            "lambda_init": 0.5,
            "alpha_theta": 0.005,
            "alpha_lamb": 0.005,
            "beta_velocity": 0.9,
            "beta_rmsprop": 0.95,
            "num_iters": 200,
            "gradient_library": "autograd",
            "use_batches": False,
            "hyper_search": None,
            "verbose": True,
        },
        regularization_hyperparams={},
        batch_size_safety=None,
        verbose=verbose,
    )

    spec_save_name = os.path.join(save_dir, "spec.pkl")
    if save:
        save_pickle(spec_save_name, spec, verbose=verbose)
    return spec


def createSupervisedSpec(
    dataset,
    metadata_pth,
    constraint_strs,
    deltas,
    frac_data_in_safety=0.6,
    save=True,
    save_dir=".",
    verbose=False,
):
    """Convenience function for creating SupervisedSpec object.
    Uses many defaults which can later be changed by updating
    the spec object.

    Saves spec.pkl file in save_dir

    :param dataset: The dataset object containing data and metadata
    :type dataset: :py:class:`.DataSet`
    :param metadata_pth: Path to metadata file
    :type metadata_pth: str
    :param constraint_strs: Constraint strings
    :type constraint_strs: List(str)
    :param deltas: Confidence thresholds
    :type deltas: List(float)
    :param save: Boolean flag determining whether to save to a file
    :param save_dir: Directory where to save the spec.pkl file
    :type save_dir: str
    :param verbose: Boolean glag to control verbosity
    """
    # Load metadata
    meta = load_supervised_metadata(metadata_pth)

    assert meta.regime == "supervised_learning"

    if meta.sub_regime == "regression":
        model = LinearRegressionModel()
        primary_objective = objectives.Mean_Squared_Error
    elif meta.sub_regime in ["classification", "binary_classification"]:
        model = BinaryLogisticRegressionModel()
        primary_objective = objectives.binary_logistic_loss
    elif meta.sub_regime == "multiclass_classification":
        model = MultiClassLogisticRegressionModel()
        primary_objective = objectives.multiclass_logistic_loss

    parse_trees = make_parse_trees_from_constraints(
        constraint_strs,
        deltas,
        regime="supervised_learning",
        sub_regime=meta.sub_regime,
        columns=meta.sensitive_col_names,
        delta_weight_method="equal",
    )

    # Save spec object, using defaults where necessary
    spec = SupervisedSpec(
        dataset=dataset,
        model=model,
        frac_data_in_safety=frac_data_in_safety,
        primary_objective=primary_objective,
        use_builtin_primary_gradient_fn=True,
        parse_trees=parse_trees,
        sub_regime=meta.sub_regime,
        initial_solution_fn=model.fit,
        optimization_technique="gradient_descent",
        optimizer="adam",
        optimization_hyperparams={
            "lambda_init": np.array([0.5]),
            "alpha_theta": 0.01,
            "alpha_lamb": 0.01,
            "beta_velocity": 0.9,
            "beta_rmsprop": 0.95,
            "use_batches": False,
            "num_iters": 1000,
            "gradient_library": "autograd",
            "hyper_search": None,
            "verbose": verbose,
        },
        verbose=verbose,
    )

    spec_save_name = os.path.join(save_dir, "spec.pkl")
    if save:
        save_pickle(spec_save_name, spec, verbose=verbose)
    return spec


def createRLSpec(
    dataset,
    policy,
    constraint_strs,
    deltas,
    env_kwargs={},
    frac_data_in_safety=0.6,
    initial_solution_fn=None,
    use_builtin_primary_gradient_fn=False,
    save=False,
    save_dir=".",
    verbose=False,
):
    """Convenience function for creating RLSpec object.
    Uses many defaults which can later be changed by updating
    the spec object.

    :type dataset: :py:class:`.DataSet`
    :type policy: :py:class:`.Policy`
    :param constraint_strs: List of constraint strings
    :param deltas: List of confidence thresholds
    :param env_kwargs: Kwargs passed to RL_model pertaining to environment,
            such as gamma, the discount factor
    :type env_kwargs: dict
    :param frac_data_in_safety: Fraction of data used in safety test.
            The remaining fraction will be used in candidate selection
    :type frac_data_in_safety: float
    :param initial_solution_fn: Function to provide
            initial model weights in candidate selection
    :type initial_solution_fn: function
    :param use_builtin_primary_gradient_fn: Whether to use the built-in
            function for the gradient of the primary objective,
            if one exists. If False, uses autograd
    :type use_builtin_primary_gradient_fn: bool, defaults to True
    :param save: Boolean flag determining whether to save to a file
    :param save_dir: Directory where to save the spec.pkl file
    :type save_dir: str
    :param verbose: Boolean glag to control verbosity
    """
    from seldonian.RL.RL_model import RL_model

    # Define primary objective
    primary_objective = objectives.IS_estimate

    # Create parse trees
    parse_trees = make_parse_trees_from_constraints(
        constraint_strs,
        deltas,
        columns=dataset.sensitive_col_names,
        regime="reinforcement_learning",
        sub_regime="all",
        delta_weight_method="equal",
    )

    model = RL_model(policy=policy, env_kwargs=env_kwargs)
    # Save spec object, using defaults where necessary
    spec = RLSpec(
        dataset=dataset,
        model=model,
        frac_data_in_safety=frac_data_in_safety,
        primary_objective=primary_objective,
        use_builtin_primary_gradient_fn=use_builtin_primary_gradient_fn,
        parse_trees=parse_trees,
        initial_solution_fn=initial_solution_fn,
        optimization_technique="gradient_descent",
        optimizer="adam",
        optimization_hyperparams={
            "lambda_init": 0.5,
            "alpha_theta": 0.005,
            "alpha_lamb": 0.005,
            "beta_velocity": 0.9,
            "beta_rmsprop": 0.95,
            "use_batches": False,
            "num_iters": 30,
            "hyper_search": None,
            "gradient_library": "autograd",
            "verbose": verbose,
        },
        regularization_hyperparams={},
        verbose=verbose,
    )

    if save:
        spec_save_name = os.path.join(save_dir, "spec.pkl")
        save_pickle(spec_save_name, spec, verbose=verbose)
    return spec


def validate_parse_trees(parse_trees):
    """Ensure that there are no duplicate
    constraints in a list of parse trees

    :param parse_trees: List of :py:class:`.ParseTree` objects
    """
    from collections import Counter

    constraint_strs = [pt.constraint_str for pt in parse_trees]
    ct_dict = Counter(constraint_strs)

    for constraint_str in ct_dict:
        if ct_dict[constraint_str] > 1:
            raise RuntimeError(
                f"The constraint: '{constraint_str}' "
                "appears more than once in the list of constraints. "
                "Duplicate constraints are not allowed."
            )
    return parse_trees
