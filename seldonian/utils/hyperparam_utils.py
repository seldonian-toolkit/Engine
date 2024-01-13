import autograd.numpy as np  # Thinly-wrapped version of Numpy
from seldonian.dataset import SupervisedDataSet
from seldonian.parse_tree.parse_tree import ParseTree


def create_shuffled_dataset(dataset):
    """Create new dataset containing the same data as the given original dataset,
        but with the data shuffled in a new order.

    :param dataset: a dataset object containing data
    :type dataset: :py:class:`.DataSet` object

    :return: shuffled_dataset, a dataset with same points in dataset, but shuffled.
    :rtype: :py:class:`.DataSet` object
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
        meta=dataset.meta,
    )

    return shuffled_dataset


def bootstrap_sample_dataset(dataset, n_bootstrap_samples, regime):
    """Bootstrap sample a dataset of size n_bootstrap_samples from the data points
        in dataset.

    :param dataset: The original dataset from which to resample
    :type dataset: pandas DataFrame
    :param n_bootstrap_samples: The size of the bootstrapped dataset
    :type n_bootstrap_samples: int
    :param savename: Path to save the bootstrapped dataset.
    :type savename: str
    """
    if regime == "supervised_learning":
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
            meta=dataset.meta,
        )

        return bootstrap_dataset

    elif regime == "reinforcement_learning":
        # TODO: Finish implementing this.
        raise NotImplementedError(
            "Creating bootstrap sampled datasets not yet implemented for "
            "reinforcement_learning regime"
        )


def set_spec_with_hyperparam_setting(spec, hyperparam_setting):
    """
    Update spec according to hyperparam_setting.

    :type hyperparam_setting: tuple containing hyperparameter values that should be
        set for this bootstrap experiment (if not given will use default from self.spec)
    :type hyperparam_setting: tuple of tuples, where each inner tuple takes the form
        (hyperparameter name, hyperparameter type, hyperparameter value)
            Example:
            (("alpha_theta", "optimization", 0.001), ("num_iters", "optimization", 500))
    """
    if hyperparam_setting is not None:
        for (hyper_name, hyper_type, hyper_value) in hyperparam_setting:
            if hyper_type == "optimization":
                spec.optimization_hyperparams[hyper_name] = hyper_value
            elif hyper_type == "model":
                # TODO: Not yet implemented
                raise NotImplementedError(
                    f"Setting hyperparameter type {hyper_type} in hyperparameter selection"
                    " is not yet implemented"
                )
            elif hyper_type == "SA":
                if hyper_name in ["bound_inflation_factor", "delta_split_dict"]:
                    new_parse_trees = rebuild_parse_trees(spec, hyper_name, hyper_value)
                    spec.parse_trees = new_parse_trees

                elif hyper_name == "frac_data_in_safety":
                    spec.frac_data_in_safety == hyper_value

            else:
                raise ValueError(f"{hyper_type} is not a valid hyperparameter type")

    return spec


def rebuild_parse_trees(spec, hyper_name, hyper_val):
    """Build new parse trees from existing spec, injecting new 
    hyperparameter value.

    :param spec: The original spec containing the parse trees to rebuild
    """
    output_parse_trees = []
    for ii, pt in enumerate(spec.parse_trees):
        new_pt = ParseTree(pt.delta, pt.regime, pt.sub_regime, columns=pt.columns)
        # Inject hyperparameter
        if hyper_name == "bound_inflation_factor":
            # hyper_val can either be a list of values (one constant factor to use for all trees)
            # or a list of lists, where each sublist is the factor list to apply to the base nodes of that tree
            if isinstance(hyper_val[ii], (float, int)):
                infl_factor_method = "constant"
            else:
                infl_factor_method = "manual"
            infl_factors = hyper_val[ii]

            # Need to figure out what the delta splitting was in the existing spec
            # base node dict is an ordered dict so we can use it to reconstruct the delta vector
            delta_vector = []
            for unique_base_node in pt.base_node_dict:
                delta_lower = pt.base_node_dict[unique_base_node]["delta_lower"]
                delta_upper = pt.base_node_dict[unique_base_node]["delta_upper"]
                if delta_lower is not None:
                    delta_vector.append(delta_lower)
                if delta_upper is not None:
                    delta_vector.append(delta_upper)
            if delta_vector == []:
                delta_weight_method = "equal"
            else:
                delta_weight_method = "manual"
            new_pt.build_tree(
                pt.constraint_str,
                delta_weight_method=delta_weight_method,
                delta_vector=delta_vector,
                infl_factor_method=infl_factor_method,
                infl_factors=infl_factors,
            )

        elif hyper_name == "delta_split_vector":
            # Need to figure out what the bound inflation factors were in the existing spec
            # base node dict is an ordered dict so we can use it to reconstruct the inflation factors
            infl_factors = []
            for unique_base_node in pt.base_node_dict:
                infl_factor_lower = pt.base_node_dict[unique_base_node][
                    "infl_factor_lower"
                ]
                infl_factor_upper = pt.base_node_dict[unique_base_node][
                    "infl_factor_upper"
                ]
                if infl_factor_lower is not None:
                    infl_factors.append(infl_factor_lower)
                if infl_factor_upper is not None:
                    infl_factors.append(infl_factor_upper)
            if infl_factors == []:
                infl_factor_method = "constant"
            else:
                infl_factor_method = "manual"

            new_pt.build_tree(
                spec.constraint_str,
                delta_weight_method="manual",
                delta_vector=hyper_val,
                infl_factor_method="manual",
                infl_factors=infl_factors,
            )

        output_parse_trees.append(new_pt)
    return output_parse_trees


def ttest_bound(hyperparam_spec, bootstrap_trial_data, delta=0.1):
    """
    Compute ttest bound on the probability of passing using the bootstrap data across
        bootstrap trials.

    :param bootstrap_trial_data: Array of size n_bootstrap_samples, containing the 
        result of each bootstrap trial.
    :type bootstrap_trial_data: np.array
    :param delta: confidence level, i.e. 0.05
    :type delta: float
    """
    bs_data_mean = np.nanmean(bootstrap_trial_data)  # estimated probability of passing
    bs_data_stddev = np.nanstd(bootstrap_trial_data)

    lower_bound = bs_data_mean - bs_data_stddev / np.sqrt(
        hyperparam_spec.n_bootstrap_trials
    ) * tinv(1.0 - delta, hyperparam_spec.n_bootstrap_trials - 1)
    upper_bound = bs_data_mean + bs_data_stddev / np.sqrt(
        hyperparam_spec.n_bootstrap_trials
    ) * tinv(1.0 - delta, self.hyperparam_spec.n_bootstrap_trials - 1)

    return lower_bound, upper_bound


def clopper_pearson_bound(hyperparam_spec, pass_count, alpha=0.1):
    # TODO: Write tests.
    """
    Computes a 1-alpha clopper pearson bound on the probability of passing. 

    :param pass_count: number of trials out of n_bootstrap_trials that passed
    :type pass_count: int
    :param alpha: confidence parameter
    :type alpha : float
    """
    lower_bound = scipy.stats.beta.ppf(
        alpha / 2, pass_count, hyperparam_spec.n_bootstrap_trials - pass_count + 1
    )
    upper_bound = scipy.stats.beta.ppf(
        1 - alpha / 2, pass_count + 1, hyperparam_spec.n_bootstrap_trials - pass_count
    )

    return lower_bound, upper_bound
