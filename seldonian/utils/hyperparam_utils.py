import autograd.numpy as np  # Thinly-wrapped version of Numpy
from seldonian.dataset import SupervisedDataSet
from seldonian.parse_tree.parse_tree import ParseTree

def create_shuffled_dataset(
    dataset
    ):
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


def bootstrap_sample_dataset(
    dataset,
    n_bootstrap_samples,
    regime
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
                "reinforcement_learning regime")

def rebuild_parse_trees(spec,hyperparam_name,hyperparam_value):
    """Build new parse trees from existing spec, injecting new 
    hyperparameter value.

    :param spec: The original spec containing the parse trees to rebuild
    """ 
    output_parse_trees = []
    for pt in spec.parse_trees:
        new_pt = ParseTree( delta, regime, sub_regime, columns=[])
