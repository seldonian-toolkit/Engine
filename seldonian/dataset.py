""" Build and load datasets for running Seldonian algorithms """

import autograd.numpy as np
import pandas as pd
import pickle
from seldonian.utils.io_utils import load_json, load_pickle


class DataSetLoader:
    def __init__(self, regime, **kwargs):
        """Object for loading datasets from disk into DataSet objects

        :param regime: The category of the machine learning algorithm,
                e.g., supervised_learning or reinforcement_learning
        :type regime: str
        """
        self.regime = regime

    def load_supervised_dataset(self, filename, metadata_filename, file_type="csv"):
        """Create SupervisedDataSet object from file

        :param filename: The file
                containing the features, labels and sensitive attributes
        :type filename: str
        :param metadata_filename: The file
                containing the metadata describing the data in filename
        :type metadata_filename: str
        :param file_type: the file extension of filename
        :type file_type: str, defaults to 'csv'
        """
        # Load metadata
        meta = load_supervised_metadata(metadata_filename)

        if file_type.lower() == "csv":
            df = pd.read_csv(filename, header=None, names=meta.all_col_names)
            # separate out features, labels, and sensitive attrs
            features = df.loc[:, meta.feature_col_names].values
            labels = np.squeeze(
                df.loc[:, meta.label_col_names].values
            )  # converts shape from (N,1) -> (N,) if only a single label column.
            if meta.sensitive_col_names != []:
                sensitive_attrs = df.loc[:, meta.sensitive_col_names].values
            else:
                sensitive_attrs = []
            num_datapoints = len(df)
        else:
            raise NotImplementedError(f"File type: {file_type} not supported")

        return SupervisedDataSet(
            features=features,
            labels=labels,
            sensitive_attrs=sensitive_attrs,
            num_datapoints=num_datapoints,
            meta=meta,
        )

    def load_RL_dataset_from_csv(self, filename, metadata_filename=None):
        """Create RLDataSet object from file
        containing the episodes as a CSV with format:
        episode_index,obs,action,reward,probability_of_action.

        :param filename: The file
                containing the data you want to load
        :type filename: str
        :param metadata_filename: Name of metadata file
        :type metadata_filename: str
        """

        required_col_names = ["episode_index", "O", "A", "R", "pi_b"]
        # Load metadata
        meta = load_RL_metadata(metadata_filename, required_col_names)

        df = pd.read_csv(filename, header=None)
        df.columns = meta.all_col_names
        episodes = []
        # Set flag if there are alt rewards
        has_alt_rewards = False
        n_min_required_cols = len(required_col_names)
        if len(meta.all_col_names) > n_min_required_cols:
            if "R_alt_1" in meta.all_col_names:
                has_alt_rewards = True
                n_alt_rewards = len(meta.all_col_names) - n_min_required_cols
            else:
                raise RuntimeError(
                    "You specified in 'all_col_names' more than the minimum "
                    f"required number of columns: {n_min_required_cols} "
                    "and the extra column names do not follow the 'R_alt_1','R_alt_2', ... pattern. "
                    "Update the names of these columns, which represent the optional alternate rewards."
                )
        for episode_index in df.episode_index.unique():
            df_ep = df.loc[df.episode_index == episode_index]
            if has_alt_rewards:
                alt_reward_names = [f"R_alt_{ii}" for ii in range(1, n_alt_rewards + 1)]
                alt_rewards = df.loc[:, alt_reward_names].values
            else:
                alt_rewards = []

            episode = Episode(
                observations=df_ep.O.values,
                actions=df_ep.A.values,
                rewards=df_ep.R.values,
                action_probs=df_ep.pi_b.values,
                alt_rewards=alt_rewards,
            )
            episodes.append(episode)
        if meta.sensitive_col_names != []:
            sensitive_attrs = df.loc[:, meta.sensitive_col_names].values
        else:
            sensitive_attrs = []
        return RLDataSet(episodes=episodes, meta=meta)

    def load_RL_dataset_from_episode_file(self, filename, metadata_filename=None):
        """Create RLDataSet object from file

        :param filename: The file
                containing the pickled lists of :py:class:`.Episode` objects
        :type filename: str
        """
        required_col_names = ["episode_index", "O", "A", "R", "pi_b"]
        episodes = load_pickle(filename)
        meta = load_RL_metadata(metadata_filename, required_col_names)
        return RLDataSet(episodes=episodes, meta=meta)


class DataSet(object):
    def __init__(self, num_datapoints, meta, regime, **kwargs):
        """Object for holding dataframe and dataset metadata

        :param num_datapoints: Number of rows or episodes (for RL) in the dataset
        :type num_datapoints: int
        :param meta: Metadata
        :type meta: :py:class:`.MetaData`
        :param regime: The category of the machine learning algorithm,
                e.g., "supervised_learning", "reinforcement_learning" or "custom"
        :type regime: str
        """
        self.num_datapoints = num_datapoints
        self.meta = meta
        self.regime = regime


class SupervisedDataSet(DataSet):
    def __init__(self, features, labels, sensitive_attrs, num_datapoints, meta):
        super().__init__(
            num_datapoints=num_datapoints,
            meta=meta,
            regime="supervised_learning",
        )

        self.features = features

        assert isinstance(labels, np.ndarray), "labels must be a numpy array"
        self.labels = labels

        self.sensitive_attrs = sensitive_attrs
        assert (
            isinstance(self.sensitive_attrs, np.ndarray) or self.sensitive_attrs == []
        ), "sensitive_attrs must be a numpy array or []"

        self.feature_col_names = meta.feature_col_names
        self.label_col_names = meta.label_col_names
        self.sensitive_col_names = meta.sensitive_col_names

        self.n_features = len(self.feature_col_names)
        self.n_labels = len(self.label_col_names)
        self.n_sensitive_attrs = len(self.sensitive_col_names)


class RLDataSet(DataSet):
    def __init__(
        self,
        episodes,
        meta,
        sensitive_attrs=[],
        **kwargs,
    ):
        """Object for holding RL dataframe and dataset metadata

        :param episodes: List of episodes
        :type episodes: list(:py:class:`.Episode`)
        :param meta_information: List of attribute names in each Episode,
                e.g. ['o','a','r','pi_b']
        :type meta_information: list(str)
        """

        self.episodes = episodes
        self.sensitive_attrs = sensitive_attrs
        self.sensitive_col_names = meta.sensitive_col_names
        super().__init__(
            num_datapoints=len(self.episodes),
            meta=meta,
            regime="reinforcement_learning",
        )

class CustomDataSet(DataSet):
    def __init__(self, data, sensitive_attrs, num_datapoints, meta, **kwargs):
        """Object for holding data of arbitrary form. 

        :param data: An 1D array of data samples, where the samples can have any form.
        :param num_datapoints: Number of data samples
        :type num_datapoints: int
        :param meta_information: list of all column names in the dataframe
        :type meta_information: List(str)
        :param regime: The category of the machine learning algorithm,
                e.g., supervised_learning or reinforcement_learning
        :type regime: str
        """
        super().__init__(
            num_datapoints=num_datapoints,
            meta=meta,
            regime="custom",
        )
        self.data = data
        if not ( (isinstance(self.data, np.ndarray) and self.data.ndim == 2) or isinstance(self.data,list)):
            raise RuntimeError("data must be a numpy array or list")

        if isinstance(self.data,list):
            if type(self.data[0]) != type(self.data[-1]):
                raise ValueError("All elements of data must be of same type")
                
        self.sensitive_attrs = sensitive_attrs
        assert (
            isinstance(self.sensitive_attrs, np.ndarray) or self.sensitive_attrs == []
        ), "sensitive_attrs must be a numpy array or []"
        self.sensitive_col_names = meta.sensitive_col_names

class Episode(object):
    def __init__(self, observations, actions, rewards, action_probs, alt_rewards=[]):
        """Object for holding RL episodes

        :param observations: List of observations for each timestep
                in the episode
        :param actions: List of actions
        :param rewards: List of rewards
        :param action_probs: List of action probabilities
                from the behavior policy
        """
        self.observations = np.array(observations)
        self.actions = np.array(actions)
        self.rewards = np.array(rewards)
        self.action_probs = np.array(action_probs)
        self.alt_rewards = np.array(alt_rewards)
        self.n_alt_rewards = (
            0 if self.alt_rewards.size == 0 else self.alt_rewards.shape[1]
        )

    def __str__(self):
        s = (
            f"return = {sum(self.rewards)}\n"
            + f"{len(self.observations)} observations, type of first in array is {type(self.observations[0])}: {self.observations}\n"
            + f"{len(self.actions)} actions, type of first in array is {type(self.actions[0])}: {self.actions}\n"
            + f"{len(self.rewards)} rewards, type of first in array is {type(self.rewards[0])}: {self.rewards}\n"
            + f"{len(self.action_probs)} action_probs, type of first in array is {type(self.action_probs[0])}: {self.action_probs}"
        )
        if self.n_alt_rewards > 0:
            for ii in range(self.n_alt_rewards):
                alt_reward = self.alt_rewards[:, ii]
                s += (
                    f"\n{len(alt_reward)} of alt reward {ii+1} of {self.n_alt_rewards}, "
                    f"type of first in array is {type(alt_reward[0])}: {alt_reward}"
                )
        return s

    def __repr__(self):
        n_timesteps = len(self.observations)
        tup_str = "(obs,action,primary_reward,pi_b)_i"
        if self.n_alt_rewards > 0:
            alt_rewards_str = ",".join(
                [f"alt_reward_{ii}" for ii in range(1, self.n_alt_rewards + 1)]
            )
            tup_str = tup_str.replace(")_i", "," + alt_rewards_str + ")_i")
        repr_s = f"Episode with {n_timesteps} timesteps: {tup_str}"

        return repr_s


class MetaData(object):
    def __init__(self, regime, sub_regime, all_col_names, sensitive_col_names=None):
        """Base class for holding dataset metadata"""
        self.regime = regime
        self.sub_regime = sub_regime
        self.all_col_names = all_col_names
        self.sensitive_col_names = sensitive_col_names


class SupervisedMetaData(MetaData):
    def __init__(
        self,
        sub_regime,
        all_col_names,
        feature_col_names,
        label_col_names,
        sensitive_col_names=[],
    ):
        """Class for holding supervised learning dataset metadata"""
        super().__init__(
            "supervised_learning", sub_regime, all_col_names, sensitive_col_names
        )
        self.feature_col_names = feature_col_names
        self.label_col_names = label_col_names


class RLMetaData(MetaData):
    def __init__(self, all_col_names, sensitive_col_names=[]):
        """Class for holding supervised learning dataset metadata"""
        super().__init__(
            regime="reinforcement_learning",
            sub_regime="all",
            all_col_names=all_col_names,
            sensitive_col_names=sensitive_col_names,
        )

class CustomMetaData(MetaData):
    def __init__(
        self,
        all_col_names,
        sensitive_col_names=[],
    ):
        """Class for holding custom dataset metadata"""
        super().__init__(
            regime="custom", sub_regime=None, all_col_names=all_col_names, sensitive_col_names=sensitive_col_names
        )


def load_supervised_metadata(filename):
    """Load metadata from JSON file into a dictionary

    :param filename: The file to load
    """
    metadata_dict = load_json(filename)
    regime = metadata_dict["regime"]
    assert regime == "supervised_learning"
    sub_regime = metadata_dict["sub_regime"]
    assert sub_regime in [
        "regression",
        "classification",
        "binary_classification",
        "multiclass_classification",
    ]
    all_col_names = metadata_dict["all_col_names"]

    label_col_names = metadata_dict["label_col_names"]

    if "sensitive_col_names" not in metadata_dict:
        sensitive_col_names = []
    else:
        sensitive_col_names = metadata_dict["sensitive_col_names"]

    if "feature_col_names" not in metadata_dict:
        # infer feature column names - keep order same
        feature_col_names = [
            x
            for x in all_col_names
            if (x not in label_col_names) and (x not in sensitive_col_names)
        ]
    else:
        feature_col_names = metadata_dict["feature_col_names"]

    return SupervisedMetaData(
        sub_regime,
        all_col_names,
        feature_col_names,
        label_col_names,
        sensitive_col_names,
    )


def load_RL_metadata(metadata_filename, required_col_names):
    if metadata_filename:
        metadata_dict = load_json(metadata_filename)
        all_col_names = metadata_dict["all_col_names"]
        if not all([x in all_col_names for x in required_col_names]):
            raise RuntimeError(
                "You are missing some or all of the following required columns "
                "in the 'all_col_names' key of your metadata file:"
                f"{required_col_names}"
            )
        if "sensitive_col_names" in metadata_dict:
            sensitive_col_names = metadata_dict["sensitive_col_names"]
        else:
            sensitive_col_names = []

    else:
        all_col_names = required_col_names
        sensitive_col_names = []

    meta = RLMetaData(
        all_col_names=all_col_names, sensitive_col_names=sensitive_col_names
    )
    return meta


def load_custom_metadata(filename):
    """Load metadata from JSON file into a dictionary

    :param filename: The file to load
    """
    metadata_dict = load_json(filename)
    regime = metadata_dict["regime"]
    assert regime == "custom"

    all_col_names = metadata_dict["all_col_names"]

    if "sensitive_col_names" not in metadata_dict:
        sensitive_col_names = []
    else:
        sensitive_col_names = metadata_dict["sensitive_col_names"]

    return CustomMetaData(
        all_col_names,
        sensitive_col_names,
    )
