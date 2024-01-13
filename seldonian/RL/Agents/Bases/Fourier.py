from seldonian.utils.RL_utils import *
from math import factorial, pi


class Fourier(object):
    def __init__(self, hyperparam_and_setting_dict, env_desc):
        """Fourier basis used for linear value function
        approximation. See http://irl.cs.brown.edu/fb.php for
        a reference. Each feature is a sine or cosine function of a certain order.

        :param hyperparameter_and_setting_dict: Specifies the
            environment, agent, number of episodes per trial,
            and number of trials
        :param env_desc: an object for accessing attributes
            of the environment
        :type env_desc: :py:class:`.Env_Description`
        """
        self.num_observation_dims = env_desc.get_num_observation_dims()
        self.order = hyperparam_and_setting_dict["order"]
        if self.order <= 0:
            error("order must be positive")

        if self.num_observation_dims <= 0:  # pragma: no cover
            error("num_observation_dims must be positive")

        self.max_coupled_vars = hyperparam_and_setting_dict["max_coupled_vars"]
        if self.max_coupled_vars == -1:
            self.max_coupled_vars = self.num_observation_dims
        if self.max_coupled_vars <= 0:
            error("max_coupled_vars must be positive")
        if self.max_coupled_vars > self.num_observation_dims:
            error("max_coupled_vars > num_observation_dims")

        self.mins = env_desc.observation_space.bounds[:, 0]
        self.maxes = env_desc.observation_space.bounds[:, 1]
        self.ranges = self.maxes - self.mins

        self.num_features = self.calculate_num_features(
            self.order, self.max_coupled_vars, self.num_observation_dims
        )
        self.basis_matrix = self.construct_basis_matrix()

    def calculate_num_features(self, order, max_coupled_vars, num_obs_dims):
        """Calculate the number of features in a Reinforcement Learning (RL) Fourier basis.

        This function computes the total number of Fourier basis features for a given order,
        considering the maximum number of coupled variables and the number of observation
        dimensions in the RL environment.

        :param order: The maximum order of the Fourier basis. This defines the highest
            frequency of sine and cosine functions in the basis.
        :type order: int
        :param max_coupled_vars: The maximum number of variables that can be coupled in a
            single feature. This parameter controls the complexity of interaction terms in the
            Fourier basis.
        :type max_coupled_vars: int
        :param num_obs_dims: The number of dimensions in the observation space of the RL
            environment. Each dimension can be considered as a variable in the Fourier basis.
        :type num_obs_dims: int

        :return num_features: The total number of features in the Fourier basis, which includes all possible
            combinations of sine and cosine functions up to the specified order, considering the
            limit on coupled variables and the number of observation dimensions.
        """
        num_features = (order + 1) ** num_obs_dims
        for mandatory_0_observation_variables in range(
            max_coupled_vars + 1, num_obs_dims + 1
        ):
            num_features -= (
                order ** mandatory_0_observation_variables
                * factorial(num_obs_dims)
                / factorial(num_obs_dims - mandatory_0_observation_variables)
                / factorial(mandatory_0_observation_variables)
            )
        return num_features

    def construct_basis_matrix(self):
        """Construct and return the basis matrix for a Reinforcement Learning (RL) Fourier basis.

        This function builds a matrix where each row represents a feature in the Fourier basis.

        :return basis_matrix: A 2D numpy array of shape (num_features, num_observation_dims), where
        num_features is the total number of valid Fourier basis features and num_observation_dims
        is the number of dimensions in the observation space. Each entry in the matrix is an
        integer representing the frequency for that dimension in the Fourier basis.
        """
        basis_matrix = np.zeros(
            (self.num_features, self.num_observation_dims), dtype=int
        )
        row = 0  # row of the matrix corresponds to the features
        fully_coupled_num_features = (self.order + 1) ** self.num_observation_dims
        for fully_coupled_feature in range(fully_coupled_num_features):
            num_in_row_non_zero = 0
            for observation_dim in range(self.num_observation_dims):
                rep_size = (self.order + 1) ** (
                    self.num_observation_dims - observation_dim - 1
                )  # representation size of count on this entry (e.g. if binary (meaning order = 1) then last column has rep_size 1, 2nd-to last has 2, 3rd to last has 4, etc.)
                entry_value = (fully_coupled_feature / rep_size) % (self.order + 1)
                if entry_value != 0:
                    num_in_row_non_zero += 1
                basis_matrix[
                    row, observation_dim
                ] = entry_value  # adding plus one for C++-to-Julia translation
            if num_in_row_non_zero > self.max_coupled_vars:  # redo the row
                row -= 1
            row += 1
            if row == self.num_features:
                break  # don't want it to keep running invalid lines if it got all the good ones (they'll be out of range of the matrix)
        if row != self.num_features:
            error("row != num_features at this point, this should never happen")
        return basis_matrix

    def get_features(self, obs):
        """Get the basis feature given an observation

        :param obs: unnormalized observation

        :return ret_matrix: A matrix of features for the given observation
        """
        normalized_obs = self.get_normalized_observation(obs)
        ret_matrix = np.dot(self.basis_matrix, normalized_obs)
        ret_matrix = np.cos(pi * ret_matrix)
        return ret_matrix

    def get_normalized_observation(self, obs):
        """Get the normalized observation given an observation

        :param obs: unnormalized observation

        :return norm_obs: normalized observation
        """
        norm_obs = np.zeros(self.num_observation_dims)
        for obs_dim in range(self.num_observation_dims):
            norm_obs[obs_dim] = (obs[obs_dim] - self.mins[obs_dim]) / self.ranges[
                obs_dim
            ]
        return norm_obs
