from seldonian.utils.RL_utils import *
from math import factorial, pi

class Fourier:
    def __init__(self, hyperparam_and_setting_dict, env_desc):
        self.num_state_dims = env_desc.get_num_state_dims()
        self.order = hyperparam_and_setting_dict["order"]
        if self.order <= 0:
            error("order must be positive")

        if self.num_state_dims <= 0:
            error("num_state_dims must be positive")

        self.max_coupled_vars = hyperparam_and_setting_dict["max_coupled_vars"]
        if self.max_coupled_vars == -1:
            self.max_coupled_vars = self.num_state_dims
        if self.max_coupled_vars <= 0:
            error("max_coupled_vars must be positive")
        if self.max_coupled_vars > self.num_state_dims:
            error("max_coupled_vars > num_state_dims")

        self.mins = env_desc.observation_space.bounds[:, 0]
        self.maxes = env_desc.observation_space.bounds[:, 1]
        self.ranges = self.maxes - self.mins

        self.num_features = self.calculate_num_features(self.order, self.max_coupled_vars, self.num_state_dims)
        self.basis_matrix = self.construct_basis_matrix()

    def calculate_num_features(self, order, max_coupled_vars, num_state_dims):
        num_features = (order + 1) ** num_state_dims
        for mandatory_0_state_variables in range(max_coupled_vars+1, num_state_dims+1):
            num_features -= order ** mandatory_0_state_variables * factorial(num_state_dims) / factorial(num_state_dims - mandatory_0_state_variables) / factorial(mandatory_0_state_variables)
        return num_features

    def construct_basis_matrix(self):
        basis_matrix = np.zeros((self.num_features, self.num_state_dims), dtype=int)
        row = 0  # row of the matrix corresponds to the features
        fully_coupled_num_features = (self.order + 1) ** self.num_state_dims
        for fully_coupled_feature in range(fully_coupled_num_features):
            num_in_row_non_zero = 0
            for state_dim in range(self.num_state_dims):
                rep_size = (self.order + 1) ** (self.num_state_dims - state_dim - 1)  # representation size of count on this entry (e.g. if binary (meaning order = 1) then last column has rep_size 1, 2nd-to last has 2, 3rd to last has 4, etc.)
                entry_value = (fully_coupled_feature / rep_size) % (self.order + 1)
                if entry_value != 0:
                    num_in_row_non_zero += 1
                basis_matrix[row, state_dim] = entry_value  # adding plus one for C++-to-Julia translation
            if num_in_row_non_zero > self.max_coupled_vars:  # redo the row
                row -= 1
            row += 1
            if row == self.num_features:
                break  # don't want it to keep running invalid lines if it got all the good ones (they'll be out of range of the matrix)
        if row != self.num_features:
            error("row != num_features at this point, this should never happen")
        return basis_matrix

    def get_features(self, obs):
        normalized_state = self.get_normalized_state(obs)
        ret_matrix = np.dot(self.basis_matrix, normalized_state)
        ret_matrix = np.cos(pi*ret_matrix)
        return ret_matrix

    def get_normalized_state(self, obs):
        norm_state = np.zeros(self.num_state_dims)
        for state_dim in range(self.num_state_dims):
            norm_state[state_dim] = (obs[state_dim] - self.mins[state_dim]) / self.ranges[state_dim]
        return norm_state
