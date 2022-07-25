import autograd.numpy as np


class Discrete_Space:
    def __init__(self, min, max):
        self.min = min
        self.max = max

    def get_num_values(self):
        return self.max - self.min + 1


class Continuous_Space:
    def __init__(self, bounds):
        self.num_dims = bounds.shape[0]
        self.check_bounds_valid(bounds)
        self.bounds = bounds

    def check_bounds_valid(self, bounds):
        if type(bounds) != np.ndarray:
            raise(Exception(f"Expected np.ndarray, got {type(bounds)}"))
        shape = bounds.shape
        if len(shape) != 2 or shape[1] != 2:
            raise(Exception(f"Expected bounds with shape n x 2, got shape {shape}"))
        for dim in range(self.num_dims):
            if bounds[dim][0] > bounds[dim][1]:
                raise(Exception(f"for dimension {dim}, min {bounds[dim][0]} is greater than max {bounds[dim][1]}"))
