import autograd.numpy as np


class Discrete_Space(object):
    def __init__(self, min, max):
        """Discrete space used for observations or actions

        :param min: Minimum value of the space
        :type min: int
        :param max: Maximum value of the space
        :type max: int
        """
        self.min = min
        self.max = max

    def get_num_values(self):
        """Get the total number of values in the space"""
        return self.max - self.min + 1


class Continuous_Space(object):
    def __init__(self, bounds):
        """Continuous space used for observations or actions

        :param bounds: Defines the min,max in each dimension of the space.
            For example, if the space is 3D with x in [-1.0,1.0], y in [5.1,5.4],
            and z in [-12.5,-6.8], then bounds would be:
            np.array([[-1.0,1.0],[5.1,5.4],[-12.5,-6.8]])
        :type bounds: np.ndarray of shape (num_dims,2)

        :ivar num_dims: The number of dimensions in the space
        """
        self.num_dims = bounds.shape[0]
        self.check_bounds_valid(bounds)
        self.bounds = bounds

    def check_bounds_valid(self, bounds):
        """Check that the bounds are in valid format
        and that in each dimension the bound maximum is larger
        than the bound minimum

        :param bounds: Defines the min,max in each dimension of the space.
            For example, if the space is 3D with x in [-1.0,1.0], y in [5.1,5.4],
            and z in [-12.5,-6.8], then bounds would be:
            np.array([[-1.0,1.0],[5.1,5.4],[-12.5,-6.8]])
        :type bounds: np.ndarray of shape (num_dims,2)
        """

        if type(bounds) != np.ndarray:
            raise (Exception(f"Expected np.ndarray, got {type(bounds)}"))
        shape = bounds.shape
        if len(shape) != 2 or shape[1] != 2:
            raise (Exception(f"Expected bounds with shape n x 2, got shape {shape}"))
        for dim in range(self.num_dims):
            if bounds[dim][0] > bounds[dim][1]:
                raise (
                    Exception(
                        f"for dimension {dim}, min {bounds[dim][0]} is greater than max {bounds[dim][1]}"
                    )
                )

    def get_num_dims(self):
        """Get the number of dimensions of the space"""
        return self.num_dims
