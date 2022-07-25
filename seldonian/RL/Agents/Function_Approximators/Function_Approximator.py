import autograd.numpy as np
from utils import *

class Function_Approximator:
    pass

    def set_new_params(self, new_params):
        old_shape = np.shape(self.weights)
        new_shape = np.shape(new_params)
        if old_shape != new_shape:
            error(f"Trying to set new params, but old shape is {old_shape} and new shape is {new_shape}. They should be the same")
        self.weights = new_params