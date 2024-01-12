import autograd.numpy as np

def error(output_string):
    """ Wrapper to raise expection """
    raise Exception(output_string)

def clamp(val_in, min_val, max_val):
    """Limit val_in to be between min_val and max_val"""
    if min_val > max_val:
        error(f"min_val {min_val} > max_val {max_val}")
    return min(max(val_in, min_val), max_val)

def argmax_multi(array_in):
    """argmax, but returns multiple indices in case of tie"""
    return np.argwhere(array_in == array_in.max()).flatten()
