import os
import pickle
import json


def dir_path(path):
    """A utility for checking whether a path is a directory

    :param path: An input path that may or may not be a directory
    :type path: str
    """
    if os.path.isdir(path):
        return path
    else:
        raise NotADirectoryError(path)


def load_pickle(filename):
    """A wrapper for loading an object from a pickle file

    :param filename: A filename pointing to a pickle file
    :type filename: str
    """
    with open(filename, "rb") as infile:
        data = pickle.load(infile)
    return data


def save_pickle(filename, data, verbose=False):
    """A wrapper for saving an object to a pickle file

    :param filename: A filename for the saved pickle file
    :type filename: str
    :param data: The object you want to pickle
    :type data: Pickle-able object
    :param verbose: Boolean verbosity flag
    """
    with open(filename, "wb") as outfile:
        pickle.dump(data, outfile, protocol=pickle.HIGHEST_PROTOCOL)
        if verbose:
            print(f"Saved {filename}\n")


def load_json(filename):
    """A wrapper for loading an object from a JSON file

    :param filename: An input filename pointing to a JSON file
    :type filename: str
    """
    with open(filename, "r") as infile:
        data = json.load(infile)
    return data


def save_json(filename, data, indent=2, verbose=False):
    """A wrapper for saving an object to a JSON file

    :param filename: A filename where the JSON file will be saved
    :type filename: str
    """
    with open(filename, "w") as outfile:
        data = json.dump(data, outfile)
    if verbose:
        print(f"Saved {filename}\n")


def cmaes_logger(es,filename):
    """ A function that is used as a callback each time the
    CMA-ES evoluation strategy object, es, is evaluated,
    allowing us to log the xmean and f value at each
    iteration 

    """
    it_str = str(es.countiter)
    xmean_str = ",".join(str(x) for x in es.mean)
    f_str = str(es.best.f)
    with open(filename, "a") as logger:
        logger.write(f"{it_str},{xmean_str},{f_str}\n")
