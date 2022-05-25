import os
import pickle

def dir_path(string):
    if os.path.isdir(string):
        return string
    else:
        raise NotADirectoryError(string)


def load_pickle(filename):
    with open(filename,'rb') as infile:
        data = pickle.load(infile)
    return data
