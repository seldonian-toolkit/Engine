import os
import pickle
import json

def dir_path(string):
    if os.path.isdir(string):
        return string
    else:
        raise NotADirectoryError(string)


def load_pickle(filename):
    with open(filename,'rb') as infile:
        data = pickle.load(infile)
    return data

def load_json(filename):
    with open(filename,'r') as infile:
        data = json.load(infile)
    return data