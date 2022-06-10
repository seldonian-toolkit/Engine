import os
import pickle
import json

def dir_path(path):
    """ A utility for checking whether a path is a directory

    :param path: An input path that may or may not be a directory
    :type path: str
    """
    if os.path.isdir(path):
        return path
    else:
        raise NotADirectoryError(path)


def load_pickle(filename):
    """ A wrapper for loading an object from a pickle file

    :param filename: A filename pointing to a pickle file
    :type filename: str
    """
    with open(filename,'rb') as infile:
        data = pickle.load(infile)
    return data

def save_pickle(filename,data,verbose=False):
    """ A wrapper for saving an object to a pickle file

    :param filename: A filename for the saved pickle file
    :type filename: str

    :param data: The object you want to pickle
    :type data: Pickle-able object
    """
    with open(filename,'wb') as outfile:
        pickle.dump(data,outfile,protocol=pickle.HIGHEST_PROTOCOL)
        if verbose:
            print(f"Saved {filename}\n")

def load_json(filename):
    """ A wrapper for loading an object from a JSON file

    :param filename: An input filename pointing to a JSON file
    :type filename: str
    """
    with open(filename,'r') as infile:
        data = json.load(infile)
    return data

def yes_or_no_input(str_to_show,default_str,default_val):
    """ Show user a yes or no question and gather their 
    input from the command line. If they provide an invalid
    answer, let them know and show them the same question again

    :param str_to_show: The question shown to the user
    :type str_to_show: str

    :param default_val: The value that is returned 
        if the user enters nothing
    :type default_val: bool

    """
    if default_str == 'y':
        yes_str = '[y]'
        no_str = 'n'
    elif default_str == 'n':
        yes_str = 'y'
        no_str = '[n]'
    str_to_show += f' {yes_str} or {no_str}'
    str_to_show+= ': '
    
    while True:
        result = input(str_to_show)
        if not result:
            result = default_val
            break
        elif result == 'y':
            result = True
            break
        elif result == 'n':
            result = False
            break
        else:
            print(f'"{result}" was not a valid input. Please try again.')
    return result

