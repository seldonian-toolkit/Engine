import os
import pytest
import autograd.numpy as np

from seldonian.utils.io_utils import *

### Begin tests


def test_dir_path():
    """Test the Bessel's-corrected standard deviation
    function"""

    path = "static/"
    assert dir_path(path) == path

    path = "not_a_path/definitely_not_a_path"
    with pytest.raises(NotADirectoryError) as excinfo:
        p = dir_path(path)

    assert str(excinfo.value) == path


def test_save_pickle():
    """Test the Bessel's-corrected standard deviation
    function"""
    path = "tests/testfile.pkl"
    data = {"testkey": 1}
    save_pickle(path, data)
    assert os.path.exists(path)
    os.remove(path)


def test_save_json():
    """Test the Bessel's-corrected standard deviation
    function"""
    path = "tests/testfile.json"
    data = {"testkey": 1}
    save_json(path, data)
    assert os.path.exists(path)
    os.remove(path)
