import pytest
import autograd.numpy as np

from seldonian.utils.tutorial_utils import generate_data


### Begin tests
def test_generate_data():
    """Test the function used for generating synthetic
    data for the tutorial example"""
    np.random.seed(42)
    numPoints = 1000
    X, Y = generate_data(numPoints)
    assert X.mean() == pytest.approx(0.01933206)
    assert Y.mean() == pytest.approx(0.09016829)
    assert X.std() == pytest.approx(0.978726208)
    assert Y.std() == pytest.approx(1.368570511)

    assert len(X) == numPoints
    assert len(Y) == numPoints
