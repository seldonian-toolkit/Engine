import pytest
from seldonian.RL.Agents.Function_Approximators.Table import *
import autograd.numpy as np

def test_tables():
    min_state = -3
    num_states = 6
    num_actions = 3
    mytable = Q_Table(min_state, num_states, num_actions)

    assert mytable.from_environment_state_to_0_indexed_state(-3) == 0
    assert mytable.from_environment_state_to_0_indexed_state(2) == 5

    assert np.allclose(mytable.get_action_values_given_state(-3), np.array([0.0, 0.0, 0.0]))

    mytable.weights[0, 0] = 1.1
    mytable.weights[1, 0] = 2.2
    mytable.weights[2, 0] = -3.3
    assert np.allclose(mytable.get_action_values_given_state(-3), np.array([1.1, 2.2, -3.3]))
    for state in range(-2, 3):
        assert np.allclose(mytable.get_action_values_given_state(state), np.array([0.0, 0.0, 0.0]))

    mytable.weights[0, 3] = 1.1
    mytable.weights[1, 3] = 2.2
    mytable.weights[2, 3] = -3.3
    assert np.allclose(mytable.get_action_values_given_state(0), np.array([1.1, 2.2, -3.3]))