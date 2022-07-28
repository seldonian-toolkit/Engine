import pytest
from seldonian.RL.Agents.Function_Approximators.Table import *
from seldonian.RL.Agents.Policies.Policy import *
from seldonian.RL.Agents.Policies.Softmax import *
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

def test_Discrete_Action_Policy():
    min_action = -1
    num_actions = 3
    p = Discrete_Action_Policy(min_action, num_actions)

    assert p.from_environment_action_to_0_indexed_action(-1) == 0
    assert p.from_environment_action_to_0_indexed_action(0) == 1
    assert p.from_environment_action_to_0_indexed_action(1) == 2

    assert p.from_0_indexed_action_to_environment_action(0) == -1
    assert p.from_0_indexed_action_to_environment_action(1) == 0
    assert p.from_0_indexed_action_to_environment_action(2) == 1

def test_Softmax():
    sm = Softmax(-1, 3)
    e_to_something_stable = np.array([0.1108031584, 0.0040867714, 1.0])
    assert np.allclose(sm.get_e_to_the_something_terms([1.1, -2.2, 3.3]), e_to_something_stable)
    assert np.allclose(sm.get_action_probs_from_action_values([1.1, -2.2, 3.3]), e_to_something_stable / sum(e_to_something_stable))