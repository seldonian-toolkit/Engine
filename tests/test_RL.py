import pytest
from seldonian.RL.Agents.Function_Approximators.Table import *
from seldonian.RL.Agents.Policies.Policy import *
from seldonian.RL.Agents.Policies.Softmax import *
from seldonian.RL.Agents.Parameterized_non_learning_softmax_agent import *
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
    sm = Softmax(-1, 1)
    e_to_something_stable = np.array([0.1108031584, 0.0040867714, 1.0])
    assert np.allclose(sm.get_e_to_the_something_terms([1.1, -2.2, 3.3]), e_to_something_stable)
    assert np.allclose(sm.get_action_probs_from_action_values([1.1, -2.2, 3.3]), e_to_something_stable / sum(e_to_something_stable))

def test_Parameterized_non_learning_softmax_agent():
    observation_space = Discrete_Space(-1, 2)
    action_space = Discrete_Space(-1, 1)
    env_desc = Env_Description(observation_space, action_space)
    agent = Parameterized_non_learning_softmax_agent(env_desc,{})

    correct_shape = (3, 4)
    assert agent.get_params().shape == correct_shape

    new_params = np.random.rand(correct_shape[0], correct_shape[1])
    agent.set_new_params(new_params)
    assert np.allclose(agent.get_params(), new_params)

    agent.set_new_params(np.array([[1.1, 1.1, 1.1, 1.1], [-2.2, -2.2, -2.2, -2.2], [3.3, 3.3, 3.3, 3.3]]))
    for state in range(-1, 3):
        assert np.allclose(agent.get_action_values(state), [1.1, -2.2, 3.3])
        assert np.allclose(agent.get_prob_this_action(state, -1), .099384841)
        assert np.allclose(agent.get_prob_this_action(state, 0), .0036656277)
        assert np.allclose(agent.get_prob_this_action(state, 1), 0.8969495313)

    incorrect_shape = (4, 3)
    bad_params = np.random.rand(incorrect_shape[0], incorrect_shape[1])
    with pytest.raises(Exception):
        agent.set_new_params(bad_params)

def test_spaces_and_env_descriptions():
    cont_space = Continuous_Space(np.array([[0.0, 1.1], [3.3, 4.4], [5.5, 6.6]])) #should be no error
    with pytest.raises(Exception):
        cont_space = Continuous_Space(np.array([[0.0, 1.1], [3.3, 4.4], [5.5, -6.6]]))
    with pytest.raises(Exception):
        cont_space = Continuous_Space(np.array([[2.0, 1.1], [3.3, 4.4], [5.5, 6.6]]))
    with pytest.raises(Exception):
        cont_space = Continuous_Space(np.array([[2.0, 2.1, 3.3], [4.4, 5.5, 6.6]]))

    obs_space = Discrete_Space(-10, 10)
    assert obs_space.get_num_values() == 21
    action_space = Discrete_Space(1, 4)
    assert action_space.get_num_values() == 4

    env_desc = Env_Description(obs_space, action_space)
    assert env_desc.get_num_states() == 21
    assert env_desc.get_num_actions() == 4
    assert env_desc.get_min_action() == 1
    assert env_desc.get_min_state() == -10


