import pytest
from seldonian.RL.Agents.Function_Approximators.Table import *
from seldonian.RL.Agents.Policies.Policy import *
from seldonian.RL.Agents.Policies.Softmax import *
from seldonian.RL.environments.mountaincar import *
from seldonian.RL.Agents.Parameterized_non_learning_softmax_agent import *
import autograd.numpy as np

def test_tables():
    """tests Table class methods"""

    min_state = -3
    num_states = 6
    num_actions = 3
    mytable = Q_Table(min_state, num_states, num_actions)

    #check from_environment_state_to_0_indexed_state()
    assert mytable.from_environment_state_to_0_indexed_state(-3) == 0
    assert mytable.from_environment_state_to_0_indexed_state(2) == 5

    #check initialization
    assert np.allclose(mytable.get_action_values_given_state(-3), np.array([0.0, 0.0, 0.0]))

    #check manually setting weights, and get_action_values_given_state()
    mytable.weights[0, 0] = 1.1
    mytable.weights[0, 1] = 2.2
    mytable.weights[0, 2] = -3.3
    assert np.allclose(mytable.get_action_values_given_state(-3), np.array([1.1, 2.2, -3.3]))
    for state in range(-2, 3):
        assert np.allclose(mytable.get_action_values_given_state(state), np.array([0.0, 0.0, 0.0]))

    mytable.weights[3, 0] = 1.1
    mytable.weights[3, 1] = 2.2
    mytable.weights[3, 2] = -3.3
    assert np.allclose(mytable.get_action_values_given_state(0), np.array([1.1, 2.2, -3.3]))

def test_Discrete_Action_Policy():
    """test Discrete_Action_Policy action index methods"""
    min_action = -1
    max_action = 1
    observation_space = Discrete_Space(-1, 2) #irrelevant for test
    action_space = Discrete_Space(min_action, max_action)
    env_description = Env_Description(observation_space, action_space)
    hyperparam_and_setting_dict = {}

    p = Discrete_Action_Policy(hyperparam_and_setting_dict, env_description)

    assert p.from_environment_action_to_0_indexed_action(-1) == 0
    assert p.from_environment_action_to_0_indexed_action(0) == 1
    assert p.from_environment_action_to_0_indexed_action(1) == 2

    assert p.from_0_indexed_action_to_environment_action(0) == -1
    assert p.from_0_indexed_action_to_environment_action(1) == 0
    assert p.from_0_indexed_action_to_environment_action(2) == 1

def test_Softmax():
    """test Softmax functions"""
    min_action = -1
    max_action = 1
    observation_space = Discrete_Space(-1, 2)  # irrelevant for test
    action_space = Discrete_Space(min_action, max_action)
    env_description = Env_Description(observation_space, action_space)
    hyperparam_and_setting_dict = {}

    sm = Softmax(hyperparam_and_setting_dict, env_description)
    e_to_something_stable = np.array([0.1108031584, 0.0040867714, 1.0])
    assert np.allclose(sm.get_e_to_the_something_terms([1.1, -2.2, 3.3]), e_to_something_stable)
    assert np.allclose(sm.get_action_probs_from_action_values([1.1, -2.2, 3.3]), e_to_something_stable / sum(e_to_something_stable))

def test_Parameterized_non_learning_softmax_agent():
    """test Parameterized_non_learning_softmax_agent"""
    observation_space = Discrete_Space(-1, 2)
    action_space = Discrete_Space(-1, 1)
    env_desc = Env_Description(observation_space, action_space)
    hyperparam_and_setting_dict = {}
    hyperparam_and_setting_dict["basis"] = "Fourier"
    agent = Parameterized_non_learning_softmax_agent(env_desc, hyperparam_and_setting_dict)

    correct_shape = (4, 3)
    assert agent.get_params().shape == correct_shape

    new_params = np.random.rand(correct_shape[0], correct_shape[1])
    agent.set_new_params(new_params)
    assert np.allclose(agent.get_params(), new_params) #test set_new_params() and get_params()

    #test get_action_values() and get_prob_this_action()
    agent.set_new_params(np.array([[1.1, 1.1, 1.1, 1.1], [-2.2, -2.2, -2.2, -2.2], [3.3, 3.3, 3.3, 3.3]]).T)
    for state in range(-1, 3):
        assert np.allclose(agent.get_action_values(state), [1.1, -2.2, 3.3])
        assert np.allclose(agent.get_prob_this_action(state, -1), .099384841)
        assert np.allclose(agent.get_prob_this_action(state, 0), .0036656277)
        assert np.allclose(agent.get_prob_this_action(state, 1), 0.8969495313)

    incorrect_shape = (2, 5)
    bad_params = np.random.rand(incorrect_shape[0], incorrect_shape[1])
    with pytest.raises(Exception):
        agent.set_new_params(bad_params) #make sure throws error with bad shape

def test_spaces_and_env_descriptions():
    """test Continuous_Space constructor and methods"""
    cont_space = Continuous_Space(np.array([[0.0, 1.1], [3.3, 4.4], [5.5, 6.6]])) #should be no error
    with pytest.raises(Exception):
        cont_space = Continuous_Space(np.array([[0.0, 1.1], [3.3, 4.4], [5.5, -6.6]])) #max is smaller than min
    with pytest.raises(Exception):
        cont_space = Continuous_Space(np.array([[2.0, 1.1], [3.3, 4.4], [5.5, 6.6]]))  #max is smaller than min
    with pytest.raises(Exception):
        cont_space = Continuous_Space(np.array([[2.0, 2.1, 3.3], [4.4, 5.5, 6.6]])) #need 2 values, not 3

    obs_space = Discrete_Space(-10, 10)
    assert obs_space.get_num_values() == 21
    action_space = Discrete_Space(1, 4)
    assert action_space.get_num_values() == 4

    env_desc = Env_Description(obs_space, action_space)
    assert env_desc.get_num_states() == 21
    assert env_desc.get_num_actions() == 4
    assert env_desc.get_min_action() == 1
    assert env_desc.get_min_state() == -10

def test_Fourier():
    hyperparam_and_setting_dict = {}
    hyperparam_and_setting_dict["order"] = 2
    hyperparam_and_setting_dict["max_coupled_vars"] = -1
    env = Mountaincar()
    env_desc = env.env_description
    basis = Fourier(hyperparam_and_setting_dict, env_desc)
    assert basis.num_features == 9
    assert np.array_equal(basis.basis_matrix, np.array([[0, 0], [0, 1], [0, 2], [1, 0], [1, 1], [1, 2], [2, 0], [2, 1], [2, 2]]))
