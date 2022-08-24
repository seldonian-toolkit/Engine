# WARNING: naming this file "simglucose" causes some very confusing and misleading gym errors (ultimately caused by a silent conflict between that file name and the simglucose package)
import gym
from gym.envs.registration import register
from seldonian.RL.environments.Environment import *
from seldonian.RL.Env_Description.Env_Description import *
import simglucose

class Simglucose(Environment):
    def __init__(self):
        self.num_actions = 5  # how many actions to discretize
        self.id = 'simglucose-adolescent2-v0'
        self.patient_name = 'adolescent#002'

        self.deregister_and_register()
        self.action_multiplier = 30. / (self.num_actions - 1)
        self.gym_env = gym.make(self.id)
        self.env_description = self.create_env_description()
        self.terminal_state = False
        self.observation = None
        self.reset() #updates self.observation

    def reset(self):
        self.observation = self.gym_observation_to_observation(self.gym_env.reset())

    def transition(self, action):
        environment_action = action * self.action_multiplier
        observation, reward, self.terminal_state, info = self.gym_env.step(environment_action)
        self.observation = self.gym_observation_to_observation(observation)
        return reward

    def get_observation(self):
        return self.observation

    def gym_observation_to_observation(self, gym_obs):
        return np.array([gym_obs[0]]) #get the scalar out of the simglucose Observation object, then put it in a numpy array

    def create_env_description(self):
        fake_max = 100.0 #in gym it's technically infinity
        obs_space_bounds = np.array([[0, fake_max]])
        obs_space = Continuous_Space(obs_space_bounds)
        action_space = Discrete_Space(0, self.num_actions-1)
        return Env_Description(obs_space, action_space)

    def deregister_and_register(self):
        self.deregister()
        register(
            id=self.id,
            entry_point='simglucose.envs:T1DSimEnv',
            kwargs={'patient_name': self.patient_name}
        )

    def deregister(self):
        env_dict = gym.envs.registration.registry.env_specs.copy()
        for env in env_dict:
            if self.id in env:
                del gym.envs.registration.registry.env_specs[env]