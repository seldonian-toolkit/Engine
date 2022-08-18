import gym
from gym.envs.registration import register
from seldonian.RL.environments.Environment import *
from seldonian.RL.Env_Description.Env_Description import *

class Simglucose(Environment):
    def __init__(self):
        register(
            id='simglucose-adolescent2-v0',
            entry_point='simglucose.envs:T1DSimEnv',
            kwargs={'patient_name': 'adolescent#002'}
        )
        self.gym_env = gym.make('simglucose-adolescent2-v0')
        # self.env_description = self.create_env_description()
        self.terminal_state = False
        self.observation = None

    def reset(self):
        self.gym_env.reset()

    def transition(self, action):
        observation, reward, self.terminal_state, info = self.gym_env.step(action)
        self.observation = observation[0] # it contains a float
        return reward

    def get_observation(self):
        return self.observation

    # def create_env_description(self):
    #

Zipdog = Simglucose()
print(Zipdog.gym_env.action_space)
print(type(Zipdog.gym_env.action_space))
print(Zipdog.gym_env.action_space.low)
print(Zipdog.gym_env.action_space.high)
print(Zipdog.gym_env.action_space.shape)
print()
print(Zipdog.gym_env.observation_space)
print(type(Zipdog.gym_env.observation_space))
print(Zipdog.gym_env.observation_space.low)
print(Zipdog.gym_env.observation_space.high)
print(Zipdog.gym_env.observation_space.shape)