from seldonian.RL.environments.Environment import *
from seldonian.RL.Env_Description.Env_Description import *

import gym
from gym.envs.registration import register

MAX_TIMESTEPS = 480 # 480 minutes is a day at 3 minutes per timestep
################################
# Running a single simglucose episode #
################################

TARGET = 160
MIN_INSULIN = 0
MAX_INSULIN = 30

class SimglucoseBasalOnlyEnv(Environment):
    def __init__(self):
        """

        :ivar num_states: The number of distinct grid cells
        :ivar env_description: contains attributes describing the environment
        :vartype env_description: :py:class:`.Env_Description`
        :ivar obs: The current obs
        :vartype obs: int
        :ivar terminal_state: Whether the terminal obs is occupied
        :vartype terminal_state: bool
        :ivar time: The current timestep
        :vartype time: int
        :ivar max_time: Maximum allowed timestep
        :vartype max_time: int
        :ivar gamma: The discount factor in calculating the expected return
        :vartype gamma: float
        """
        self.id = "simglucose-adolescent2-v0"
        self.patient_name = "adolescent#002"

        self.num_states = 1
        self.num_actions = 20

        self.env_description = self.create_env_description()
        self.deregister_and_register()
        self.gym_env = gym.make(self.id)
        self.state = 0 # initial state and the only state
        self.terminal_state = False
        self.time = 0
        self.max_time = 480 # a day at 3 minutes per timestep
        # vis is a flag for visual debugging during obs transitions
        self.vis = False

    def create_env_description(self,):
        """Creates the environment description object.

        :param num_states: The number of states
        :return: Environment description for the obs and action spaces
        :rtype: :py:class:`.Env_Description`
        """
        observation_space = Discrete_Space(0, self.num_states - 1)
        action_space = Discrete_Space(0, self.num_actions - 1)
        return Env_Description(observation_space, action_space)
    
    def deregister_and_register(self):
        self.deregister()
        register(
            id=self.id,
            entry_point="simglucose.envs:T1DSimEnv",
            kwargs={"patient_name": self.patient_name},
        )

    def deregister(self):
        env_dict = gym.envs.registration.registry.env_specs.copy()
        for env in env_dict:
            if self.id in env:
                del gym.envs.registration.registry.env_specs[env]

    def reset(self):
        """Go back to initial obs and timestep"""
        self.state = 0
        self.time = 0
        self.terminal_state = False

    def transition(self, action):
        """A single transition for this P controller 
        is an entire trial of the simglucose simulator

        :param action: The P value of the P controller
        :return: reward for reaching the next obs
        """
        observation = self.gym_env.reset()
        done = False
        t = 0
        total_reward = 0

        while not done:
            f_t = self.get_features(observation)
            env_action = self.get_insulin(f_t, action)
            observation, reward, done, info = self.gym_env.step(
                env_action
            )
            t += 1
            total_reward += reward
            if t == MAX_TIMESTEPS:
                break
        self.terminal_state = True
        # print("Simglucose trial finished after {} timesteps with total reward {}".format(t, total_reward))
        return total_reward

    def get_features(self,observation):
        # Features = (Blood Glucose)
        features = [observation[0]]
        return features

    def get_insulin(self,f_t, P):
        # This is a simple P controller based on Blogg Glucose (BG)
        bg = f_t[0]
        
        diff = bg - TARGET

        insulin = P * diff
    
        # Clip insulin between allowed amount
        return max(MIN_INSULIN, min(MAX_INSULIN, insulin))

    def get_observation(self):
        """Get the current obs"""
        return self.state

    def update_position(self, action):
        """Helper function for transition() that updates the
        current position given an action

        :param action: A possible action at the current obs
        """
        

    def is_in_goal_state(self):
        """Check whether current obs is goal obs

        :return: True if obs is in goal obs, False if not
        """
        return self.state == self.num_states - 1

    def visualize(self):
        """Print out current obs information"""
        print_state = 0
        for y in range(self.size):
            for x in range(self.size):
                if print_state == self.state:
                    print("A", end="")
                else:
                    print("X", end="")
                print_state += 1
            print()
        print()
