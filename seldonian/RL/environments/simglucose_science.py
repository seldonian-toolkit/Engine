from itertools import product

from seldonian.RL.environments.Environment import *
from seldonian.RL.Env_Description.Env_Description import *

import gym
from gym.envs.registration import register
from simglucose.controller.bolus_only_controller import BolusController

class SimglucoseScienceEnv(Environment):
    def __init__(self,
        bb_crmin,
        bb_crmax,
        bb_cfmin,
        bb_cfmax):
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
        self.bb_crmin = bb_crmin
        self.bb_crmax = bb_crmax
        self.bb_cfmin = bb_cfmin
        self.bb_cfmax = bb_cfmax

        self.id = "simglucose-adult3-v0"
        self.patient_name = "adult#003"
        self.target_bg = 108 # mg/dL
        self.num_states = 1

        self.env_description = self.create_env_description()
        self.deregister_and_register()
        self.gym_env = gym.make(self.id)
        self.state = 0 # initial state and the only state
        self.terminal_state = False
        self.time = 0
        self.max_time = 480 # a day at 3 minutes per timestep
        # vis is a flag for visual debugging during obs transitions
        self.vis = False

    def create_env_description(self):
        """Creates the environment description object.

        :param num_states: The number of states
        :return: Environment description for the obs and action spaces
        :rtype: :py:class:`.Env_Description`
        """
        observation_space = Discrete_Space(0, self.num_states - 1)
        action_space = Continuous_Space(bounds=np.array([[self.bb_crmin,self.bb_crmax],[self.bb_cfmin,self.bb_cfmax]]))
        return Env_Description(observation_space, action_space)
    
    def deregister_and_register(self):
        self.deregister()
        register(
            id=self.id,
            entry_point="simglucose.envs:ScienceT1DSimEnv",
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
        observation, primary_reward, alt_reward, done, info = self.gym_env.reset()
        t = 0
        primary_return = primary_reward
        alt_return = alt_reward
        cr, cf = action
        controller = BolusController(cr=cr, cf=cf, target=self.target_bg)
        while not done:
            env_action = controller.policy(observation, primary_reward, done, patient_name=self.patient_name, meal=info['meal'])
            observation, primary_reward, alt_reward, done, info = self.gym_env.step(env_action)
            primary_return += primary_reward
            alt_return += alt_reward
            # calculate the alt reward from the BG values manually
            t += 1
            if t == self.max_time:
                break
        primary_return/=(t+1)
        alt_return/=(t+1)
        self.terminal_state = True

        return (primary_return,alt_return)

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
