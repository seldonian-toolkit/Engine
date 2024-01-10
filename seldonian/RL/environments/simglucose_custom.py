from itertools import product

from seldonian.RL.environments.Environment import *
from seldonian.RL.Env_Description.Env_Description import *

import gym
from gym.envs.registration import register
from simglucose.controller.bolus_only_controller import BolusController


class SimglucoseCustomEnv(Environment):
    def __init__(self, bb_crmin, bb_crmax, bb_cfmin, bb_cfmax):
        """
        A custom Simglucose environment that runs simglucose
        simulations given cr,cf values of a bolus calculator.
        See this example for details: https://seldonian.cs.umass.edu/Tutorials/examples/diabetes/

        :param bb_crmin: The bounding box minimum value in CR space.
        :type bb_crmin: float
        :param bb_crmax: The bounding box maximum value in CR space.
        :type bb_crmax: float
        :param bb_cfmin: The bounding box minimum value in CF space.
        :type bb_cfmin: float
        :param bb_cfmax: The bounding box maximum value in CF space.
        :type bb_cfmax: float

        :ivar id: The RL Gym ID
        :ivar patient_name: The RL Gym patient name
        :ivar target_bg: The target blood glucose concentration
        :ivar num_states: Number of states, just one in this case.
        :ivar env_description: contains attributes describing the environment
        :vartype env_description: :py:class:`.Env_Description`
        :ivar gym_env: The RL Gym environment we are wrapping here.
        :ivar state: The current observation (fixed at state 0 always)
        :ivar terminal_state: Whether the terminal obs is occupied
        :vartype terminal_state: bool
        :ivar time: The current timestep
        :vartype time: int
        :ivar max_time: Maximum allowed timestep
        :vartype max_time: int
        :ivar low_cutoff_BG: Cutoff blood glucose concentration, below which
            the simulation will cease and return the minimum possible reward
            for the remainder of the timesteps.
        :ivar high_cutoff_BG: Cutoff blood glucose concentration, above which
            the simulation will cease and return the minimum possible reward
            for the remainder of the timesteps.
        :ivar min_primary_reward: The minimum possible reward of the primary
            reward function.
        :ivar min_secondary_reward: The minimum possible reward of the secondary
            reward function.
        """
        self.bb_crmin = bb_crmin
        self.bb_crmax = bb_crmax
        self.bb_cfmin = bb_cfmin
        self.bb_cfmax = bb_cfmax

        self.id = "simglucose-adult8-v0"
        self.patient_name = "adult#008"
        self.target_bg = 108  # mg/dL
        self.num_states = 1

        self.env_description = self.create_env_description()
        self.deregister_and_register()
        self.gym_env = gym.make(self.id)
        self.state = 0  # initial state and the only state
        self.terminal_state = False
        self.time = 0
        self.max_time = 480  # a day at 3 minutes per timestep
        self.low_cutoff_BG = 36
        self.high_cutoff_BG = 350
        self.min_primary_reward = (
            -1.5e-8 * np.abs(self.low_cutoff_BG - self.target_bg) ** 5
        )
        self.min_secondary_reward = self.min_primary_reward
        # vis is a flag for visual debugging during obs transitions
        self.vis = False

    def create_env_description(self):
        """Creates the environment description object.

        :return: Environment description for the obs and action spaces
        :rtype: :py:class:`.Env_Description`
        """
        observation_space = Discrete_Space(0, self.num_states - 1)
        action_space = Continuous_Space(
            bounds=np.array(
                [[self.bb_crmin, self.bb_crmax], [self.bb_cfmin, self.bb_cfmax]]
            )
        )
        return Env_Description(observation_space, action_space)

    def deregister_and_register(self):
        """Wrapper to run deregister and register back to back."""
        self.deregister()
        register(
            id=self.id,
            entry_point="simglucose.envs:CustomT1DSimEnv",
            kwargs={"patient_name": self.patient_name},
        )

    def deregister(self):
        """Remove this environment from the Gym registry"""
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
        :return: (primary_return, alt_return), the expected primary
            and secondary returns from the simglucose simulation.
        """
        observation, primary_reward, alt_reward, done, info = self.gym_env.reset()
        t = 0
        primary_rewards = self.min_primary_reward * np.ones(self.max_time)
        alt_rewards = self.min_secondary_reward * np.ones(self.max_time)

        cr, cf = action
        controller = BolusController(cr=cr, cf=cf, target=self.target_bg)
        while not done:
            env_action = controller.policy(
                observation,
                primary_reward,
                done,
                patient_name=self.patient_name,
                meal=info["meal"],
            )
            observation, primary_reward, alt_reward, done, info = self.gym_env.step(
                env_action
            )
            primary_rewards[t] = primary_reward
            alt_rewards[t] = alt_reward
            t += 1
            if t == self.max_time:
                break
        primary_return = np.mean(primary_rewards)
        alt_return = np.mean(alt_rewards)
        self.terminal_state = True

        return (primary_return, alt_return)

    def get_observation(self):
        """Get the current obs"""
        return self.state

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
