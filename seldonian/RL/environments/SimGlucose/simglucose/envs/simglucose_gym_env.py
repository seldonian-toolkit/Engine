from seldonian.RL.environments.SimGlucose.simglucose.simulation.env import T1DSimEnv as _T1DSimEnv
from seldonian.RL.environments.SimGlucose.simglucose.patient.t1dpatient import T1DPatient
from seldonian.RL.environments.SimGlucose.simglucose.sensor.cgm import CGMSensor
from seldonian.RL.environments.SimGlucose.simglucose.actuator.pump import InsulinPump
from seldonian.RL.environments.SimGlucose.simglucose.simulation.scenario_gen import RandomScenario, WeightScenario
from seldonian.RL.environments.SimGlucose.simglucose.simulation.env import risk_diff, neg_risk
from seldonian.RL.environments.SimGlucose.simglucose.controller.base import Action
import pandas as pd
import numpy as np
import gym
from gym import error, spaces, utils
from gym.utils import seeding
from datetime import datetime
from Uno_Space import Space
import matplotlib.pyplot as plt

from os import path
curr_path = path.abspath(path.join(path.dirname(__file__)))
PATIENT_PARA_FILE = path.join(curr_path, '..', 'params', 'vpatient_params.csv')



class T1DSimEnv_original(gym.Env):
    '''
    A wrapper of simglucose.simulation.env.T1DSimEnv to support gym API
    '''
    metadata = {'render.modes': ['human']}

    def __init__(self, patient_name=None, reward_fun=None):
        '''
        patient_name must be 'adolescent#001' to 'adolescent#010',
        or 'adult#001' to 'adult#010', or 'child#001' to 'child#010'
        '''
        seeds = [0, 0, 0, 0, 0] #self._seed()
        # have to hard code the patient_name, gym has some interesting
        # error when choosing the patient
        if patient_name is None:
            patient_name = 'adolescent#001'
        print(patient_name)
        patient = T1DPatient.withName(patient_name)
        sensor = CGMSensor.withName('Dexcom', seed=seeds[1])
        # sensor = CGMSensor.withName('Navigator', seed=seeds[1])
        # sensor = CGMSensor.withName('GuardianRT', seed=seeds[1])
        hour = 0 #self.np_random.randint(low=0.0, high=24.0)
        start_time = datetime(2018, 1, 1, hour, 0, 0)
        scenario = RandomScenario(start_time=start_time, seed=seeds[2])
        pump = InsulinPump.withName('Insulet')
        self.env = _T1DSimEnv(patient, sensor, pump, scenario)
        self.reward_fun = reward_fun

    @staticmethod
    def pick_patient():
        # TODO: cannot be used to pick patient at the env constructing space
        # for now
        patient_params = pd.read_csv(PATIENT_PARA_FILE)
        while True:
            print('Select patient:')
            for j in range(len(patient_params)):
                print('[{0}] {1}'.format(j + 1, patient_params['Name'][j]))
            try:
                select = int(input('>>> '))
            except ValueError:
                print('Please input a number.')
                continue

            if select < 1 or select > len(patient_params):
                print('Please input 1 to {}'.format(len(patient_params)))
                continue

            return select

    def _step(self, action):
        # This gym only controls basal insulin
        act = Action(basal=action, bolus=0)
        if self.reward_fun is None:
            return self.env.step(act)
        else:
            return self.env.step(act, reward_fun=self.reward_fun)

    def _reset(self):
        obs, _, _, _ = self.env.reset()
        return obs

    def _seed(self, seed=None):
        self.np_random, seed1 = seeding.np_random(seed=seed)
        # Derive a random seed. This gets passed as a uint, but gets
        # checked as an int elsewhere, so we need to keep it below
        # 2**31.
        seed2 = seeding.hash_seed(seed1 + 1) % 2**31
        seed3 = seeding.hash_seed(seed2 + 1) % 2**31
        return [seed1, seed2, seed3]

    def _render(self, mode='human', close=False):
        self.env.render(close=close)

    @property
    def action_space(self):
        ub = self.env.pump._params['max_basal']
        return spaces.Box(low=0, high=ub, shape=(1,))

    @property
    def observation_space(self):
        return spaces.Box(low=0, high=np.inf, shape=(1,))





class T1DSimEnv(gym.Env):
    '''
    A wrapper of simglucose.simulation.env.T1DSimEnv to support gym API
    '''
    metadata = {'render.modes': ['human']}

    def __init__(self, patient_name=None, reward_fun=neg_risk, seed=0):
        '''
        patient_name must be 'adolescent#001' to 'adolescent#010',
        or 'adult#001' to 'adult#010', or 'child#001' to 'child#010'
        '''

        self._gym_disable_underscore_compat = True

        seeds = [0, 0, 0, 0, 0] #self._seed()

        patient_name_a = 'adolescent#003'

        patient_a = T1DPatient.withName(patient_name_a)

        # sensor = CGMSensor.withName('Navigator', seed=seeds[1])    # Sample frequency = 1 min
        sensor = CGMSensor.withName('Dexcom', seed=seed)# seed=seeds[1])    # Sample frequency = 3 min
        # sensor = CGMSensor.withName('GuardianRT', seed=seeds[1])  # Sample frequency = 5 min

        pump = InsulinPump.withName('Insulet')

        hour = 0  #self.np_random.randint(low=0.0, high=24.0)
        start_time = datetime(2018, 1, 1, hour, 0, 0)
        scenario = RandomScenario(start_time=start_time, seed=seed)#, seed=seeds[2])
        # scenario = WeightScenario(weight=patient._params.BW, start_time=start_time, seed=seeds[2])

        self.env = _T1DSimEnv(patient_a, sensor, pump, scenario)
        self.reward_fun = reward_fun
        self.target = 140

        # CR and CF lower and upper bound
        self.lb = np.array([3, 5])
        self.ub = np.array([30, 50])

        self.max_horizon = 1
        self.min_reward = -15
        self.max_reward = +15

    @staticmethod
    def pick_patient():
        # TODO: cannot be used to pick patient at the env constructing space for now
        patient_params = pd.read_csv(PATIENT_PARA_FILE)
        while True:
            print('Select patient:')
            for j in range(len(patient_params)):
                print('[{0}] {1}'.format(j + 1, patient_params['Name'][j]))
            try:
                select = int(input('>>> '))
            except ValueError:
                print('Please input a number.')
                continue

            if select < 1 or select > len(patient_params):
                print('Please input 1 to {}'.format(len(patient_params)))
                continue

            return select

    def step(self, action):
        # Goal is to estimate the correct CR and CF value for the patient
        CR, CF = action

        # Clip them to be within the range
        CR = np.clip(CR, self.lb[0], self.ub[0])
        CF = np.clip(CF, self.lb[1], self.ub[1])

        basal = 0

        obs, r, done, info = self.all_vars
        total_r = 0
        ctr = 0
        # temp = []
        while not done:
            meal = info['meal']
            glucose = obs[0]

            bolus = 0
            # Basal-Bolus controller
            # Note: Value of Bolus gets clipped to the desired range in the simulator
            if meal > 0:
                bolus = meal / CR + (glucose > 150) * (glucose - self.target) / CF

            # Clip bolus to be positive always
            if bolus < 0:
                bolus = 0

            # This gym only controls bolus insulin
            # Divide bolus by sample time because this action will be repeated 'sample time' times in the simulator
            bolus = bolus / info['sample_time']
            act = Action(basal=basal, bolus=bolus)
            obs, r, done, info = self.env.step(act, reward_fun=self.reward_fun)

            total_r += r
            ctr += 1

        reward = (total_r/ctr + 26.5) * 2          # makes the return _roughly_ normalized to [-10, 10]
        reward = np.clip(reward, self.min_reward, self.max_reward)

        return [1], reward, done, info

    def reset(self):
        self.all_vars = self.env.reset()
        obs, _, _, _ = self.all_vars
        return [1]

    def seed(self, seed=None):
        self.np_random, seed1 = seeding.np_random(seed=seed)
        # Derive a random seed. This gets passed as a uint, but gets
        # checked as an int elsewhere, so we need to keep it below
        # 2**31.
        seed2 = seeding.hash_seed(seed1 + 1) % 2**31
        seed3 = seeding.hash_seed(seed2 + 1) % 2**31
        return [seed1, seed2, seed3]

    def _render(self, mode='human', close=False):
        self.env.render(close=close)

    @property
    def action_space(self):
        return spaces.Box(low=self.lb, high=self.ub)

    @property
    def observation_space(self):
        return spaces.Box(low=0, high=1, shape=(1,))





class T1DSimEnv_discrete(gym.Env):
    '''
    A wrapper of simglucose.simulation.env.T1DSimEnv to support gym API
    '''
    metadata = {'render.modes': ['human']}

    def __init__(self, patient_name=None, reward_fun=neg_risk, seed=0):
        '''
        patient_name must be 'adolescent#001' to 'adolescent#010',
        or 'adult#001' to 'adult#010', or 'child#001' to 'child#010'
        '''

        self._gym_disable_underscore_compat = True

        seeds = [0, 0, 0, 0, 0] #self._seed()

        patient_name_a = 'adolescent#003'

        patient_a = T1DPatient.withName(patient_name_a)

        # sensor = CGMSensor.withName('Navigator', seed=seeds[1])    # Sample frequency = 1 min
        sensor = CGMSensor.withName('Dexcom', seed=seed)# seed=seeds[1])    # Sample frequency = 3 min
        # sensor = CGMSensor.withName('GuardianRT', seed=seeds[1])  # Sample frequency = 5 min

        pump = InsulinPump.withName('Insulet')

        hour = 0  #self.np_random.randint(low=0.0, high=24.0)
        start_time = datetime(2018, 1, 1, hour, 0, 0)
        scenario = RandomScenario(start_time=start_time, seed=seed)#, seed=seeds[2])
        # scenario = WeightScenario(weight=patient._params.BW, start_time=start_time, seed=seeds[2])

        self.env = _T1DSimEnv(patient_a, sensor, pump, scenario)
        self.reward_fun = reward_fun
        self.target = 140

        # CR and CF lower and upper bound
        self.lb = np.array([3, 5])
        self.ub = np.array([30, 50])

        self.max_horizon = 1
        self.min_reward = -15
        self.max_reward = +15

        # Discretizer
        self.bins_per_dim = 4
        self.n_max_actions = self.bins_per_dim ** 2

    def mapper(self, val):
        # val = val - 1
        x1 = val // self.bins_per_dim
        y1 = val % self.bins_per_dim

        x2 = (x1 / (self.bins_per_dim - 1)) * (self.ub[0] - self.lb[0]) + self.lb[0]
        y2 = (y1 / (self.bins_per_dim - 1)) * (self.ub[1] - self.lb[1]) + self.lb[1]

        return x2, y2

    @staticmethod
    def pick_patient():
        # TODO: cannot be used to pick patient at the env constructing space for now
        patient_params = pd.read_csv(PATIENT_PARA_FILE)
        while True:
            print('Select patient:')
            for j in range(len(patient_params)):
                print('[{0}] {1}'.format(j + 1, patient_params['Name'][j]))
            try:
                select = int(input('>>> '))
            except ValueError:
                print('Please input a number.')
                continue

            if select < 1 or select > len(patient_params):
                print('Please input 1 to {}'.format(len(patient_params)))
                continue

            return select

    def step(self, action):
        # Goal is to estimate the correct CR and CF value for the patient
        assert 0 <= action < self.n_max_actions

        CR, CF = self.mapper(action)

        # Clip them to be within the range
        # CR = np.clip(CR, self.lb[0], self.ub[0])
        # CF = np.clip(CF, self.lb[1], self.ub[1])
        #
        basal = 0

        obs, r, done, info = self.all_vars
        total_r = 0
        ctr = 0
        # temp = []
        while not done:
            meal = info['meal']
            glucose = obs[0]

            bolus = 0
            # Basal-Bolus controller
            # Note: Value of Bolus gets clipped to the desired range in the simulator
            if meal > 0:
                bolus = meal / CR + (glucose > 150) * (glucose - self.target) / CF

            # Clip bolus to be positive always
            if bolus < 0:
                bolus = 0

            # This gym only controls bolus insulin
            # Divide bolus by sample time because this action will be repeated 'sample time' times in the simulator
            bolus = bolus / info['sample_time']
            act = Action(basal=basal, bolus=bolus)
            obs, r, done, info = self.env.step(act, reward_fun=self.reward_fun)

            total_r += r
            ctr += 1

        reward = (total_r/ctr + 26.5) * 2          # makes the return _roughly_ normalized to [-10, 10]
        reward = np.clip(reward, self.min_reward, self.max_reward)

        return [1], reward, done, info

    def reset(self):
        self.all_vars = self.env.reset()
        obs, _, _, _ = self.all_vars
        return [1]

    def seed(self, seed=None):
        self.np_random, seed1 = seeding.np_random(seed=seed)
        # Derive a random seed. This gets passed as a uint, but gets
        # checked as an int elsewhere, so we need to keep it below
        # 2**31.
        seed2 = seeding.hash_seed(seed1 + 1) % 2**31
        seed3 = seeding.hash_seed(seed2 + 1) % 2**31
        return [seed1, seed2, seed3]

    def _render(self, mode='human', close=False):
        self.env.render(close=close)

    @property
    def action_space(self):
        return Space(size=self.n_max_actions)
        # return spaces.Box(low=self.lb, high=self.ub)

    @property
    def observation_space(self):
        return spaces.Box(low=0, high=1, shape=(1,))

