from seldonian.RL.environments.SimGlucose.simglucose.controller.base import Controller, Action
import numpy as np
import pandas as pd
import logging
from os import path

curr_path = path.abspath(path.join(path.dirname(__file__)))
INSULIN_PUMP_PARA_FILE = path.join(curr_path, '..', 'params', 'pump_params.csv')

logger = logging.getLogger(__name__)
CONTROL_QUEST = path.join(curr_path, '..', 'params', 'Quest.csv')
PATIENT_PARA_FILE = path.join(curr_path, '..', 'params', 'vpatient_params.csv')

quest = pd.read_csv(CONTROL_QUEST)
q = quest[quest.Name.str.match('adolescent#002')]
print(q)

patient_params = pd.read_csv(PATIENT_PARA_FILE)
params = patient_params[patient_params.Name.str.match('adolescent#002')]
u2ss = params.u2ss.values
BW = params.BW.values
print(u2ss, BW)

u2ss = patient_params.u2ss.values
BW = patient_params.BW.values
print(min(u2ss), max(u2ss), min(BW), max(BW))

class BBController(Controller):
    def __init__(self, target=140):
        self.quest = pd.read_csv(CONTROL_QUEST)
        self.patient_params = pd.read_csv(
            PATIENT_PARA_FILE)
        self.target = target

    def policy(self, observation, reward, done, **kwargs):
        sample_time = kwargs.get('sample_time', 1)
        pname = kwargs.get('patient_name')
        meal = kwargs.get('meal')

        action = self._bb_policy(
            pname,
            meal,
            observation.CGM,
            sample_time)
        return action

    def _bb_policy(self, name, meal, glucose, env_sample_time):
        if any(self.quest.Name.str.match(name)):
            q = self.quest[self.quest.Name.str.match(name)]
            params = self.patient_params[self.patient_params.Name.str.match(name)]
            u2ss = np.asscalar(params.u2ss.values)
            BW = np.asscalar(params.BW.values)
        else:
            q = pd.DataFrame([['Average', 1 / 15, 1 / 50, 50, 30]],
                             columns=['Name', 'CR', 'CF', 'TDI', 'Age'])
            u2ss = 1.43
            BW = 57.0

        basal = u2ss * BW / 6000
        # basal = 0.0093
        if meal > 0:
            logger.info('Calculating bolus ...')
            logger.debug('glucose = {}'.format(glucose))
            # bolus = np.asscalar(meal / q.CR.values + (glucose > 150)
            #                     * (glucose - self.target) / q.CF.values)
            # bolus = np.asscalar(meal /23 + (glucose > 150) * (glucose - self.target) / 33.5)
            bolus = np.asscalar(meal /4 + (glucose > 150) * (glucose - self.target) / 12)

            # bolus = np.asscalar(meal / 30 + (glucose > 150) * (glucose - self.target) / 25)
            # bolus = np.asscalar(meal /6 + (glucose > 150) * (glucose - self.target) / 25)
            # bolus = np.asscalar(meal /9 + (glucose > 150) * (glucose - self.target) / 25)
            # bolus = np.asscalar(meal /12 + (glucose > 150) * (glucose - self.target) / 25)
            # bolus = np.asscalar(meal /15 + (glucose > 150) * (glucose - self.target) / 25)
            # bolus = np.asscalar(meal /18 + (glucose > 150) * (glucose - self.target) / 25)
            # bolus = np.asscalar(meal /30 + (glucose > 150) * (glucose - self.target) / 50)

            # bolus = np.asscalar(meal / q.CR.values + (glucose - self.target) / q.CF.values)
        else:
            bolus = 0

        bolus = bolus / env_sample_time
        # action = Action(basal=basal, bolus=bolus)
        action = Action(basal=0, bolus=bolus)
        # action = Action(basal=basal, bolus=0)
        return action

    def reset(self):
        pass
