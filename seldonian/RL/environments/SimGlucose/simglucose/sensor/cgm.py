# from .noise_gen import CGMNoiseGenerator
from .noise_gen import CGMNoise
import pandas as pd
import logging
from os import path

logger = logging.getLogger(__name__)

curr_path = path.abspath(path.join(path.dirname(__file__)))
SENSOR_PARA_FILE = path.join(curr_path, '..', 'params', 'sensor_params.csv')

class CGMSensor(object):
    def __init__(self, params, seed=None):
        self._counter = 0
        self._params = params
        self.name = params.Name
        self.sample_time = params.sample_time
        self.seed = seed
        self._last_CGM = 0

    @classmethod
    def withName(cls, name, **kwargs):
        sensor_params = pd.read_csv(SENSOR_PARA_FILE)
        params = sensor_params.loc[sensor_params.Name == name].squeeze()
        return cls(params, **kwargs)

    def measure(self, patient):
        if patient.t % self.sample_time == 0:
            BG = patient.observation.Gsub
            CGM = BG + next(self._noise_generator)
            CGM = max(CGM, self._params["min"])
            CGM = min(CGM, self._params["max"])
            self._last_CGM = CGM
            return CGM

        # Zero-Order Hold
        return self._last_CGM

    @property
    def seed(self):
        return self._seed

    @seed.setter
    def seed(self, seed):
        self._seed = seed
        self._noise_generator = CGMNoise(self._params, seed=seed)

    def reset(self):
        logger.debug('Resetting CGM sensor ...')

        # Original code from SimGlucose re-uses same seed
        # That makes noise across trajectories dependent
        # THerefore episodes are no longer independent
        # self._noise_generator = CGMNoise(self._params, seed=self.seed)

        self._counter += 1
        self._noise_generator = CGMNoise(self._params, seed=self.seed + self._counter)
        self._last_CGM = 0


if __name__ == '__main__':
    pass
