import numpy as np

class GridworldEnvrionment():
    def __init__(self):
        self.states = np.arange(9,dtype='int') # 0-8
        self.actions = np.array([0,1,2,3]) # U,D,L,R

        # initialize parameter weights to all zeros
        self.param_weights = np.zeros(
            (len(self.states)-1)*len(self.actions)
            ) 