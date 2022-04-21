import numpy as np

class GridworldEnvrionment():
    def __init__(self):
        self.states = list(range(9)) # 0-8
        self.actions = [0,1,2,3]
        # Define reward for each state in a dict
        self.reward_dict = {x:0 for x in self.states} # initialize
        self.reward_dict[7]=-1
        self.reward_dict[8]=1
        # Define environment dict containing 
        # which state each action/state combo results in
        # Will be defined s.t. environ[S][A] = S'
        self.environ_dict = {}
        self.environ_dict[0] = {0:0,1:3,2:0,3:1}
        self.environ_dict[1] = {0:1,1:4,2:0,3:2}
        self.environ_dict[2] = {0:2,1:5,2:1,3:2}
        self.environ_dict[3] = {0:0,1:6,2:3,3:4}
        self.environ_dict[4] = {0:1,1:7,2:3,3:5}
        self.environ_dict[5] = {0:2,1:8,2:4,3:5}
        self.environ_dict[6] = {0:3,1:6,2:6,3:7}
        self.environ_dict[7] = {0:4,1:7,2:6,3:8}
        self.environ_dict[8] = {0:5,1:8,2:7,3:8}
        
        self.initial_state = 0
        self.terminal_state = 8

        # initial transition dict (parameter weights)
        # to all zeros
        self.param_weights = np.zeros((len(self.states)-1)*len(self.actions))
        # self.transition_dict = {
        #     s:{a:0 for a in self.actions} for s in self.states} 