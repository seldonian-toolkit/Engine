from utils import *

class RL_model: #consist of agent, env, and evaluator
    def __init__(self, agent, env, evaluator):
        self.agent = agent
        self.env = env
        self.evaluator = evaluator

    def get_probs_from_observations_and_actions(self, new_params, observations, actions):
        self.agent.set_new_params(new_params)
        num_probs = len(observations)
        if num_probs != len(actions):
            error(f"different number of observations ({observations}) and actions ({actions})")

        probs = [self.get_prob_this_action(observations[index], actions[index]) for index in range(num_probs)]

        return np.array(probs)

    def get_prob_this_action(self, observation, action):
        return self.agent.get_prob_this_action(observation, action)

    def sample_from_statistic(self, statistic_name,model,theta,data_dict):
        return self.evaluator.sample_from_statistic(statistic_name,self,theta,data_dict)