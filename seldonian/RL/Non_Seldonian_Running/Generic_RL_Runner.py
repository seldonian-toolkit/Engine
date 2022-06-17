from seldonian.RL.RL_runner import *

def define_hyperparameter_and_setting_dict():
    the_dict = {}
    the_dict["env"] = "gridworld"
    the_dict["agent"] = "discrete_random"
    the_dict["num_trials"] = 2
    the_dict["num_episodes"] = 3
    the_dict["vis"] = False
    return the_dict

if __name__ == "__main__":
    hyperparameter_and_setting_dict = define_hyperparameter_and_setting_dict()
    returns = run_all_trials(hyperparameter_and_setting_dict)
    print(returns)