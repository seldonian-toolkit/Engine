from seldonian.RL.RL_runner import *
from seldonian.RL.hyperparams_and_settings import *

def print_results(results):
    for trial_num, trial in enumerate(results):
        print(f"\n\n\ntrial {trial_num}:")
        for episode_num, episode in enumerate(trial):
            print(f"\nepisode {episode_num}:")
            print(episode)

if __name__ == "__main__":
    hyperparameter_and_setting_dict = define_hyperparameter_and_setting_dict()
    results = run_all_trials(hyperparameter_and_setting_dict)
    print("returns:")
    for trial_num in range(hyperparameter_and_setting_dict["num_trials"]):
        print(f"\ntrial number {trial_num}")
        for episode in results[trial_num]:
            print(sum(episode.rewards))
