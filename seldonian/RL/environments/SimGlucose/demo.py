import gym
import numpy as np
from seldonian.RL.environments.SimGlucose.simglucose.controller.basal_bolus_ctrller import BBController
import time

def reward_fn(vals):
    return (np.mean(vals) + 26.5) * 2               # convert [-31, -22] into [-10, 10]

class Demo():
    def test_gym_BBC_agent(self):
        from gym.envs.registration import register
        register(
            id='simglucose-adolescent2-v0',
            entry_point='seldonian.RL.environments.SimGlucose.simglucose.envs:T1DSimEnv',
            kwargs={'patient_name': 'adolescent#003'}
        )

        env = gym.make('simglucose-adolescent2-v0')
        ctrller = BBController()

        reward = 0
        done = False
        info = {'sample_time': 5,
                'patient_name': 'adolescent#003',
                'meal': 0}

        observation = env.reset()
        total_risk = []
        for t in range(1500):
            # env.render(mode='human')
            # print(observation)
            # action = env.action_space.sample()
            ctrl_action = ctrller.policy(observation, reward, done, **info)
            # print(ctrl_action)
            action = ctrl_action.basal + ctrl_action.bolus
            # print(action)
            observation, reward, done, info = env.step(action)
            # observation, reward, done, info = env.step(10)
            total_risk.append(reward)
            if done:
                print("Episode finished after {} timesteps".format(t + 1))
                break

        print("Average risk: {}, Total risk: {}, New:{}".format(np.mean(total_risk), np.sum(total_risk), reward_fn(total_risk)))


if __name__ == '__main__':
    t = time.time()
    Demo().test_gym_BBC_agent()
    print("time taken: {}".format(time.time() - t))