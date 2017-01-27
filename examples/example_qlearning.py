import gym
import pandas as pd
import numpy as np
import random
from ifqi.algorithms import QLearner, Binning


def cart_pole_with_qlearning():
    from gym.wrappers import Monitor
    env = gym.make('CartPole-v0')
    experiment_filename = './cartpole-experiment-1'
    env = Monitor(env, experiment_filename, force=True)
    observation = env.reset()

    goal_average_steps = 195
    max_number_of_steps = 200
    number_of_iterations_to_average = 100

    number_of_features = env.observation_space.shape[0]
    last_time_steps = np.ndarray(0)

    cart_position_bins = pd.cut([-2.4, 2.4], bins=10, retbins=True)[1][1:-1]
    pole_angle_bins = pd.cut([-2, 2], bins=10, retbins=True)[1][1:-1]
    cart_velocity_bins = pd.cut([-1, 1], bins=10, retbins=True)[1][1:-1]
    angle_rate_bins = pd.cut([-3.5, 3.5], bins=10, retbins=True)[1][1:-1]

    learner = QLearner(state_discretization=Binning([[-2.4, 2.4], [-2, 2], [-1., 1], [-3.5, 3.5]], [10] * 4),
                       discrete_actions=[i for i in range(env.action_space.n)],
                       alpha=0.2,
                       gamma=1,
                       random_action_rate=0.5,
                       random_action_decay_rate=0.99)

    for episode in range(50000):
        action = learner.set_initial_state(observation)

        for step in range(max_number_of_steps - 1):
            observation, reward, done, info = env.step(action)

            if done:
                reward = -200
                observation = env.reset()

            action = learner.move(observation, reward)

            if done:
                last_time_steps = np.append(last_time_steps, [int(step + 1)])
                if len(last_time_steps) > number_of_iterations_to_average:
                    last_time_steps = np.delete(last_time_steps, 0)
                break

        if last_time_steps.mean() > goal_average_steps:
            print "Goal reached!"
            print "Episodes before solve: ", episode + 1
            print u"Best 100-episode performance {} {} {}".format(last_time_steps.max(),
                                                                  unichr(177),  # plus minus sign
                                                                  last_time_steps.std())
            break

    env.close()


def pendulum_with_qlearning():
    from gym.wrappers import Monitor
    env = gym.make('Pendulum-v0')
    experiment_filename = './pendulum-experiment-1'
    #env = Monitor(env, experiment_filename, force=True)
    observation = env.reset()

    goal_average_steps = -140
    max_number_of_steps = 200
    number_of_iterations_to_average = 100

    last_time_steps = np.ndarray(0)

    learner = QLearner(state_discretization=Binning([[-1., 1.], [-1., 1.], [-8., 8]], [10, 10, 40]),
                       discrete_actions=np.linspace(-2., 2., 10).tolist(),
                       alpha=0.2,
                       gamma=1,
                       random_action_rate=0.9,
                       random_action_decay_rate=0.99,
                       action_type="numpy")

    cum_reward = 0.
    for episode in range(5000):
        cum_reward = 0.
        env.reset()
        action = learner.set_initial_state(observation)
        done = False
        for step in range(max_number_of_steps - 1):
            observation, reward, done, info = env.step(action)

            cum_reward += reward

            if not done or step < max_number_of_steps -2:

                action = learner.move(observation, reward)

            if done:
                print(cum_reward)
                last_time_steps = np.append(last_time_steps, [cum_reward])
                if len(last_time_steps) > number_of_iterations_to_average:
                    last_time_steps = np.delete(last_time_steps, 0)
                observation = env.reset()
                cum_reward = 0.
                break

                # if last_time_steps.mean() > goal_average_steps:
                #     print "Goal reached!"
                #     print "Episodes before solve: ", episode + 1
                #     print u"Best 100-episode performance {} {} {}".format(last_time_steps.max(),
                #                                                           unichr(177),  # plus minus sign
                #                                                           last_time_steps.std())
                #     break
        if not done:
            print(cum_reward)

    env.close()


if __name__ == "__main__":
    random.seed(0)
    # cart_pole_with_qlearning()
    pendulum_with_qlearning()
