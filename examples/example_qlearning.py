import gym
import pandas as pd
import numpy as np
import random


class QLearner(object):
    def __init__(self,
                 num_states=100,
                 discrete_actions=list([0, 1, 2]),
                 alpha=0.2,
                 gamma=0.9,
                 random_action_rate=0.5,
                 random_action_decay_rate=0.99):
        self.num_states = num_states
        # self.num_actions = num_actions
        self.discrete_actions = discrete_actions
        self.num_actions = len(discrete_actions)
        self.alpha = alpha
        self.gamma = gamma
        self.random_action_rate = random_action_rate
        self.random_action_decay_rate = random_action_decay_rate
        self.state = 0
        self.action = 0
        self.qtable = np.random.uniform(low=-1, high=1, size=(num_states, self.num_actions))

    def set_initial_state(self, state):
        """
        @summary: Sets the initial state and returns an action
        @param state: The initial state
        @returns: The selected action
        """
        self.state = state
        self.action = self.qtable[state].argsort()[-1]
        real_action = self.discrete_actions[self.action]
        return np.array([real_action])

    def move(self, state_prime, reward):
        """
        @summary: Moves to the given state with given reward and returns action
        @param state_prime: The new state
        @param reward: The reward
        @returns: The selected action
        """
        alpha = self.alpha
        gamma = self.gamma
        state = self.state
        action = self.action
        qtable = self.qtable

        choose_random_action = (1 - self.random_action_rate) <= np.random.uniform(0, 1)

        if choose_random_action:
            action_prime = random.randint(0, self.num_actions - 1)
        else:
            action_prime = self.qtable[state_prime].argsort()[-1]

        self.random_action_rate *= self.random_action_decay_rate

        qtable[state, action] = (1 - alpha) * qtable[state, action] + alpha * (
        reward + gamma * qtable[state_prime, action_prime])

        self.state = state_prime
        self.action = action_prime
        real_action = self.discrete_actions[self.action]

        return np.array([real_action])


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

    def build_state(features):
        return int("".join(map(lambda feature: str(int(feature)), features)))

    def to_bin(value, bins):
        return np.digitize(x=[value], bins=bins)[0]

    learner = QLearner(num_states=10 ** number_of_features,
                       discrete_actions=[i for i in range(env.action_space.n)],
                       alpha=0.2,
                       gamma=1,
                       random_action_rate=0.5,
                       random_action_decay_rate=0.99)

    for episode in range(50000):
        cart_position, pole_angle, cart_velocity, angle_rate_of_change = observation
        state = build_state([to_bin(cart_position, cart_position_bins),
                             to_bin(pole_angle, pole_angle_bins),
                             to_bin(cart_velocity, cart_velocity_bins),
                             to_bin(angle_rate_of_change, angle_rate_bins)])
        action = learner.set_initial_state(state)

        for step in xrange(max_number_of_steps - 1):
            observation, reward, done, info = env.step(action)

            cart_position, pole_angle, cart_velocity, angle_rate_of_change = observation

            state_prime = build_state([to_bin(cart_position, cart_position_bins),
                                       to_bin(pole_angle, pole_angle_bins),
                                       to_bin(cart_velocity, cart_velocity_bins),
                                       to_bin(angle_rate_of_change, angle_rate_bins)])

            if done:
                reward = -200
                observation = env.reset()

            action = learner.move(state_prime, reward)

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
    env = Monitor(env, experiment_filename, force=True)
    observation = env.reset()

    goal_average_steps = -160
    max_number_of_steps = 200
    number_of_iterations_to_average = 100

    number_of_features = env.observation_space.shape[0]
    last_time_steps = np.ndarray(0)

    sin_bins = pd.cut([-1., 1.], bins=10, retbins=True)[1][1:-1]
    cos_bins = pd.cut([-1., 1.], bins=10, retbins=True)[1][1:-1]
    speed_bins = pd.cut([-8, 8], bins=10, retbins=True)[1][1:-1]

    num_states = (len(sin_bins) + 1) * (len(cos_bins) + 1) * (len(speed_bins) + 1)

    def build_state(features):
        return int("".join(map(lambda feature: str(int(feature)), features)))

    def to_bin(value, bins):
        return np.digitize(x=[value], bins=bins)[0]

    learner = QLearner(num_states=num_states,
                       discrete_actions=np.linspace(-2., 2., 10).tolist(),
                       alpha=0.2,
                       gamma=1,
                       random_action_rate=0.5,
                       random_action_decay_rate=0.99)

    cum_reward = 0.
    for episode in range(5000):
        #env.render()
        sin_position, cos_pos, pole_speed = observation
        state = build_state([to_bin(sin_position, sin_bins),
                             to_bin(cos_pos, cos_bins),
                             to_bin(pole_speed, speed_bins)])
        action = learner.set_initial_state(state)

        print('starting')
        done = False
        for step in range(max_number_of_steps - 1):
            observation, reward, done, info = env.step(action)

            sin_position, cos_pos, pole_speed = observation

            state_prime = build_state([to_bin(sin_position, sin_bins),
                                       to_bin(cos_pos, cos_bins),
                                       to_bin(pole_speed, speed_bins)])
            cum_reward += reward

            action = learner.move(state_prime, reward)

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
