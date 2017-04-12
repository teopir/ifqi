import numpy as np
import numpy.linalg as la
import scipy.optimize as opt

class MaximumEntropyIRL(object):

    def __init__(self, dataset):
        self.dataset = dataset
        self.n_samples = dataset.shape[0]
        self.n_episodes = int(dataset[:, -1].sum())

    def _compute_episode_rewards(self, policy_gradient, reward_features, state_space=None, action_space=None):
        n_features = reward_features.shape[1]
        if state_space is not None and action_space is not None:
            n_states, n_actions = len(state_space), len(action_space)

        episode_reward_features = np.zeros((self.n_episodes, n_features))

        i = 0
        episode = 0
        while i < self.n_samples:
            if state_space is not None and action_space is not None:
                s = np.argwhere(state_space == self.dataset[i, 0])
                a = np.argwhere(action_space == self.dataset[i, 1])
                index = s * n_actions + a
            else:
                index = i

            d = self.dataset[i, 4]

            episode_reward_features[episode, :] += d * reward_features[index, :].squeeze()

            if self.dataset[i, -1] == 1:
                episode += 1

            i += 1

        return episode_reward_features

    def fit(self, policy_gradient, reward_features, state_space=None, action_space=None):
        episode_reward_features = self._compute_episode_rewards(policy_gradient, \
                        reward_features, state_space=None, action_space=None)

        def loss(x):
            episode_reward = np.dot(episode_reward_features, x)
            exp_episode_reward = np.exp(episode_reward)
            partition_function = np.sum(exp_episode_reward)
            episode_prob = exp_episode_reward / partition_function
            log_episode_prob = np.log(episode_prob)
            return -np.sum(log_episode_prob)

        w0 = np.ones(reward_features.shape[1])
        w0 /= la.norm(w0)
        res = opt.minimize(loss, w0, options={'disp': True})
        return res.x


