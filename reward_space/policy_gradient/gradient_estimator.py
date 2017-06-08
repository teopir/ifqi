import numpy as np
import numpy.linalg as la

class GradientEstimator(object):
    def __init__(self, dataset):
        self.dataset = dataset
        self.n_samples = dataset.shape[0]
        self.n_episodes = dataset[:, -1].sum()

    def estimate_return(self):
        return 1. / self.n_episodes * np.dot(self.dataset[:, 2],
                                             self.dataset[:, 4])

    def estimate_gradient(self,
                          reward_features,
                          policy_gradient,
                          use_baseline=False,
                          state_space=None,
                          action_space=None):
        pass

    def estimate_hessian(self,
                         reward_features,
                         policy_gradient,
                         policy_hessian,
                         use_baseline=False,
                         state_space=None,
                         action_space=None):
        pass

class MaximumLikelihoodEstimator(GradientEstimator):

    def __init__(self, dataset):
        self.dataset = dataset
        self.n_samples = dataset.shape[0]
        self.n_episodes = int(dataset[:, -1].sum())

    def estimate_gradient(self,
                          reward_features,
                          policy_gradient,
                          use_baseline=False,
                          state_space=None,
                          action_space=None):
        if np.ndim(reward_features) == 1:
            reward_features = reward_features[:, np.newaxis]

        n_features = reward_features.shape[1]
        n_params = policy_gradient.shape[1]
        if state_space is not None and action_space is not None:
            n_states, n_actions = len(state_space), len(action_space)

        episode_reward_features = np.zeros((self.n_episodes, n_features))
        episode_gradient = np.zeros((self.n_episodes, n_params))

        if use_baseline:
            baseline = self._estimate_gradient_baseline(reward_features,
                                                        policy_gradient,
                                                        state_space,
                                                        action_space)
        else:
            baseline = 0.

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
            episode_gradient[episode, :] += policy_gradient[index, :].squeeze()

            if self.dataset[i, -1] == 1:
                episode += 1

            i += 1

        episode_reward_features_baseline = episode_reward_features - baseline

        return_gradient = 1. / self.n_episodes * np.dot(
                                            episode_reward_features_baseline.T,
                                            episode_gradient)

        return return_gradient

    def _estimate_gradient_baseline(self,
                                    reward_features,
                                    policy_gradient,
                                    state_space=None,
                                    action_space=None):

        if np.ndim(reward_features) == 1:
            reward_features = reward_features[:, np.newaxis]

        n_features = reward_features.shape[1]
        n_params = policy_gradient.shape[1]
        if state_space is not None and action_space is not None:
            n_states, n_actions = len(state_space), len(action_space)

        episode_reward_features = np.zeros((self.n_episodes, n_features))
        episode_gradient = np.zeros((self.n_episodes, n_params))
        numerator = denominator = 0.

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
            episode_gradient[episode, :] += policy_gradient[index, :].squeeze()

            if self.dataset[i, -1] == 1:
                vectorized_gradient = episode_gradient[episode, :].ravel()
                numerator += episode_reward_features[episode, :] * la.norm(
                    vectorized_gradient) ** 2
                denominator += la.norm(vectorized_gradient) ** 2
                episode += 1

            i += 1

        baseline = numerator / denominator

        return baseline

    def estimate_hessian(self,
                         reward_features,
                         policy_gradient,
                         policy_hessian,
                         use_baseline=False,
                         state_space=None,
                         action_space=None):

        if np.ndim(reward_features) == 1:
            reward_features = reward_features[:, np.newaxis]

        n_features = reward_features.shape[1]
        n_params = policy_hessian.shape[1]
        if state_space is not None and action_space is not None:
            n_states, n_actions = len(state_space), len(action_space)

        episode_reward_features = np.zeros((self.n_episodes, n_features))

        episode_policy_gradient = np.zeros((self.n_episodes, n_params))
        episode_policy_hessian = np.zeros((self.n_episodes, n_params, n_params))


        if use_baseline:
            baseline = self._estimate_hessian_baseline(reward_features,
                                                       policy_gradient,
                                                       policy_hessian,
                                                       state_space,
                                                       action_space)
        else:
            baseline = 0.

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
            episode_policy_gradient[episode, :] += policy_gradient[index, :].squeeze()
            episode_policy_hessian[episode, :, :] += policy_hessian[index, :, :].squeeze()

            if self.dataset[i, -1] == 1:
                episode += 1

            i += 1
        episode_hessian = episode_policy_gradient.reshape(self.n_episodes, 1,
                                                          n_params) * episode_policy_gradient.reshape(
            self.n_episodes, n_params, 1) + episode_policy_hessian

        episode_reward_features_baseline = episode_reward_features - baseline

        return_hessians = 1. / self.n_episodes * np.tensordot(
            episode_reward_features_baseline.T,
            episode_hessian, axes=1)

        return return_hessians

    def _estimate_hessian_baseline(self,
                                   reward_features,
                                   policy_gradient,
                                   policy_hessian,
                                   state_space=None,
                                   action_space=None):

        if np.ndim(reward_features) == 1:
            reward_features = reward_features[:, np.newaxis]

        n_features = reward_features.shape[1]
        n_params = policy_gradient.shape[1]
        if state_space is not None and action_space is not None:
            n_states, n_actions = len(state_space), len(action_space)

        episode_reward_features = np.zeros((self.n_episodes, n_features))
        episode_policy_gradient = np.zeros((self.n_episodes, n_params))
        episode_policy_hessian = np.zeros((self.n_episodes, n_params, n_params))

        numerator = denominator = 0.

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
            episode_policy_gradient[episode, :] += policy_gradient[index, :].squeeze()
            episode_policy_hessian[episode, :, :] += policy_hessian[index, :, :].squeeze()

            if self.dataset[i, -1] == 1:
                episode_hessian = episode_policy_gradient.reshape(self.n_episodes, 1,
                                                          n_params) * episode_policy_gradient.reshape(
            self.n_episodes, n_params, 1) + episode_policy_hessian
                vectorized_hessian = episode_hessian.ravel()
                numerator += episode_reward_features[episode, :] * la.norm(
                    vectorized_hessian) ** 2
                denominator += la.norm(vectorized_hessian) ** 2
                episode += 1

            i += 1

        baseline = numerator / denominator

        return baseline