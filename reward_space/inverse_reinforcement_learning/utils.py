import numpy as np

def compute_feature_expectations(reward_features,
                                  trajectories,
                                  state_space=None,
                                  action_space=None):
    n_features = reward_features.shape[1]
    n_trajectories = int(trajectories[:, -1].sum())
    n_samples = trajectories.shape[0]

    if state_space is not None and action_space is not None:
        n_states, n_actions = len(state_space), len(action_space)

    feature_expectation = np.zeros((n_trajectories, n_features))

    i = 0
    trajectory = 0
    while i < n_samples:
        if state_space is not None and action_space is not None:
            s = np.argwhere(state_space == trajectories[i, 0])
            a = np.argwhere(action_space == trajectories[i, 1])
            index = s * n_actions + a
        else:
            index = i

        d = trajectories[i, 4]

        feature_expectation[trajectory, :] += d * reward_features[index, :].squeeze()

        if trajectories[i, -1] == 1:
            trajectory += 1

        i += 1

    return feature_expectation.sum(axis=0).ravel()