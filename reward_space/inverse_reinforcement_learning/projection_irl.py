from reward_space.inverse_reinforcement_learning.utils import *
from ifqi.evaluation import evaluation
import numpy.linalg as la
import numpy as np

class ProjectionIRL(object):

    def __init__(self,
                 mdp,
                 solver,
                 policy_init,
                 policy_expert,
                 reward_features,
                 n_episodes=10,
                 tol=0.,
                 state_space=None,
                 action_space=None):
        self.mdp = mdp
        self.solver = solver
        self.policy_init = policy_init
        self.policy_expert = policy_expert
        self.reward_features = reward_features
        self.n_episodes = n_episodes
        self.tol = tol
        self.state_space = state_space
        self.action_space = action_space

    def fit(self, verbose=False):
        trajectories = evaluation.collect_episodes(self.mdp, self.policy_expert, self.n_episodes)
        expert_feature_expectation = compute_feature_expectations(self.reward_features,
                                                           trajectories,
                                                           self.state_space,
                                                           self.action_space)

        trajectories = evaluation.collect_episodes(self.mdp, self.policy_init, self.n_episodes)
        feature_expectation = compute_feature_expectations(self.reward_features,
                                                           trajectories,
                                                           self.state_space,
                                                           self.action_space)

        w = expert_feature_expectation - feature_expectation
        mean_feature_expectation = feature_expectation

        margin = la.norm(w)

        while margin > self.tol:
            reward = np.dot(self.reward_features, w)
            optimal_policy = self.solver(self.mdp, reward)

            trajectories = evaluation.collect_episodes(self.mdp,
                                                       optimal_policy,
                                                       self.n_episodes)
            feature_expectation = compute_feature_expectations(
                                                    self.reward_features,
                                                    trajectories,
                                                    self.state_space,
                                                    self.action_space)

            mean_feature_expectation = mean_feature_expectation + \
                np.dot(feature_expectation - mean_feature_expectation,
                       expert_feature_expectation - mean_feature_expectation) / \
                np.dot(feature_expectation - mean_feature_expectation,
                       feature_expectation - mean_feature_expectation) * \
                (feature_expectation - mean_feature_expectation)

            w = expert_feature_expectation - mean_feature_expectation
            margin = la.norm(w)

        reward = np.dot(self.reward_features, w)
        return reward



