import numpy as np
import gym
from builtins import range
import time


class Environment(gym.Env):
    def __init__(self):
        self.gamma = 1
        self.horizon = None

    def _eval_and_render(self, policy, nbEpisodes=1, metric='discounted',
                         initialState=None, render=True):
        """
        This function evaluate a policy on the specified metric by executing
        multiple episode and visualize its performance
        Params:
            policy (object): a policy object (method drawAction is expected)
            nbEpisodes (int): the number of episodes to execute
            metric (string): the evaluation metric ['discounted', 'average']
        Return:
            metric (float): the selected evaluation metric
            confidence (float): 95% confidence level for the provided metric
        """
        fps = self.metadata.get('video.frames_per_second') or 100
        values = np.zeros(nbEpisodes)
        gamma = self.gamma
        if metric == 'average':
            gamma = 1
        for e in range(nbEpisodes):
            epPerformance = 0.0
            df = 1
            t = 0
            done = False
            if render:
                self.render(mode='human')
            state = self.reset(initialState)
            while (t < self.horizon) and (not done):
                action = policy.drawAction(state)
                state, r, done, _ = self.step(action)
                epPerformance += df * r
                df *= gamma
                t += 1

                if render:
                    self.render()
                    time.sleep(1.0 / fps)
            if gamma == 1:
                epPerformance /= t
            values[e] = epPerformance
        return values.mean(), 2 * values.std() / np.sqrt(nbEpisodes)

    def _parallel_eval(self, policy, nbEpisodes, metric, initialState):
        # TODO using joblib
        return self._eval_and_render(policy, nbEpisodes, metric,
                                     initialState, False)

    def evaluate(self, policy, nbEpisodes=1,
                 metric='discounted', initialState=None, render=False):
        """
        This function evaluate a policy on the specified metric by executing
        multiple episode.
        Params:
            policy (object): a policy object (method drawAction is expected)
            nbEpisodes (int): the number of episodes to execute
            metric (string): the evaluation metric ['discounted', 'average']
            initialState (np.array, None): initial state where to start the episode.
                                    If None the initial state is selected by the mdp.
            render (bool): flag indicating whether to visualize the behavior of
                            the policy
        Return:
            metric (float): the selected evaluation metric
            confidence (float): 95% confidence level for the provided metric
        """
        assert metric in ['discounted', 'average'], "unsupported metric for evaluation"
        if render:
            return self._eval_and_render(policy, nbEpisodes, metric,
                                         initialState, True)
        else:
            return self._parallel_eval(policy, nbEpisodes, metric,
                                       initialState)

    def collectSamples(self, policy=None):
        """
        This function can be used to collect a dataset running an episode
        from the environment using a given policy.

        Params:
            policy (object): an object that can be evaluated in order to get
                             an action

        Returns:
            - a dataset composed of:
                - a flag indicating the end of an episode
                - state
                - action
                - reward
                - next state
                - a flag indicating whether the reached state is absorbing
        """
        self._reset()
        t = 0
        data = list()
        action = None
        while (t < self.horizon) and (not self._isAbsorbing()):
            state = self._getState()
            if policy:
                action = policy.drawAction(state)
                if isinstance(action, tuple):
                    action = action[0]
            else:
                action = self.action_space.sample()

            reward = self._step(action)
            nextState = self._getState()
            t += 1

            if not self._isAbsorbing():
                if t < self.horizon:
                    newEl = [0] + state + [action, reward] + nextState + [0]
                else:
                    newEl = [1] + state + [action, reward] + nextState + [0]
            else:
                newEl = [1] + state + [action, reward] + nextState + [1]

            data.append(newEl)

        return np.array(data)
