import gym
from .. import evaluation as eval


class Environment(gym.Env):
    def __init__(self):
        self.gamma = 1
        self.horizon = None

    def setSeed(self,seed=None):
        self._seed(seed=seed)

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
        return eval.evaluate_policy(self, policy, nbEpisodes,
                                          metric, initialState, render)

    def collectEpisode(self, policy=None):
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
        return eval.collectEpisode(self, policy)
