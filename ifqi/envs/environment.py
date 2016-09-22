import numpy as np
from abc import ABCMeta, abstractmethod


class Environment(object):
    """
    Environment abstract class.

    """
    __metaclass__ = ABCMeta

    @abstractmethod
    def __init__(self):
        """
        Constructor.

        """
        # Properties
        self.stateDim = None
        self.actionDim = None
        self.nStates = None
        self.nActions = None
        self.horizon = None
        # End episode
        self._atGoal = None
        self._absorbing = None
        # Discount factor
        self.gamma = None

    @abstractmethod
    def evaluate(self, fqi, expReplay=False, render=False):
        """
        This function evaluates the regressor in the provided object parameter.
        Params:
            fqi (object): an object containing the trained regressor
            expReplay (bool): flag indicating whether to do experience replay
            render (bool): flag indicating whether to render visualize behavior
                           of the agent
        Returns:
            J

        """
        pass

    def collect(self, policy=None):
        """
        This function can be used to collect a dataset from the environment
        using a random policy.

        Params:
            policy (object): an object that can be evaluated in order to get
                             an action

        Returns:
            - a dataset composed of:
                - a flag indicating the beginning of an episode
                - state
                - action
                - reward
                - next state
                - a flag indicating wheter the reached state is absorbing
        """
        self._reset()
        t = 0
        data = list()
        action = None
        while (t < self.horizon) and (not self._isAbsorbing()):
            state = self._getState()
            if policy:
                action = policy.predict(state)
                if isinstance(action, tuple):
                    action = action[0]
            else:
                action = np.random.choice(np.arange(self.nActions))
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

    def runEpisode(self, fqi, expReplay, render=False):
        """
        This function runs an episode using the regressor in the provided
        object parameter.
        Args:
            fqi (object): an object containing the trained regressor
            expReplay (bool): flag indicating whether to do experience replay
            render (bool): flag indicating whether to render visualize behavior
                           of the agent
        Returns:
            - J
            - number of steps
            - a flag indicating if the goal state has been reached
            - augmented training set (if using experience replay)
            - augmented target set (if using experience replay)

        """
        J = 0
        t = 0
        testSuccessful = 0

        # reset the environment (draw a random initial state)
        self._reset()
        if expReplay:
            stateList = list()
            actionList = list()
            rewardList = list()
            df = 1.0
            while (t < self.horizon) and (not self._isAbsorbing()):
                state = self._getState()
                stateList.append(state)
                action, _ = fqi.predict(np.array(state))
                actionList.append(action)
                r = self._step(int(action[0]), render=render)
                rewardList.append(r)
                J += df * r
                t += 1
                df *= self.gamma

            if self._isAtGoal():
                testSuccessful = 1
                print("Goal reached")
            else:
                print("Failure")

            state = self._getState()
            stateList.append(state)

            s = np.array(stateList)
            a = np.array(actionList)
            s1 = s[1:]
            s = s[:-1]
            t = np.array([[0] * (s.shape[0] - 1) + [1]]).T
            r = np.array(rewardList)
            sast = np.concatenate((s, a, s1, t), axis=1)

            return J, t, testSuccessful, sast, r
        else:
            df = 1.0
            while (t < self.horizon) and (not self._isAbsorbing()):
                state = self._getState()
                action, _ = fqi.predict(np.array(state))
                r = self._step(action, render=render)
                J += df * r
                t += 1
                df *= self.gamma

            if self._isAtGoal():
                testSuccessful = 1
                print("Goal reached")
            else:
                print("Failure")

            return J, t, testSuccessful

    @abstractmethod
    def _step(self, u, render=False):
        """
        This function performs the step function of the environment.
        Args:
            u (int): the id of the action to be performed.
        Returns:
            the new state and the obtained reward

        """
        pass

    @abstractmethod
    def _reset(self, state=None):
        """
        This function set the current state to the initial state
        and reset flags.
        Args:
            state (np.array): the initial state

        """
        pass

    @abstractmethod
    def _getState(self):
        """
        Returns:
            a tuple containing the current state.

        """
        pass

    def _isAbsorbing(self):
        """
        Returns:
            True if the state is absorbing, False otherwise.

        """
        return self._absorbing

    def _isAtGoal(self):
        """
        Returns:
            True if the state is a goal state, False otherwise.

        """
        return self._atGoal
