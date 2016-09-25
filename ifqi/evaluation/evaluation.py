from __future__ import print_function
import sys
if sys.version_info[0]==3:
    from builtins import range
import time
import numpy as np


def _eval_and_render(mdp, policy, nbEpisodes=1, metric='discounted',
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
    fps = mdp.metadata.get('video.frames_per_second') or 100
    values = np.zeros(nbEpisodes)
    gamma = mdp.gamma
    if metric == 'average':
        gamma = 1
    for e in range(nbEpisodes):
        epPerformance = 0.0
        df = 1
        t = 0
        H = None
        done = False
        if render:
            mdp.render(mode='human')
        if hasattr(mdp, 'horizon'):
            H = mdp.horizon
        mdp.reset()
        state = mdp._reset(initialState)
        while (t < H) and (not done):
            action = policy.drawAction(state)
            state, r, done, _ = mdp.step(action)
            epPerformance += df * r
            df *= gamma
            t += 1

            if render:
                mdp.render()
                time.sleep(1.0 / fps)
        if gamma == 1:
            epPerformance /= t
        values[e] = epPerformance
    return values.mean(), 2 * values.std() / np.sqrt(nbEpisodes)


def _parallel_eval(mdp, policy, nbEpisodes, metric, initialState):
    # TODO using joblib
    return _eval_and_render(mdp, policy, nbEpisodes, metric, initialState, False)


def evaluate_policy(mdp, policy, nbEpisodes=1,
                    metric='discounted', initialState=None, render=False):
    """
    This function evaluate a policy on the given environment w.r.t.
    the specified metric by executing multiple episode.
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
        return _eval_and_render(mdp, policy, nbEpisodes, metric, initialState, True)
    else:
        return _parallel_eval(mdp, policy, nbEpisodes, metric, initialState)


def collectEpisode(mdp, policy=None):
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
    done = False
    t = 0
    H = None
    data = list()
    action = None
    if hasattr(mdp, 'horizon'):
        H = mdp.horizon
    state = mdp.reset()
    assert len(state.shape) == 1
    while (t < H) and (not done):
        if policy:
            action = policy.drawAction(state)
        else:
            action = mdp.action_space.sample()
        nextState, reward, done, _ = mdp.step(action)

        if not done:
            if t < mdp.horizon:
                newEl = np.column_stack((0, state, action, reward, nextState, 0)).ravel()
                #newEl = [0] + state.tolist() + action.tolist() + [reward] + nextState.tolist() + [0]
            else:
                newEl = np.column_stack((1, state, action, reward, nextState, 0)).ravel()
                # newEl = [1] + state.tolist() + action.tolist() + [reward] + nextState.tolist() + [0]
        else:
            newEl = np.column_stack((1, state, action, reward, nextState, 1)).ravel()
            #newEl = [1] + state.tolist() + action.tolist() + [reward] + nextState.tolist() + [1]

        assert len(newEl.shape) == 1
        data.append(newEl.tolist())
        state = nextState
        t += 1

    return np.array(data)
