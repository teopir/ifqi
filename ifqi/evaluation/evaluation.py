from __future__ import print_function
from builtins import range
import time

import gym
import numpy as np
from ifqi.algorithms.selection.feature_extraction.helpers import crop_state
from ifqi.evaluation.utils import filter_state_with_RFS

from ..envs.utils import get_space_info
from joblib import Parallel, delayed


def _eval_and_render(mdp, policy, metric='discounted',
                     initial_states=None, render=True):
    """
    This function evaluate a policy on the specified metric by executing
    multiple episode and visualize its performance
    Params:
        mdp (object): the environment to solve
        policy (object): a policy object (method draw_action is expected)
        metric (string, 'discounted'): the evaluation metric ['discounted',
            'average']
        initial_states (np.array, None): initial states to use to evaluate
            policy
        render (bool, True): whether to render the step of the environment
    Return:
        metric (float): the selected evaluation metric
        confidence (float): 95% confidence level for the provided metric
        step (float): average number of step before finish
        step_confidence (float):  95% confidence level for step average
    """
    n_episodes = initial_states.shape[0]
    values, steps = _eval_and_render_vectorial(mdp, policy, metric,
                                               initial_states, render)

    return values.mean(), 2 * values.std() / np.sqrt(n_episodes), \
           steps.mean(), 2 * steps.std() / np.sqrt(n_episodes)


def _eval_and_render_vectorial(mdp, policy, metric='discounted',
                               initial_states=None, n_episodes=1, render=True):
    """
    This function evaluate a policy on the specified metric by executing
    multiple episode and visualize its performance
    Params:
        mdp (object): the environment to solve
        policy (object): a policy object (method draw_action is expected)
        metric (string, 'discounted'): the evaluation metric ['discounted',
            'average']
        initial_states (np.array, None): initial states to use to evaluate
            policy. If None the state is choosen by the mdp
        n_episodes (int): number of episodes to be simulated. It is used
            only when initial_states is None
        render (bool, True): whether to render the step of the environment
    Return:
        metric (float): the selected evaluation metric
        step (float): average number of step before finish
    """
    fps = mdp.metadata.get('video.frames_per_second') or 100

    if initial_states is not None:
        n_episodes = initial_states.shape[0] \
            if len(initial_states.shape) > 1 else 1
    values = np.zeros(n_episodes)
    steps = np.zeros(n_episodes)
    gamma = mdp.gamma
    if hasattr(mdp, 'horizon'):
        H = mdp.horizon
    else:
        H = np.inf
    if metric == 'average':
        gamma = 1
    for e in range(n_episodes):
        ep_performance = 0.0
        df = 1
        t = 0

        done = False
        if render:
            mdp.render(mode='human')
        state = mdp.reset(initial_states[e, :]
                          if initial_states is not None else None)
        while t < H and not done:
            action = policy.draw_action(state, done, True)
            state, r, done, _ = mdp.step(action)
            ep_performance += df * r
            df *= gamma
            t += 1

            if render:
                mdp.render()
                time.sleep(1.0 / fps)
        if gamma == 1:
            ep_performance /= t
        values[e] = ep_performance
        steps[e] = t

    return values, steps


def _parallel_eval(mdp, policy, metric, initial_states, n_episodes,
                   n_jobs, n_episodes_per_job):
    if initial_states is not None:
        n_episodes = initial_states.shape[0] \
            if len(initial_states.shape) > 1 else 1

    if hasattr(mdp, 'spec') and mdp.spec is not None:
        how_many = int(round(n_episodes / n_episodes_per_job))
        out = Parallel(
            n_jobs=n_jobs, verbose=2,
        )(
            delayed(_eval_and_render)(gym.make(mdp.spec.id), policy,
                                      n_episodes_per_job, metric,
                                      initial_states)
            for _ in range(how_many))

        # out is a list of quadruplet: mean J, 95% conf lev J, mean steps,
        # 95% conf lev steps
        # (confidence level should be 0 or NaN)
        values, steps = np.array(out)
    else:
        values, steps = _eval_and_render_vectorial(mdp, policy, metric,
                                                   initial_states, n_episodes, False)
    return values.mean(), 2 * values.std() / np.sqrt(n_episodes), \
           steps.mean(), 2 * steps.std() / np.sqrt(n_episodes)


def _eval_with_FE(mdp, policy, AE, metric, selected_states=None, max_ep_len=np.inf, render=False):
    gamma = mdp.gamma if metric == 'discounted' else 1
    ep_performance = 0.0
    df = 1.0  # Discount factor

    # Start episode
    reward = 0
    done = False
    state = mdp.reset()
    # Get encoded features
    preprocessed_state = np.expand_dims(np.asarray(crop_state(state)), axis=0)
    encoded_state = AE.flat_encode(preprocessed_state)
    if selected_states is not None:
        filtered_state = filter_state_with_RFS(encoded_state, selected_states)
    else:
        filtered_state = encoded_state

    frame_counter = 0
    while not done and frame_counter <= max_ep_len:
        frame_counter += 1

        # Select an action
        action = policy.draw_action(filtered_state, done, evaluation=True)
        # Execute the action, get next state and reward
        next_state, reward, done, info = mdp.step(action)
        ep_performance += df * reward  # Update performance
        df *= gamma  # Update discount factor

        # Get encoded features
        preprocessed_next_state = np.expand_dims(crop_state(next_state), axis=0)
        encoded_next_state = AE.flat_encode(preprocessed_next_state)
        if selected_states is not None:
            filtered_next_state = filter_state_with_RFS(encoded_next_state, selected_states)
        else:
            filtered_next_state = encoded_next_state

        # Render environment
        if render:
            mdp.render(mode='human')

        # Update state
        state = next_state
        filtered_state = filtered_next_state

    if gamma == 1:
        ep_performance /= frame_counter

    return ep_performance, frame_counter

def evaluate_policy(mdp, policy, metric='discounted', initial_states=None,
                    n_episodes=1, render=False, n_jobs=-1, n_episodes_per_job=10):
    """
    This function evaluate a policy on the given environment w.r.t.
    the specified metric by executing multiple episode.
    Params:
        mdp (object): the environment to solve
        policy (object): a policy object (method draw_action is expected)
        metric (string, 'discounted'): the evaluation metric ['discounted',
            'average']
        initial_states (np.array, None): initial states to use to evaluate
            policy. If none the state is selected by the mdp
        render (bool, True): whether to render the step of the environment
    Return:
        metric (float): the selected evaluation metric
        confidence (float): 95% confidence level for the provided metric
        step (float): average number of step before finish
        step_confidence (float):  95% confidence level for step average
    """
    assert metric in ['discounted', 'average'], "unsupported metric"
    if render:
        return _eval_and_render(mdp, policy, metric, initial_states, True)
    else:
        return _parallel_eval(mdp, policy, metric, initial_states,
                              n_episodes, n_jobs, n_episodes_per_job)


def evaluate_policy_with_FE(mdp, policy, AE, metric='discounted', n_episodes=1,
                            selected_states=None, max_ep_len=np.inf, render=False, n_jobs=1):
    """
        This function evaluate a policy on the given environment w.r.t.
        the specified metric by executing multiple episode, using the provided
        feature extraction model to encode states.
        Params:
            mdp (object): the environment to solve
            policy (object): a policy object (method draw_action is expected)
            AE (object): the feature extraction model (method flat_encode is
                expected)
            metric (string, 'discounted'): the evaluation metric ['discounted',
                'average']
            selected_states (iterable, None): the subset of states on which the
                policy was trained. If None, use all states.
            max_ep_len (int, inf): allow evaluation episodes to run at most this number
                of frames.
            render (bool, False): whether to render the step of the environment
            n_jobs (int, 1): the number of processes to use for evaluation
                (leave default value if the feature extraction model - AE - runs
                on GPU)
        Return:
            metric (float): the selected evaluation metric
            confidence (float): 95% confidence level for the provided metric
    """

    assert metric in ['discounted', 'average'], "unsupported metric"
    out =  Parallel(n_jobs=n_jobs)(
        delayed(_eval_with_FE)(
            mdp, policy, AE, metric, selected_states=selected_states, max_ep_len=max_ep_len, render=render
        )
        for ep in range(n_episodes)
    )

    values, steps = np.array(zip(*out))

    return values.mean(), 2 * values.std() / np.sqrt(n_episodes), \
           steps.mean(), 2 * steps.std() / np.sqrt(n_episodes)


def collect_episodes(mdp, policy=None, n_episodes=1, n_jobs=1):
    """
    if hasattr(mdp, 'spec') and mdp.spec is not None:
        out = Parallel(n_jobs=n_jobs, verbose=2,)(
            delayed(collect_episode)(gym.make(mdp.spec.id), policy)
            for i in range(n_episodes))

        # out is a list of np.array, each one representing an episode
        # merge the results
        data = np.concatenate(out, axis=0)
    else:
        raise ValueError('collect_episodes must be implemented')
    """
    data = np.array(collect_episode(mdp, policy))
    for i in range(1, n_episodes):
        data = np.append(data, collect_episode(mdp, policy), axis=0)

    return data


def collect_episode(mdp, policy=None):
    """
    This function can be used to collect a dataset running an episode
    from the environment using a given policy.

    Params:
        mdp (object): the environment to solve
        policy (object, None): an object that can be evaluated in order to get
            an action

    Returns:
        - a dataset composed of:
            - state
            - action
            - reward
            - next state
            - a flag indicating whether the reached state is absorbing
            - a flag indicating whether the episode is finished (absorbing state
              is reached or the time horizon is met)
    """
    done = False
    t = 0
    data = list()
    horizon = mdp.horizon
    state = mdp.reset()
    # state_dim, action_dim, reward_dim = get_space_info(mdp)

    while t < horizon and not done:
        if policy is not None:
            action = policy.draw_action(state, done)
        else:
            action = mdp.action_space.sample()
        action = np.array([action]).ravel()
        next_state, reward, done, _ = mdp.step(action)
        new_el = state.tolist() + action.tolist() + [reward] + \
                 next_state.tolist()
        if not done:
            if t < horizon - 1:
                new_el += [0, 0]
            else:
                new_el += [0, 1]
        else:
            new_el += [1, 1]

        data.append(new_el)
        state = next_state
        t += 1

    return np.array(data)
