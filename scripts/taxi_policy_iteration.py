# !/usr/bin/env python3
import gym
import argparse
import numpy as np
from collections import OrderedDict


class ValueFuntion(OrderedDict):
    def __init__(self, env, *args, **kwargs):
        self.env = env
        super(ValueFuntion, self).__init__(*args, **kwargs)
        for s in range(self.env.nS):
            self[s] = 0

    def __str__(self, *args, **kwargs):
        if hasattr(env, 'ncol') and hasattr(env, 'nrow'):
            return np.array_str(np.array(list(self.values())).reshape(env.ncol, env.nrow),
                                precision=2, suppress_small=True)
        return super(ValueFuntion, self).__str__(*args, **kwargs)


class TablePolicy(OrderedDict):
    def __init__(self, env):
        self.env = env
        self.policy = dict()
        for s in range(self.env.nS):
            self.policy[s] = self.env.action_space.sample()

    def __str__(self, *args, **kwargs):
        if isinstance(self.env, gym.envs.toy_text.frozen_lake.FrozenLakeEnv):
            mapping = {
                0: '<',
                1: 'v',
                2: '>',
                3: '^'
            }
            result = np.array_str(np.array([mapping[policy(s)] if env.desc.flatten()[s] != b'H' else 'H'
                                            for s in range(env.nS)]).reshape(env.ncol, env.nrow),
                                  precision=2, suppress_small=True)
            return result
        return super(TablePolicy, self).__str__(*args, **kwargs)

    def __call__(self, s):
        return self.policy[s]


def policy_update(env, value, policy, gamma, theta=1e-3):
    stable = True
    for s in range(env.nS):
        nexts = [sum(p * (r + gamma * value[next_s])
                     for p, next_s, r, done in env.P[s][a])
                 for a in range(env.action_space.n)]
        na = np.argmax(nexts)
        stable = stable and (policy.policy[s] == na)
        policy.policy[s] = na
    return stable


def value_update(env, value, policy, gamma, theta=1e-3):
    delta = np.inf
    while delta > theta:
        delta = 0
        for s in range(env.nS):
            a = policy(s)
            nv = sum([p * (r + gamma * value[next_s]) * (not done)
                      for p, next_s, r, done
                      in env.P[s][a]])
            delta = max(delta, abs(value[s] - nv))
            value[s] = nv


def policy_iteration(env, value, policy, gamma=0.9):
    policy_stable = 10
    while policy_stable:
        value_update(env, value, policy, gamma)
        policy_stable -= policy_update(env, value, policy, gamma)

def compute_policy():
    env = gym.make('Taxi-v1')

    value = ValueFuntion(env)
    policy = TablePolicy(env)

    policy_iteration(env, value, policy, gamma=0.99)
    return policy.policy



