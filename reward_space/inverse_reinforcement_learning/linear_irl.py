import numpy as np
import numpy.linalg as la
from cvxopt import matrix, solvers

class LinearIRL(object):

    '''
    Abbeel, Pieter, and Andrew Y. Ng.
    "Apprenticeship learning via inverse reinforcement learning."
    Proceedings of the twenty-first international conference
    on Machine learning. ACM, 2004.
    '''

    def __init__(self,
                 transition_model,
                 policy,
                 gamma,
                 r_max,
                 l1_penalty=0.,
                 type_='state'):

        #transition model: tensor (n_states, n_actions, n_states)

        self.transition_model = transition_model
        self.policy = policy
        self.gamma = gamma
        self.r_max = r_max
        self.l1_penalty = l1_penalty
        if not type_ in ['state', 'state-action']:
            raise ValueError()

        self.type_ = type_

        self.n_states, self.n_actions = transition_model.shape[:2]
        self.n_states_actions = self.n_states * self.n_actions

        self.greedy_action = policy.argmax(axis=1)
        self.greedy_policy = np.zeros((self.n_states, self.n_states_actions))
        self.greedy_policy[np.arange(self.n_states),\
            np.arange(self.n_states) * self.n_actions + self.greedy_action] = 1.

    def fit(self, verbose=False):
        P_star = self.transition_model[np.arange(self.n_states), self.greedy_action, :].squeeze()
        multiplier = la.inv(np.eye(self.n_states) - self.gamma * P_star)

        # Min constraint
        #(n_state*(n_actions-1), n_states)
        min_constraint = np.repeat(np.eye(self.n_states), self.n_actions - 1,
                                   axis=0)
        #Inequality constraints
        if self.type_ == 'state':
            c = matrix(-np.concatenate([np.zeros(self.n_states), \
                np.ones(self.n_states), - self.l1_penalty * np.ones(self.n_states)]))
            #((n_actions-1)*n_states, n_states)
            constraints = np.vstack([-np.dot((P_star[s] - self.transition_model[s, a, :].squeeze()),\
                                  multiplier)
                                  for s in np.arange(self.n_states) \
                                  for a in set(np.arange(self.n_actions)) - set([self.greedy_action[s]])])

            #bound on R

            eye_pad = np.eye(self.n_states)
            zero_pad1 = np.zeros((self.n_states * (self.n_actions - 1), self.n_states))
            zero_pad2 = np.zeros((self.n_states, self.n_states))
            G = np.hstack([np.vstack([constraints, constraints, -eye_pad, eye_pad, eye_pad, -eye_pad]),
                           np.vstack([min_constraint, zero_pad1, zero_pad2, zero_pad2, zero_pad2, zero_pad2]),
                           np.vstack([zero_pad1, zero_pad1, -eye_pad, -eye_pad, zero_pad2, zero_pad2])])
            h = np.concatenate([np.zeros(2 * self.n_states * (self.n_actions - 1) + 2 * self.n_states),
                           self.r_max * np.ones(2 * self.n_states)])
        else:
            c = matrix(-np.concatenate([np.zeros(self.n_states_actions), \
                np.ones(self.n_states), - self.l1_penalty * np.ones(self.n_states_actions)]))
            #((n_actions-1)*n_states, n_states*n_actions)
            constraints = np.vstack([-la.multi_dot([(P_star[s] - self.transition_model[s, a, :].squeeze()),\
                                  multiplier, self.greedy_policy])
                                  for s in np.arange(self.n_states) \
                                  for a in set(np.arange(self.n_actions)) - set([self.greedy_action[s]])])

            print(constraints.shape)
            eye_pad = np.eye(self.n_states_actions)
            zero_pad1 = np.zeros((self.n_states * (self.n_actions - 1), self.n_states))
            zero_pad2 = np.zeros((self.n_states_actions, self.n_states))
            zero_pad3 = np.zeros((self.n_states_actions, self.n_states_actions))
            zero_pad4 = np.zeros((self.n_states * (self.n_actions - 1), self.n_states_actions))
            G = np.hstack([np.vstack([constraints, constraints, -eye_pad, eye_pad, eye_pad, -eye_pad]),
                           np.vstack([min_constraint, zero_pad1, zero_pad2, zero_pad2, zero_pad2, zero_pad2]),
                           np.vstack([zero_pad4, zero_pad4, -eye_pad, -eye_pad, zero_pad3, zero_pad3])])
            h = np.concatenate([np.zeros(2 * self.n_states * (self.n_actions - 1) + 2 * self.n_states_actions),
                                self.r_max * np.ones(2 * self.n_states_actions)])

        G = matrix(G)
        h = matrix(h)
        # Solve the LP problem
        result = solvers.lp(c, G, h)

        # Get the reward
        if self.type_ == 'state':
            x = np.array(result['x'][:self.n_states])
        else:
            x = np.array(result['x'][:self.n_states_actions])

        return x.ravel()