# Authors: Matteo Pirotta <matteo.pirotta@polimi.it>
#          Marcello Restelli <marcello.restelli@polimi.it>
#
# License: BSD 3 clause

"""Recursive feature selection"""

from __future__ import print_function
import numpy as np
from sklearn.utils import check_array
from sklearn.utils.metaestimators import if_delegate_has_method
from sklearn.base import BaseEstimator, is_classifier
from sklearn.base import MetaEstimatorMixin
from sklearn.base import clone
from sklearn.feature_selection.base import SelectorMixin
from sklearn.metrics import r2_score, mean_squared_error
import sklearn
if sklearn.__version__ == '0.17':
    from sklearn.cross_validation import cross_val_score, check_cv
else:
    from sklearn.model_selection import cross_val_score, check_cv
from sklearn.preprocessing import StandardScaler


class RFS(BaseEstimator, MetaEstimatorMixin, SelectorMixin):
    def __init__(self, feature_selector, features_names=None, verbose=0):
        self.feature_selector = feature_selector
        self.features_names = features_names
        self.verbose = verbose

    def fit(self, state, actions, next_states, reward):
        """Fit the RFS model. The input data is a set of transitions
        (state, action, next_state, reward).

        Parameters
        ----------
        state : {array-like, sparse matrix}, shape = [n_samples, n_states]
            The set of states.

        actions : {array-like, sparse matrix}, shape = [n_samples, n_actions]
            The set of actions.

        next_states : {array-like, sparse matrix}, shape = [n_samples, n_states]
            The set of states reached by applying the given action in the given state.

        reward : {array-like, sparse matrix}, shape = [n_samples, n_rewards]
            The set of rewords associate to the transition.
        """
        check_array(state, accept_sparse=True)
        check_array(actions, accept_sparse=True)
        check_array(next_states, accept_sparse=True)
        check_array(reward, accept_sparse=True)
        return self._fit(state, actions, next_states, reward)

    def _fit(self, states, actions, next_states, reward):
        X = np.column_stack((states, actions))
        support = np.zeros(X.shape[1], dtype=np.bool)
        self.support_ = self._recursive_step(X, next_states, reward, support)
        return self

    def _recursive_step(self, X, next_state, Y, curr_support):
        """
        Recursively selects the features that explains the provided target
        (initially Y must be the reward)
        Args:
            X (numpy.array): features. shape = [n_samples, (state_dim + action_dim)]
            next_state (numpy.array): features of the next state [n_samples,  state_dim]
            Y (numpy.array): target to fit (intially reward, than the state)
            curr_support (numpy.array): selected features of X (ie. selected state and action).
                Boolean array of shape [state_dim + action_dim, 1]

        Returns:
            support (numpy.array): updated support

        """
        n_states = next_state.shape[1]
        n_actions = X.shape[1] - n_states

        fs = clone(self.feature_selector)

        if hasattr(fs, 'set_feature_names'):
            fs.set_feature_names(self.features_names)

        fs.fit(X, Y)

        sa_support = fs.get_support()  # get selected features of X
        new_state_support = sa_support[:n_states]  # get only state features
        new_state_support[curr_support[:n_states]] = False  # remove state features already selected
        idxs = np.where(new_state_support)[0]  # get numerical index

        # update support with features already selected
        # new_support + old_support
        sa_support[curr_support] = True

        for idx in idxs:
            target = next_state[:, idx]
            rfs_s_features = self._recursive_step(X, next_state, target, sa_support)
            sa_support[rfs_s_features] = True
        return sa_support

    def _get_support_mask(self):
        """
        The selected features of state and action
        Returns:
            support (numpy.array): the selected features of
                state and action
        """
        return self.support_
