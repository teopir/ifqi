# Authors: Matteo Pirotta <matteo.pirotta@polimi.it>
#          Marcello Restelli <marcello.restelli@polimi.it>
#
# License: BSD 3 clause

"""Iterative feature selection"""

from __future__ import print_function
import numpy as np
from sklearn.utils import check_X_y, safe_sqr
from sklearn.utils.metaestimators import if_delegate_has_method
from sklearn.base import BaseEstimator, is_classifier
from sklearn.base import MetaEstimatorMixin
from sklearn.base import clone
from sklearn.feature_selection.base import SelectorMixin
from sklearn.metrics import r2_score, mean_squared_error
import sklearn
if sklearn.__version__ == '0.17':
    from sklearn.cross_validation import cross_val_score, check_cv, cross_val_predict
else:
    from sklearn.model_selection import cross_val_score, check_cv, cross_val_predict
from sklearn.preprocessing import StandardScaler


class IFS(BaseEstimator, MetaEstimatorMixin, SelectorMixin):
    """Feature ranking with recursive feature elimination.

    Given an external estimator that assigns weights to features (e.g., the
    coefficients of a linear model), the goal of iterative feature selection
    (IFS) is to select features by recursively consider bigger sets of features.
    Given the targets to be explained and the set of candidate features, the
    IFS algorithm first globally ranks the features according to a statistical
    measure of significance (default R2 score). To account for feature redundancy,
    only the most significant features are then added to the set of selected features,
    which is used to fit a model to explain the targets.
    The algorithm proceeds by repeating the ranking process using as new output
    feature the residuals of the model built at the previous iteration.
    The algorithm iterates these operations until the best features returned by the
    ranking algorithm are already in the select set or the accuracy of the model
    built upon the selected features does not significantly improve.

    Read more in the :ref:`User Guide <rfe>`.

    Parameters
    ----------
    estimator : object
        A supervised learning estimator with a `fit` method that updates a
        `coef_` attribute that holds the fitted parameters. Important features
        must correspond to high absolute values in the `coef_` array.

        For instance, this is the case for most supervised learning
        algorithms such as Support Vector Classifiers and Generalized
        Linear Models from the `svm` and `linear_model` modules.

    n_features_step : int or float, optional (default=1)
        If greater than or equal to 1, then `step` corresponds to the (integer)
        number of features to remove at each iteration.
        If within (0.0, 1.0), then `step` corresponds to the percentage
        (rounded down) of features to remove at each iteration.

    cv : int, cross-validation generator or an iterable, optional
        Determines the cross-validation splitting strategy.
        Possible inputs for cv are:

        - None, to use the default 3-fold cross-validation,
        - integer, to specify the number of folds.
        - An object to be used as a cross-validation generator.

        - An iterable yielding train/test splits.

        For integer/None inputs, if ``y`` is binary or multiclass,
        :class:`StratifiedKFold` used. If the estimator is a classifier
        or if ``y`` is neither binary nor multiclass, :class:`KFold` is used.

        Refer :ref:`User Guide <cross_validation>` for the various
        cross-validation strategies that can be used here.

    scale: bool, default=True
        Scale the feature and the targets before to fit the estimator

    features_names, list of string, None
        Feature names

    verbose : int, default=0
        Controls verbosity of output.

    Attributes
    ----------
    n_features_ : int
        The number of selected features.

    support_ : array of shape [n_features]
        The mask of selected features.

    estimator_ : object
        The external estimator fit on the reduced dataset.

    Examples
    --------
    The following example shows how to retrieve the most informative
    features in the Friedman #1 dataset.

    >>> from sklearn.datasets import make_friedman1
    >>> from sklearn.feature_selection import IFS
    >>> from sklearn.svm import SVR
    >>> X, y = make_friedman1(n_samples=50, n_features=10, random_state=0)
    >>> estimator = SVR(kernel="linear")
    >>> selector = IFS(estimator, n_features_step=2)
    >>> selector = selector.fit(X, y)
    >>> selector.support_ # doctest: +NORMALIZE_WHITESPACE
    array([False  True False  True  True
           False False False False False], dtype=bool)

    References
    ----------

    .. [1] Castelletti, A., Galelli, S., Restelli, M., & Soncini-Sessa, R., "Tree-based
           Feature Selection for Dimensionality Reduction of Large-scale Control Systems",
           ADPRL, 2011. DOI: 10.1109/ADPRL.2011.5967387
    .. [2] Galelli, S., Castelletti, A., "Tree-based Iterative Input variable Selection
           for hydrological modelling", Water Resources Research, 49(7), 4295-4310, 2013.
    """

    def __init__(self, estimator, n_features_step=1,
                 cv=None,
                 scale=True,
                 force_iterations=None,
                 features_names=None,
                 verbose=0):
        self.estimator = estimator
        assert n_features_step == 1,\
            'currently only one features per iteration is supported'
        self.n_features_step = n_features_step
        self.cv = cv
        self.scale = scale
        self.force_iterations = force_iterations
        self.features_names = features_names
        self.verbose = verbose

    @property
    def _estimator_type(self):
        return self.estimator._estimator_type

    def fit(self, X, y):
        """Fit the IFS model and then the underlying estimator on the selected
           features.

        Parameters
        ----------
        X : {array-like, sparse matrix}, shape = [n_samples, n_features]
            The training input samples.

        y : array-like, shape = [n_samples]
            The target values.
        """
        return self._fit(X, y, self.features_names)

    def _fit(self, X, y, step_score=None, features_names=None):
        X, y = check_X_y(X, y, accept_sparse=['csr', 'csc', 'coo'],
                         multi_output=True)
        # Initialization
        n_features = X.shape[1]
        features = np.arange(n_features)

        cv = self.cv
        cv = check_cv(cv, y, classifier=is_classifier(self.estimator))
        if sklearn.__version__ == '0.17':
            n_splits = cv.n_folds
        else:
            n_splits = cv.get_n_splits(X, y)

        if self.verbose > 0:
            print("Fitting {0} folds for each of iteration".format(n_splits))

        if 0.0 < self.n_features_step < 1.0:
            step = int(max(1, self.n_features_step * n_features))
        else:
            step = int(self.n_features_step)
        if step <= 0:
            raise ValueError("Step must be >0")

        # if self.force_iterations is None:
        #     force_iteration = False
        # else:
        #     force_iteration = self.force_iterations

        # if step_score is None:
        #     step_score = r2_score

        if features_names is not None:
            features_names = np.array(features_names)
        else:
            if self.features_names is not None:
                features_names = self.features_names
            else:
                features_names = np.arange(n_features) # use indices

        tentative_support_ = np.zeros(n_features, dtype=np.bool)
        current_support_ = np.zeros(n_features, dtype=np.bool)

        self.scores_ = []
        self.features_per_it_ = []

        target = y

        score, confidence_interval = -np.inf, 0
        proceed = True
        while proceed:
            if self.verbose > 0:
                print('\nN-times variance of target: {}'.format(target.var() * target.shape[0]))
            # update values
            old_confidence_interval = confidence_interval
            old_score = score

            if self.verbose > 0:
                print()
                print('Feature ranking')
                print()

            if self.scale:
                target = StandardScaler().fit_transform(target.reshape(-1,1)).ravel()

            # Rank the remaining features
            rank_estimator = clone(self.estimator)
            rank_estimator.fit(X, target)

            # Get coefs
            if hasattr(rank_estimator, 'coef_'):
                coefs = rank_estimator.coef_
            elif hasattr(rank_estimator, 'feature_importances_'):
                coefs = rank_estimator.feature_importances_
            else:
                raise RuntimeError('The classifier does not expose '
                                   '"coef_" or "feature_importances_" '
                                   'attributes')

            # Get ranks by ordering in ascending way
            if coefs.ndim > 1:
                ranks = np.argsort(safe_sqr(coefs).sum(axis=0))
                coefs = coefs.sum(axis=0)
            else:
                ranks = np.argsort(safe_sqr(coefs))

            # for sparse case ranks is matrix
            ranks = np.ravel(ranks)

            if self.verbose > 0:
                ranked_f = features[ranks]
                if features_names is not None:
                    ranked_n = features_names[ranks]
                else:
                    ranked_n = ['-'] * n_features
                print('{:6}\t{:6}\t{:8}\t{}'.format('Rank', 'Index', 'Score', 'Feature Name'))
                for i in range(n_features):
                    idx = n_features - i - 1
                    print('#{:6}\t{:6}\t{:6f}\t{}'.format(str(i), str(ranked_f[idx]), coefs[ranks[idx]], ranked_n[idx]))

            # if coefs[ranks][-1] < 1e-5:
            #     if self.verbose > 0:
            #         import warnings
            #         warnings.warn('scores are too small to be used, please standardize inputs')
            #     break

            # get the best features (ie, the latest one)
            # if the most ranked features is selected go on a select
            # other features accordingly to the ranking

            # threshold = step
            # step_features = features[ranks][-threshold:]

            ii = len(features_names) - 1
            step_features = features[ranks][ii]
            while np.all(current_support_[step_features]) and ii > 0:
                ii -= 1
                step_features = features[ranks][ii]

            if np.all(current_support_[step_features]):
                if self.verbose > 0:
                    print("Selected features: {} {}".format(features_names[step_features], step_features))
                    # if features_names is not None:
                    #     print("Selected features: {} {}".format(features_names[ranks][-threshold:], step_features))
                    # else:
                    #     print("Selected features: {}".format(step_features))
                    print('Ended because selected features already selected')
                step_features = None
                break

            # update selected features
            tentative_support_[step_features] = True

            # get the selected features
            X_selected = X[:, features[tentative_support_]]

            # cross validates to obtain the scores
            # cv_scores = cross_val_score(clone(self.estimator), X_selected, y, cv=cv, scoring='r2')
            y_hat = cross_val_predict(clone(self.estimator), X_selected, y, cv=cv)

            # compute new target
            target = y - y_hat

            # compute score and confidence interval
            score = r2_score(y_true=y, y_pred=y_hat, multioutput='uniform_average')  # np.mean(cv_scores)
            if self.verbose > 0:
                print('r2: {}'.format(r2_score(y_true=y, y_pred=y_hat, multioutput='raw_values')))
            # m2 = np.mean(cv_scores * cv_scores)
            SIGNIFICANCE = 0.0
            confidence_interval = SIGNIFICANCE  # * np.sqrt((m2 - score * score) / (n_splits - 1))

            if self.verbose > 0:
                # if features_names is not None:
                print("Selected features: {} {}".format(features_names[step_features], step_features))
                print("Total features: {} {}".format(features_names[tentative_support_],
                                                     features[tentative_support_]))
                # else:
                #     print("Selected features: {}".format(step_features))
                #     print("Total features: {}".format(features[tentative_support_]))
                print("R2= {} +- {}".format(score, confidence_interval))

            self.scores_.append(score)
            self.features_per_it_.append(features_names[tentative_support_])

            # check terminal condition
            proceed = score - old_score > old_confidence_interval + confidence_interval

            if proceed or np.sum(current_support_) == 0:
                # last feature set proved to be informative
                # we need to take into account of the new features (update current support)
                current_support_[step_features] = True

                # all the features are selected, stop
                if np.sum(current_support_) == n_features:
                    if self.verbose > 0:
                        print("All the features has been selected.")
                    proceed = False
            else:
                # last feature set proved to be not informative
                # keep old support and delete the current one (it is no more necessary)
                del tentative_support_
                if self.verbose > 0:
                    print('Last feature {} not added to the set'.format(features_names[step_features]))

        # Set final attributes
        self.estimator_ = clone(self.estimator)
        # self.estimator_.fit(Xns[:, current_support_], yns)
        self.estimator_.fit(X[:, current_support_], y)

        self.n_features_ = current_support_.sum()
        self.support_ = current_support_
        # self.ranking_ = ranking_

        return self

    @if_delegate_has_method(delegate='estimator')
    def predict(self, X):
        """Reduce X to the selected features and then predict using the
           underlying estimator.

        Parameters
        ----------
        X : array of shape [n_samples, n_features]
            The input samples.

        Returns
        -------
        y : array of shape [n_samples]
            The predicted target values.
        """
        return self.estimator_.predict(self.transform(X))

    @if_delegate_has_method(delegate='estimator')
    def score(self, X, y):
        """Reduce X to the selected features and then return the score of the
           underlying estimator.

        Parameters
        ----------
        X : array of shape [n_samples, n_features]
            The input samples.

        y : array of shape [n_samples]
            The target values.
        """
        return self.estimator_.score(self.transform(X), y)

    def _get_support_mask(self):
        return self.support_

    @if_delegate_has_method(delegate='estimator')
    def decision_function(self, X):
        return self.estimator_.decision_function(self.transform(X))

    @if_delegate_has_method(delegate='estimator')
    def predict_proba(self, X):
        return self.estimator_.predict_proba(self.transform(X))

    @if_delegate_has_method(delegate='estimator')
    def predict_log_proba(self, X):
        return self.estimator_.predict_log_proba(self.transform(X))

    def set_feature_names(self, fs):
        if fs is None:
            self.features_names = None
        else:
            self.features_names = np.array(fs)
