from __future__ import print_function
import theano
import theano.tensor as T
import numpy as np
from six import iteritems
import time

from keras.engine.training import slice_X, batch_shuffle, make_batches, \
    standardize_input_data, check_array_lengths
from keras import optimizers


def increment_base_termination(old_theta, new_theta, norm_value=2, tol=1e-3):
    theta_l = old_theta[0]
    theta_lm1 = new_theta[0]
    stop = np.linalg.norm(theta_l - theta_lm1, norm_value) < tol
    return stop


DEFAULT_TERM = {'theta_improvement': increment_base_termination}


class GradPBO(object):
    """
    Construct a GradPBO instance given the specified parameters

    Args:
        bellman_model (object): A class representing the Bellman operator.
            It must comply with the following structure
            - inputs (list): attributes that defines the inputs (theano variables) of the model.
                             len(inputs) must be equal to 1.
            - outputs (list): attributes that defines the outputs (theano variables) of the model.
                             len(outputs) must be equal to 1.
            - trainable_weights (list): list of theano variables representing trainable weights.
            -
        q_model (object): A class representing the Q-function approximation
        steps_ahead (int): How many steps of PBO have to be considered for the computation of the error
        gamma (float): discount factor
        discrete_actions (numpy.array): discrete actions used to approximate the maximum (nactions, action_dim)
        optimizer: str (name of optimizer) or optimizer object.
                See [Keras optimizers](https://keras.io/optimizers/).
        state_dim (None, int): state dimension (state_dim)
        action_dim (None, int): action dimension (action_dim)
        incremental (boolean): if true the incremental version of the Bellman operator is used:
                                Incremental: theta' = theta + f(theta). Not incremental: theta' = f(theta).
        norm_value (np.inf, int): defines the norm used to compute the error
        update_theta_every (int): is the number of steps of the gradient before to update theta.
                                  =1 it means that theta is updated at every gradient step (default 1)
        steps_per_theta_update (None, int): number of steps of projected Bellman operator to apply to theta when
                                        it is updateds. If None it is equal to steps_ahead
        independent (boolean): if True, the gradient over K steps is computed without considering the sequentiality
                            (ie it is approximated without computing the derivative of the maximum). Default False
        verbose (int): verbosity level
    """

    def __init__(self, bellman_model, q_model, steps_ahead,
                 gamma, discrete_actions,
                 optimizer,
                 state_dim=None, action_dim=None, incremental=True,
                 norm_value=np.inf, update_theta_every=1,
                 steps_per_theta_update=None,
                 independent=False,
                 verbose=0, term_condition=None):
        # save MDP information
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.incremental = incremental
        self.gamma = gamma
        self.norm_value = norm_value
        self.update_theta_every = update_theta_every if update_theta_every > 0 else -1
        self.verbose = verbose
        self.independent = independent
        self.steps_per_theta_update = steps_ahead if steps_per_theta_update is None else max(
            1, steps_per_theta_update)

        # create theano variables
        T_s = T.dmatrix()
        T_a = T.dmatrix()
        T_s_next = T.dmatrix()
        T_r = T.dvector()
        T_absorbing = T.dvector()
        # T_r = T.dmatrix()
        T_discrete_actions = T.dmatrix()

        # store models of bellman apx and Q-function
        self.bellman_model = bellman_model
        self.q_model = q_model
        self.steps_ahead = steps_ahead

        # define bellman operator (check that BOP has only one output)
        assert isinstance(bellman_model.inputs, list)
        assert len(bellman_model.inputs) == 1
        assert isinstance(bellman_model.outputs, list)
        assert len(bellman_model.outputs) == 1

        # construct (theano) Bellman error
        self.theta_list = [bellman_model.inputs[0]]
        if not independent:
            self.T_bellman_err, _ = self.k_step_bellman_error(
                T_s, T_a, T_s_next, T_r, T_absorbing,
                self.theta_list[0], gamma, T_discrete_actions, steps_ahead)
            assert len(self.theta_list) == 1
        else:
            self.theta_list += [T.fmatrix(str(ll)) for ll in
                                range(
                                    steps_ahead - 1)]  # theta_0, theta_1, ..., theta_steps
            T_bellman_err = None
            for theta in self.theta_list:
                if T_bellman_err is None:
                    T_bellman_err = self.bellman_error(
                        T_s, T_a, T_s_next, T_r, theta,
                        gamma, T_discrete_actions)[0]
                else:
                    T_bellman_err = T_bellman_err + \
                                    self.bellman_error(
                                        T_s, T_a, T_s_next, T_r, theta,
                                        gamma, T_discrete_actions)[0]
            self.T_bellman_err = T_bellman_err
            assert len(self.theta_list) == steps_ahead

        # define function to be used for train and drawing actions
        self.train_function = None
        self.draw_action_function = None

        self.T_s = T_s
        self.T_a = T_a
        self.T_s_next = T_s_next
        self.T_r = T_r
        self.T_discrete_actions = T_discrete_actions
        self.T_absorbing = T_absorbing

        # get keras optimizer
        self.optimizer = optimizers.get(optimizer)

        # validate input data (the output is a list storing the validated input)
        self.discrete_actions = standardize_input_data(
            discrete_actions, ['discrete_actions'],
            [(None, self.action_dim)] if self.action_dim is not None else None,
            exception_prefix='discrete_actions')

        if isinstance(term_condition, str):
            self.term_condition = DEFAULT_TERM[term_condition]
        else:
            self.term_condition = term_condition

    def _compute_max_q(self, s, discrete_actions, theta):
        """
        Compute the maximum value of the Q-function in the given state.

        Args:
            s (theano.matrix): the state matrix (1 x state_dim)
            discrete_actions (theano.matrix): the discrete actions (nactions x action_dim)
            theta (theano.matrix): the parameters of the Q-function (1 x n_q_params)

        Returns:
            The maximum values of the Q-function in the given state w.r.t. all the discrete actions

        """
        q_values, _ = theano.scan(
            fn=lambda a, s, theta: self.q_model.model(s, a, theta),
            sequences=[discrete_actions], non_sequences=[s, theta])
        return T.max(q_values)
        # return q_values

    def _compute_argmax_q(self, s, discrete_actions, theta):
        """
        Compute the index of the action with the maximum Q-value in the given state.

        Args:
            s (theano.matrix): the state matrix (1 x state_dim)
            discrete_actions (theano.matrix): the discrete actions (nactions x action_dim)
            theta (theano.matrix): the parameters of the Q-function (1 x n_q_params)

        Returns:
            The index of the action that maximixes the Q-function in the given state
        """
        q_values, _ = theano.scan(
            fn=lambda a, s, theta: self.q_model.model(s, a, theta),
            sequences=[discrete_actions], non_sequences=[s, theta])
        return T.argmax(q_values)

    def bellman_error(self, s, a, nexts, r, absorbing,
                      theta, gamma, discrete_actions):
        """
        Compute the symbolic expression of the Bellman error.

        Args:
            s (theano.matrix): the state matrix (nsamples x state_dim)
            a (theano.matrix): the action matrix (nsamples x state_dim)
            nexts (theano.matrix): the next state matrix (nsamples x state_dim)
            r (theano.vector): the reward vector (nsamples)
            abserbing (theano.vector): the flag for absorbing states (nsamples)
            theta (theano.matrix): the parameters of the Q-function (1 x n_q_params)
            gamma (float): discount factor
            discrete_actions (theano.matrix): the discrete actions (nactions x action_dim)

        Returns:
            err (theano): The theano expression of the Bellman error
            theta_tp1 (theano): New point obtained evaluating the approximate PBO
        """
        # compute new parameters
        # theta_tp1 = self.bellman_model.outputs[0]
        theta_tp1 = self.bellman_model.model(theta)
        if self.incremental:
            theta_tp1 = theta_tp1 + theta
        # compute q-function with new parameters
        qbpo = self.q_model.model(s, a, theta_tp1)

        # compute max over actions with old parameters
        qmat, _ = theano.scan(fn=self._compute_max_q,
                              sequences=[nexts],
                              non_sequences=[discrete_actions, theta])

        # compute empirical BOP
        v = qbpo - r - gamma * qmat * (1. - absorbing)
        # compute error
        if self.norm_value == np.inf:
            err = T.max(v ** 2)
            # err = T.max(abs(v))
        elif self.norm_value % 2 == 0:
            err = T.sum(v ** self.norm_value) ** (1. / self.norm_value)
        else:
            err = T.sum(abs(v) ** self.norm_value) ** (1. / self.norm_value)
        return err, theta_tp1

    def k_step_bellman_error(self, s, a, nexts, r, absorbing,
                             theta, gamma, discrete_actions, steps):
        steps = max(1, steps)
        loss, theta_k = self.bellman_error(
            s, a, nexts, r, absorbing, theta, gamma, discrete_actions)
        for _ in range(steps - 1):
            err_k, theta_k = self.bellman_error(
                s, a, nexts, r, absorbing, theta_k, gamma, discrete_actions)
            loss += err_k
        return loss, theta_k

    def _make_train_function(self):
        """
        Construct the python train function from theano to be used in the fit process
        Returns:
            None
        """
        if self.train_function is None:
            print('compiling train function...')
            start = time.time()
            inputs = [self.T_s, self.T_a, self.T_s_next, self.T_r,
                      self.T_absorbing] + self.theta_list \
                     + [self.T_discrete_actions]

            training_updates = self.optimizer.get_updates(
                self.bellman_model.trainable_weights,
                {}, self.T_bellman_err)

            # returns loss and metrics. Updates weights at each call.
            self.train_function = theano.function(
                inputs, [self.T_bellman_err], updates=training_updates)
            print('compiled in {}s'.format(time.time() - start))

    def _standardize_user_data(self, s, a, s_next, r, absorbing, theta,
                               check_batch_dim=False):
        """

        Args:
            s (numpy.array): the samples of the state (nsamples, state_dim)
            a (numpy.array): the samples of the state (nsamples, action_dim)
            s_next (numpy.array): the samples of the next (reached) state (nsamples, state_dim)
            r (numpy.array): the sample of the reward (nsamples, )
            theta (numpy.array): the sample of the Q-function parameters (1, n_params)
            check_batch_dim (bool): default False

        Returns:
            The standardized values (s, a, s_next, r, theta)

        """
        s = standardize_input_data(
            s, ['s'],
            [(None, self.state_dim)] if self.state_dim is not None else None,
            exception_prefix='state')
        a = standardize_input_data(
            a, ['a'],
            [(None, self.action_dim)] if self.action_dim is not None else None,
            exception_prefix='action')
        # r = standardize_input_data(r, ['r'], [(None, 1)],
        #                            check_batch_dim=False, exception_prefix='reward')
        s_next = standardize_input_data(
            s_next, ['s_next'],
            [(None, self.state_dim)] if self.state_dim is not None else None,
            exception_prefix='state_next')
        theta = standardize_input_data(theta, ['theta'],
                                       (None, self.bellman_model.n_inputs()),
                                       exception_prefix='theta')
        check_array_lengths(s, a, s_next)
        return s, a, s_next, r, absorbing, theta

    def fit(self, s, a, s_next, r, absorbing, theta,
            batch_size=32, nb_epoch=10, shuffle=True,
            theta_metrics={}):
        """

        Args:
            s (numpy.array): the samples of the state (nsamples, state_dim)
            a (numpy.array): the samples of the state (nsamples, action_dim)
            s_next (numpy.array): the samples of the next (reached) state (nsamples, state_dim)
            r (numpy.array): the sample of the reward (nsamples, )
            theta (numpy.array): the sample of the Q-function parameters (1, n_params)
            batch_size (int): dimension of the batch used for a single step of the gradient
            nb_epoch (int): number of epochs
            verbose (int): 0 or 1. Verbosity mode. 0 = silent, 1 = verbose.
            callbacks (list): list of callbacks to be called during training.
                See [Keras Callbacks](https://keras.io/callbacks/).
            validation_split (float): float between 0 and 1:
                fraction of the training data to be used as validation data.
                The model will set apart this fraction of the training data,
                will not train on it, and will evaluate the loss and any model metrics
                on this data at the end of each epoch.
            validation_data (tuple): data on which to evaluate the loss and any model metrics
                at the end of each epoch. The model will not be trained on this data.
                This could be a tuple (val_s, val_a, val_s_next, val_r) or a tuple
                (val_s, val_a, val_s_next, val_r, val_theta).
            shuffle (boolean): whether to shuffle the training data before each epoch.
            theta_metrics (dict): dictionary storing the pairs (name: callable object).
                The callable object/function is used to evaluate the Q-function parameters
                at each iteration. The signature of the callable is simple: f(theta)
                e.g.: theta_metrics={'k': lambda theta: evaluate(theta)})

        Returns:
            A PBOHistory instance storing train information
        """
        s, a, s_next, r, absorbing, theta = self._standardize_user_data(
            s, a, s_next, r, absorbing, theta,
            check_batch_dim=False
        )

        all_actions = standardize_input_data(
            self.discrete_actions, ['all_actions'],
            [(None, self.action_dim)] if self.action_dim is not None else None,
            exception_prefix='discrete_actions')

        n_updates = 0
        history = {"theta": [], 'rho': []}
        for k in theta_metrics.keys():
            history.update({k: []})

        ins = s + a + s_next + [r, absorbing]
        self._make_train_function()
        f = self.train_function

        nb_train_sample = ins[0].shape[0]
        index_array = np.arange(nb_train_sample)

        # append evolution of theta for independent case
        for _ in range(len(self.theta_list) - 1):
            if self.incremental:
                tmp = theta[-1] + self.bellman_model.predict(theta[-1])
            else:
                tmp = self.bellman_model.predict(theta[-1])
            theta += [tmp]

        term_condition = self.term_condition
        stop = False
        old_theta = theta

        for epoch in range(nb_epoch):
            if stop:
                break

            if shuffle == 'batch':
                index_array = batch_shuffle(index_array, batch_size)
            elif shuffle:
                np.random.shuffle(index_array)
            batches = make_batches(nb_train_sample, batch_size)
            for batch_index, (batch_start, batch_end) in enumerate(batches):

                history["theta"].append(theta[0])
                if hasattr(self.bellman_model, '_model'):
                    history["rho"].append(
                        self.bellman_model._model.get_weights())
                else:
                    history["rho"].append(self.bellman_model.get_weights())
                for k, v in iteritems(theta_metrics):
                    history[k].append(v(theta))

                batch_ids = index_array[batch_start:batch_end]
                try:
                    if type(ins[-1]) is float:
                        # do not slice the training phase flag
                        ins_batch = slice_X(ins[:-1], batch_ids) + [ins[-1]]
                    else:
                        ins_batch = slice_X(ins, batch_ids)
                except TypeError:
                    raise Exception('TypeError while preparing batch. '
                                    'If using HDF5 input data, '
                                    'pass shuffle="batch".')
                inp = ins_batch + theta + all_actions
                outs = f(*inp)
                n_updates += 1

                if self.update_theta_every > 0 and n_updates % self.update_theta_every == 0:
                    tmp = self.apply_bo(theta[0],
                                        n_times=self.steps_per_theta_update)
                    theta = [tmp]
                    for _ in range(len(self.theta_list) - 1):
                        if self.incremental:
                            tmp = tmp + self.bellman_model.predict(tmp)
                        else:
                            tmp = self.bellman_model.predict(tmp)
                        theta += [tmp]

                    if term_condition is not None:
                        stop = term_condition(old_theta, theta)
                        if stop:
                            break
                        old_theta = theta

        # finally apply the bellman operator K-times to get the final point
        self.learned_theta_value = self.apply_bo(theta[0], n_times=100)
        if self.verbose > 1:
            print('learned theta: {}'.format(self.learned_theta_value))

        self.history = history
        return history

    def apply_bo(self, theta, n_times=1):
        """
        Applies the Bellman operator to the provided Q-function parameters
        Args:
            theta (numpy.array): the sample of the Q-function parameters (1, n_params)

        Returns:
            The updated parameters

        """
        for _ in range(n_times):
            if self.incremental:
                theta = theta + self.bellman_model.predict(theta)
            else:
                theta = self.bellman_model.predict(theta)
        return theta

    def _make_draw_action_function(self):
        """
        Construct the python function from theano to be used in the draw_action process
        Returns:
            None
        """
        if self.draw_action_function is None:
            # compute max over actions with old parameters
            theta = self.bellman_model.inputs[0]
            idx_max, _ = theano.scan(fn=self._compute_argmax_q,
                                     sequences=[self.T_s],
                                     non_sequences=[self.T_discrete_actions,
                                                    theta])
            inputs = [self.T_s, theta, self.T_discrete_actions]
            self.draw_action_function = theano.function(inputs, [
                self.T_discrete_actions[idx_max]])

    def draw_action(self, state, done, flag):
        """
        Samples the action to be executed.
        Args:
            state (numpy.array): the state to be evaluated (1, state_dim) or (state_dim,)
            done: ??
            flag: ??

        Returns:
            The action to be executed in the state
        """
        self._make_draw_action_function()
        state = standardize_input_data(state, ['state'],
                                       [(None,
                                         self.state_dim)] if self.state_dim is not None else None,
                                       exception_prefix='draw_state')
        return self.draw_action_function(state[0], self.learned_theta_value,
                                         self.discrete_actions[
                                             0])  # we take index zero since they are lists of numpy matrices

    def _make_additional_functions(self):
        # get trainable parameters
        params = self.bellman_model.trainable_weights
        theta = self.bellman_model.inputs[0]
        # compute (theano) gradient
        self.T_grad_bellman_err = T.grad(self.T_bellman_err, params)
        # compile all functions
        # self.F_bellman_operator = theano.function([theta], self.bellman_model.outputs[0])
        self.F_bellman_operator = theano.function([theta],
                                                  self.bellman_model.model(
                                                      theta))
        self.F_q = theano.function([self.T_s, self.T_a, theta],
                                   self.q_model.model(self.T_s, self.T_a,
                                                      theta))
        self.F_bellman_err = theano.function(
            [self.T_s, self.T_a, self.T_s_next, self.T_r, self.T_absorbing,
             theta, self.T_discrete_actions], self.T_bellman_err)
        self.F_grad_bellman_berr = theano.function(
            [self.T_s, self.T_a, self.T_s_next, self.T_r, self.T_absorbing,
             theta, self.T_discrete_actions],
            self.T_grad_bellman_err)
