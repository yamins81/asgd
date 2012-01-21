"""Averaging Stochastic Gradient Descent Classifier

naive, non-optimized implementation
"""

import copy
from itertools import izip

import numpy as np
from numpy import dot

DEFAULT_SGD_STEP_SIZE0 = None
DEFAULT_L2_REGULARIZATION = 1e-3
DEFAULT_N_ITERATIONS = 10
DEFAULT_FEEDBACK = False
DEFAULT_RSTATE = 42
DEFAULT_DTYPE = np.float64
DEFAULT_SGD_EXPONENT = 2.0 / 3.0
DEFAULT_SGD_TIMESCALE = 'l2_regularization'
# can be 'l2_regularization' or float
# This timescale default comes from [1] in which it is introduced as a
# heuristic.
# [1] http://www.dbs.ifi.lmu.de/~yu_k/cvpr11_0694.pdf
# Update: it is also recommended in Leon Bottou's SvmAsgd software.


class BaseASGD(object):
    """
    XXX
    """

    min_n_iterations = 5

    def __init__(self, n_features,
                 sgd_step_size0=DEFAULT_SGD_STEP_SIZE0,
                 l2_regularization=DEFAULT_L2_REGULARIZATION,
                 n_iterations=DEFAULT_N_ITERATIONS,
                 feedback=DEFAULT_FEEDBACK,
                 rstate=DEFAULT_RSTATE,
                 dtype=DEFAULT_DTYPE,
                 sgd_step_size_scheduling_exponent = DEFAULT_SGD_EXPONENT,
                 sgd_step_size_scheduling_multiplier = DEFAULT_SGD_TIMESCALE):

        # --
        assert n_features > 1
        self.n_features = n_features

        assert n_iterations > 0
        self.n_iterations = n_iterations

        self.min_n_iterations = min(n_iterations, self.min_n_iterations)

        if feedback:
            raise NotImplementedError("FIXME: feedback support is buggy")
        self.feedback = feedback

        if rstate is None:
            rstate = np.random.RandomState()
        elif type(rstate) is int:
            rstate = np.random.RandomState(rstate)
        self.rstate = rstate

        self.l2_regularization = l2_regularization
        self.dtype = dtype

        # --
        self.sgd_step_size0 = sgd_step_size0
        self.sgd_step_size_scheduling_exponent = \
            sgd_step_size_scheduling_exponent
        if sgd_step_size_scheduling_multiplier == 'l2_regularization':
            self.sgd_step_size_scheduling_multiplier = l2_regularization
        else:
            self.sgd_step_size_scheduling_multiplier = \
                    sgd_step_size_scheduling_multiplier

        self.n_observations = 0
        self.asgd_step_size0 = 1
        self.asgd_step_size = self.asgd_step_size0
        self.sgd_step_size = self.sgd_step_size0
        self.train_means = []

    def fit_converged(self):
        train_means = self.train_means
        if len(train_means) >= self.min_n_iterations:
            midpt = len(train_means) // 2
            if train_means[-1] > .99 * train_means[midpt]:
                return True
        return False


class DetermineStepSizeMixin(object):
    """
    Implements the automatic step-size selection logic from
    http://leon.bottou.org/projects/sgd


    This mixin requires the host class to have

    self.partial_fit(X, y)
    self.n_observations
    self.sgd_step_size0
    self.sgd_weights
    self.sgd_bias
    self.asgd_weights
    self.asgd_bias

    """

    n_examples_for_determining_step_size = 1000
    verbose = 0

    def determine_sgd_step_size0(self, X, y, base=1.0, factor=2.0):
        # trim X and y down to at most 1000 examples
        def show(msg):
            if self.verbose:
                print(msg)
        X = X[:self.n_examples_for_determining_step_size]
        y = y[:self.n_examples_for_determining_step_size]
        lo_step_size0 = base
        lo_cost = self.evaluate_step_size(X, y, lo_step_size0)
        show('determine_sgd_step_size0: lo_cost = %f' % lo_cost)
        hi_step_size0 = base * factor
        hi_cost = self.evaluate_step_size(X, y, hi_step_size0)
        show('determine_sgd_step_size0: hi_cost = %f' % hi_cost)
        if lo_cost < hi_cost:
            # do a geometric search toward 0 for bottom of curve
            while lo_cost + 1e-4 < hi_cost:
                # bring down hi_step_size0
                hi_step_size0 = lo_step_size0
                hi_cost = lo_cost
                lo_step_size0 = hi_step_size0 / factor
                lo_cost = self.evaluate_step_size(X, y, lo_step_size0);
                show('determine_sgd_step_size0: lo_size0 = %f, lo_cost = %f' %
                        (lo_step_size0, lo_cost))
        elif hi_cost < lo_cost:
            while hi_cost + 1e-4 < lo_cost:
                # bring up lo_step_size0
                lo_step_size0 = hi_step_size0
                lo_cost = hi_cost
                hi_step_size0 = lo_step_size0 * factor
                hi_cost = self.evaluate_step_size(X, y, hi_step_size0)
                show('determine_sgd_step_size0: hi_size0 = %f, hi_cost = %f' %
                        (hi_step_size0, hi_cost))
        show('determine_sgd_step_size0: final step size %f' %
                lo_step_size0)
        self.sgd_step_size0 = lo_step_size0
        self.sgd_step_size = lo_step_size0

    def evaluate_step_size(self, X, y, sgd_step_size0):
        other = copy.deepcopy(self)
        other.sgd_step_size0 = sgd_step_size0
        other.sgd_step_size = sgd_step_size0
        other.partial_fit(X, y)
        # XXX: hack - asgd is lower variance than sgd, but it's tuned to work
        #             well asymptotically, not after just a few examples
        weights = .5 * other.asgd_weights + .5 * other.sgd_weights
        bias = .5 * other.asgd_bias + .5 * other.sgd_bias
        margin = y * (dot(X, weights) + bias)
        l2_cost = other.l2_regularization * (weights ** 2).sum()
        cost = np.maximum(0, 1 - margin) + l2_cost
        return cost.mean()


class NaiveBinaryASGD(BaseASGD, DetermineStepSizeMixin):
    """
    XXX
    """

    def __init__(self, n_features, sgd_step_size0=DEFAULT_SGD_STEP_SIZE0,
                 l2_regularization=DEFAULT_L2_REGULARIZATION,
                 n_iterations=DEFAULT_N_ITERATIONS, feedback=DEFAULT_FEEDBACK,
                 rstate=DEFAULT_RSTATE, dtype=DEFAULT_DTYPE):

        super(NaiveBinaryASGD, self).__init__(
            n_features,
            sgd_step_size0=sgd_step_size0,
            l2_regularization=l2_regularization,
            n_iterations=n_iterations,
            feedback=feedback,
            rstate=rstate,
            dtype=dtype,
            )

        # --
        self.sgd_weights = np.zeros((n_features), dtype=dtype)
        self.sgd_bias = np.zeros((1), dtype=dtype)

        self.asgd_weights = np.zeros((n_features), dtype=dtype)
        self.asgd_bias = np.zeros((1), dtype=dtype)


    def partial_fit(self, X, y, margin_biases=None):
        assert np.all(y ** 2 == 1)  # make sure labels are +-1
        sgd_step_size0 = self.sgd_step_size0
        sgd_step_size = self.sgd_step_size
        sgd_step_size_scheduling_exponent = \
                self.sgd_step_size_scheduling_exponent
        sgd_step_size_scheduling_multiplier = \
                self.sgd_step_size_scheduling_multiplier
        sgd_weights = self.sgd_weights
        sgd_bias = self.sgd_bias

        asgd_weights = self.asgd_weights
        asgd_bias = self.asgd_bias
        asgd_step_size = self.asgd_step_size

        l2_regularization = self.l2_regularization

        n_observations = self.n_observations

        costs = []

        if margin_biases is None:
             margin_biases = np.zeros((len(y),))
             
        for obs, label, margin_bias in izip(X, y, margin_biases):

            # -- compute margin
            margin = label * (dot(obs, sgd_weights) + sgd_bias)

            # -- update sgd
            if l2_regularization:
                sgd_weights *= (1 - l2_regularization * sgd_step_size)

            if margin < 1 - margin_bias:
                sgd_weights += sgd_step_size * label * obs
                sgd_bias += sgd_step_size * label
                costs.append(1 - float(margin))
            else:
                costs.append(0)

            # -- update asgd
            asgd_weights = (1 - asgd_step_size) * asgd_weights \
                    + asgd_step_size * sgd_weights
            asgd_bias = (1 - asgd_step_size) * asgd_bias \
                    + asgd_step_size * sgd_bias

            # 4.1 update step_sizes
            n_observations += 1
            sgd_step_size_scheduling = (1 + sgd_step_size0 * n_observations *
                                        sgd_step_size_scheduling_multiplier)
            sgd_step_size = sgd_step_size0 / \
                    (sgd_step_size_scheduling ** \
                     sgd_step_size_scheduling_exponent)
            asgd_step_size = 1. / n_observations

        # --
        self.sgd_weights = sgd_weights
        self.sgd_bias = sgd_bias
        self.sgd_step_size = sgd_step_size

        self.asgd_weights = asgd_weights
        self.asgd_bias = asgd_bias
        self.asgd_step_size = asgd_step_size

        self.n_observations = n_observations

        self.train_means.append(np.mean(costs)
                + self.l2_regularization * (self.asgd_weights ** 2).sum())

        return self

    def fit(self, X, y, margin_biases=None):
        assert X.ndim == 2
        assert y.ndim == 1

        n_points, n_features = X.shape
        assert n_features == self.n_features
        assert n_points == y.size

        n_iterations = self.n_iterations

        if margin_biases is None:
             margin_biases = np.zeros((len(y),))

        if self.sgd_step_size0 is None:
            self.determine_sgd_step_size0(X, y)

        for i in xrange(n_iterations):

            idx = self.rstate.permutation(n_points)
            Xb = X[idx]
            yb = y[idx]
            wb = w[idx]
            mb = margin_biases[idx]
            self.partial_fit(Xb, yb, margin_biases=mb)

            if self.feedback:
                self.sgd_weights = self.asgd_weights
                self.sgd_bias = self.asgd_bias

            if self.fit_converged():
                break

        return self

    def decision_function(self, X):
        return dot(self.asgd_weights, X.T) + self.asgd_bias

    def predict(self, X):
        return np.sign(self.decision_function(X))

    def reset(self):
        BaseASGD.reset(self)
        self.asgd_weights = self.asgd_weights * 0
        self.asgd_bias = self.asgd_bias * 0
        self.sgd_weights = self.sgd_weights * 0
        self.sgd_bias = self.sgd_bias * 0


class NaiveOVAASGD(BaseASGD):
    """
    XXX
    """

    def __init__(self, n_classes, n_features,
                 sgd_step_size0=DEFAULT_SGD_STEP_SIZE0,
                 l2_regularization=DEFAULT_L2_REGULARIZATION,
                 n_iterations=DEFAULT_N_ITERATIONS,
                 feedback=DEFAULT_FEEDBACK,
                 rstate=DEFAULT_RSTATE,
                 dtype=DEFAULT_DTYPE):

        super(NaiveOVAASGD, self).__init__(
            n_features,
            sgd_step_size0=sgd_step_size0,
            l2_regularization=l2_regularization,
            n_iterations=n_iterations,
            feedback=feedback,
            rstate=rstate,
            dtype=dtype,
            )

        # --
        assert n_classes > 1
        self.n_classes = n_classes

        # --
        self.sgd_weights = np.zeros((n_features, n_classes), dtype=dtype)
        self.sgd_bias = np.zeros((n_classes,), dtype=dtype)
        self.asgd_weights = np.zeros((n_features, n_classes), dtype=dtype)
        self.asgd_bias = np.zeros((n_classes), dtype=dtype)

    def partial_fit(self, X, y, margin_biases=None):

        if set(y) > set(range(self.n_classes)):
            raise ValueError("Invalid 'y'")

        sgd_step_size0 = self.sgd_step_size0
        sgd_step_size = self.sgd_step_size
        sgd_step_size_scheduling_exponent = \
                self.sgd_step_size_scheduling_exponent
        sgd_step_size_scheduling_multiplier = \
                self.sgd_step_size_scheduling_multiplier
        sgd_weights = self.sgd_weights
        sgd_bias = self.sgd_bias

        asgd_weights = self.asgd_weights
        asgd_bias = self.asgd_bias
        asgd_step_size = self.asgd_step_size

        l2_regularization = self.l2_regularization

        n_observations = self.n_observations
        n_classes = self.n_classes

        if margin_biases is None:
             margin_biases = np.zeros((len(y),))
             
        for obs, label, margin_bias in izip(X, y, margin_biases):
            label = 2 * (np.arange(n_classes) == label).astype(int) - 1
            
            # -- compute margin
            margin = label * (dot(obs, sgd_weights) + sgd_bias)

            # -- update sgd
            if l2_regularization:
                sgd_weights *= (1 - l2_regularization * sgd_step_size)

            violations = margin < 1 - margin_bias
            label_violated = label[violations]
            sgd_weights[:, violations] += (
                sgd_step_size
                * label_violated[np.newaxis, :]
                * obs[:, np.newaxis]
            )
            sgd_bias[violations] += sgd_step_size * label_violated

            # -- update asgd
            asgd_weights = (1 - asgd_step_size) * asgd_weights \
                    + asgd_step_size * sgd_weights
            asgd_bias = (1 - asgd_step_size) * asgd_bias \
                    + asgd_step_size * sgd_bias

            # -- update step_sizes
            n_observations += 1
            sgd_step_size_scheduling = (1 + sgd_step_size0 * n_observations *
                                        sgd_step_size_scheduling_multiplier)
            sgd_step_size = sgd_step_size0 / \
                    (sgd_step_size_scheduling ** \
                     sgd_step_size_scheduling_exponent)
            asgd_step_size = 1. / n_observations

        # --
        self.sgd_weights = sgd_weights
        self.sgd_bias = sgd_bias
        self.sgd_step_size = sgd_step_size

        self.asgd_weights = asgd_weights
        self.asgd_bias = asgd_bias
        self.asgd_step_size = asgd_step_size

        self.n_observations = n_observations

        return self

    def fit(self, X, y, margin_biases=None):

        assert X.ndim == 2
        assert y.ndim == 1

        n_points, n_features = X.shape
        assert n_features == self.n_features
        assert n_points == y.size

        n_iterations = self.n_iterations
        

        if margin_biases is None:
             margin_biases = np.zeros((len(y),))        

        for i in xrange(n_iterations):

            idx = self.rstate.permutation(n_points)
            Xb = X[idx]
            yb = y[idx]
            mb = margin_biases=[idx]
            self.partial_fit(Xb, yb, margin_biases=mb)
            
            if self.feedback:
                self.sgd_weights = self.asgd_weights
                self.sgd_bias = self.asgd_bias

        return self

    def decision_function(self, X):
        return dot(X, self.asgd_weights) + self.asgd_bias

    def predict(self, X):
        return self.decision_function(X).argmax(1)
