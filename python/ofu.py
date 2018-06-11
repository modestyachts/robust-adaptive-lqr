"""ofu.py

An attempt to implement OFU.


"""

import numpy as np
import scipy.linalg
import logging
import time
import cvxpy as cvx

import utils

from adaptive import AdaptiveMethod


def function_value(Q, R, A, B):
    P, K = utils.dlqr(A, B, Q, R)
    return np.trace(P)

def gradient(Q, R, A, B):
    """Compute the gradient of th -> Tr(P(th))

    It is assumed that Q, R are both invertible.

    """

    P, K = utils.dlqr(A, B, Q, R)
    A_c = A + B.dot(K)

    n, p = B.shape

    ret = np.zeros((n, n + p))

    # TODO: we should invert the operator X -> A_c^T X A_c - X
    # once and then use it to solve many linear equations,
    # rather than repeating the inversion many times

    # See Eq. (13) of
    # https://arxiv.org/pdf/1703.08972.pdf
    for idx in range(n):
        for jdx in range(n + p):
            U = np.zeros((n, n + p))
            U[idx, jdx] = 1
            target = A_c.T.dot(P.dot(U)).dot(np.vstack((np.eye(n), K)))
            target += target.T
            DU = utils.solve_discrete_lyapunov(A_c, target)
            ret[idx, jdx] = np.trace(DU)

    return ret


def ofu_pgd(Q, R, Ahat, Bhat, projection_operator, logger=None,
            step_size=0.01, max_iters=100, rel_tol=1e-6, show_every=100,
            num_restarts=0, rng=None, div_tol=-0.5):
    """Solve the OFU problem via projected gradient descent.

    The inputs (Ahat, Bhat) represent the starting point.
    The projection operator is a function which takes
    inputs A, B and returns the (Euclidean) projection onto that set

    """

    assert len(Q.shape) == 2 and Q.shape[0] == Q.shape[1]
    assert len(R.shape) == 2 and R.shape[0] == R.shape[1]
    assert len(Ahat.shape) == 2 and Ahat.shape[0] == Ahat.shape[1]
    assert len(Bhat.shape) == 2 and Bhat.shape[0] == Ahat.shape[0]
    assert Q.shape[0] == Ahat.shape[0]
    assert R.shape[0] == Bhat.shape[1]

    if logger is None:
        logger = logging.getLogger(__name__)
    if rng is None:
        rng = np.random

    n, p = Bhat.shape

    # non-convex projected gradient descent

    restart = 0
    fvals = []
    Acur, Bcur = np.array(Ahat), np.array(Bhat)
    Abest, Bbest = Acur, Bcur

    while restart <= num_restarts:
        start_time = time.time()
        proj_time = 0

        Acur, Bcur = np.array(Ahat), np.array(Bhat)
        if restart > 0:
            # TODO: more sophisticated perturbations
            Acur += rng.normal(size=Acur.shape) 
            Bcur += rng.normal(size=Bcur.shape) 
            Acur, Bcur = projection_operator(Acur, Bcur)
        fval_cur = function_value(Q, R, Acur, Bcur)

        logger.debug("starting with fval={} ({} out of {} restarts)".format(fval_cur, restart, num_restarts))

        for iter_idx in range(max_iters):

            # compute gradient
            grad = gradient(Q, R, Acur, Bcur)

            # step
            Delta_A = grad[:, :n]
            Delta_B = grad[:, n:]

            Anext = Acur - step_size * Delta_A
            Bnext = Bcur - step_size * Delta_B

            # TODO: line search instead

            # project
            proj_start_time = time.time()
            Anext, Bnext = projection_operator(Anext, Bnext)
            proj_time += time.time() - proj_start_time

            fval_next = function_value(Q, R, Anext, Bnext)

            rel_decrease = (fval_cur - fval_next)/fval_cur

            if np.abs(rel_decrease) <= rel_tol:
                logger.debug("exiting because rel_decrease={}, final fval={} ({} out of {} restarts)".format(rel_decrease, fval_cur, restart, num_restarts))
                break

            if rel_decrease <= div_tol:
                logger.warn("rel_decrease={} indicating divergence".format(rel_decrease, fval_cur, restart, num_restarts))

            if not (iter_idx % show_every):
                logger.debug("iteration {}, fprev={}, fcur={}, ||grad||={}, elapsed_time={}, proj_time={}".format(
                    iter_idx, fval_cur, fval_next,
                    np.linalg.norm(grad, ord="fro"),
                    time.time() - start_time, proj_time))

            fval_cur = fval_next
            Acur = Anext
            Bcur = Bnext

        if iter_idx == max_iters - 1:
            logger.debug("max iterations exceeded, final fval={} ({} out of {} restarts)".format(fval_cur, restart, num_restarts))

        fvals.append(fval_cur)
        if fval_cur <= min(fvals):
            Abest, Bbest = Acur, Bcur
        restart += 1
    return Abest, Bbest


class OFUStrategy(AdaptiveMethod):
    """Adaptive control based on OFU

    """

    def __init__(self, Q, R, A_star, B_star, sigma_w, rls_lam,
                 reg, actual_error_multiplier, num_restarts=0):
        super().__init__(Q, R, A_star, B_star, sigma_w, rls_lam)
        self._reg = reg
        self._actual_error_multiplier = actual_error_multiplier
        self._logger = logging.getLogger(__name__)
        self._num_restarts = num_restarts

    def _get_logger(self):
        return self._logger

    def _design_controller(self, states, inputs, transitions, rng):
        logger = self._get_logger()
        logger.debug("_design_controller: have {} points for regression".format(inputs.shape[0]))

        # TODO(stephentu):
        # Currently I am using the algorithm of Abbasi-Yadkori and Szepesvari.
        # We should also try the subtly different algorithm in
        # https://arxiv.org/pdf/1711.07230.pdf.

        # fit the data
        Anom, Bnom, emp_cov = utils.solve_least_squares(states, inputs, transitions, reg=self._reg)

        if not self._has_primed:
            self._emp_cov = np.array(emp_cov)
            self._last_emp_cov = np.array(emp_cov)

        emp_cov /= inputs.shape[0] # normalize by T to improve numerics

        theta_nom = np.hstack((Anom, Bnom))
        theta_star = np.hstack((self._A_star, self._B_star))
        delta = theta_nom - theta_star
        actual_error = np.trace(delta.dot(emp_cov.dot(delta.T)))
        eps = self._actual_error_multiplier * actual_error
        logger.info("_design_controller: actual weighted error is {}, eps is {}".format(actual_error, eps))

        n, p = self._n, self._p

        def projection_operator(A, B):
            M = np.hstack((A, B))
            theta = utils.project_weighted_ball(M, theta_nom, emp_cov, eps)
            return theta[:, :n], theta[:, n:]

        A_ofu, B_ofu = ofu_pgd(
                Q=self._Q,
                R=self._R,
                Ahat=Anom,
                Bhat=Bnom,
                projection_operator=projection_operator,
                logger=logger,
                num_restarts=self._num_restarts)

        theta_ofu = np.hstack((A_ofu, B_ofu))
        delta_ofu = theta_ofu - theta_nom
        TOL = 1e-5
        assert np.trace(delta_ofu.dot(emp_cov.dot(delta_ofu.T))) <= eps + TOL

        _, K = utils.dlqr(A_ofu, B_ofu, self._Q, self._R)
        self._current_K = K

        # compute the infinite horizon cost of this controller
        Jnom = utils.LQR_cost(self._A_star, self._B_star, self._current_K, self._Q, self._R, self._sigma_w)

        # for debugging purposes,
        # check to see if this controller will stabilize the true system
        rho_true = utils.spectral_radius(self._A_star + self._B_star.dot(self._current_K))
        logger.info("_design_controller(epoch={}): rho(A_* + B_* K)={}".format(
            self._epoch_idx + 1 if self._has_primed else 0,
            rho_true))

        return (Anom, Bnom, Jnom)

    def _on_iteration_completion(self):
        # this is called after we take a step
        zt = np.hstack((self._state_history[-1], self._input_history[-1]))
        self._emp_cov += np.outer(zt, zt)

    def _on_epoch_completion(self):
        self._last_emp_cov = np.array(self._emp_cov) # need to make a copy

    def _should_terminate_epoch(self):

        # hack: otherwise in the beginning the epochs are very short
        min_epoch_time = 10

        # TODO(stephentu): what is the best numerical recipe for this
        # calculation?
        if (self._iteration_within_epoch_idx >= min_epoch_time) and \
                (np.linalg.det(self._emp_cov) > 2 * np.linalg.det(self._last_emp_cov)):
            # start new epoch
            return True
        else:
            # keep going
            return False

    def _get_input(self, state, rng):
        # no exploration
        return self._current_K.dot(state)


def _main():
    import examples
    A_star, B_star = examples.unstable_laplacian_dynamics()

    # perturb Ahat, Bhat
    eps_A = 0.1
    eps_B = 0.1
    Ahat = utils.sample_2_to_2_ball(Ahat, eps_A)
    Bhat = utils.sample_2_to_2_ball(Bhat, eps_B)

    Q = 1e-3 * np.eye(3)
    R = np.eye(3)

    Acur, Bcur = ofu_pgd(Q, R, Ahat, Bhat, eps_A, eps_B, step_size=1, max_iters=1000)

    print("Acur")
    print(Acur)
    print("Bcur")
    print(Bcur)

def _main():
    import examples
    A_star, B_star = examples.unstable_laplacian_dynamics()

    # define costs
    Q = 1e-3 * np.eye(3)
    R = np.eye(3)

    # initial controller
    _, K_init = utils.dlqr(A_star, B_star, 1e-3*np.eye(3), np.eye(3))

    rng = np.random

    env = OFUStrategy(Q=Q,
                      R=R,
                      A_star=A_star,
                      B_star=B_star,
                      sigma_w=1,
                      reg=1e-5,
                      actual_error_multiplier=1, 
                      rls_lam=None)

    env.reset(rng)
    env.prime(100, K_init, 0.1, rng)
    for idx in range(500):
        env.step(rng)

if __name__ == '__main__':
    logging.basicConfig(level=logging.DEBUG)
    np.set_printoptions(linewidth=200)
    _main()

