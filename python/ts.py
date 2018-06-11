"""ts.py

"""

import numpy as np
import utils
import logging
import math

from adaptive import AdaptiveMethod


class TSStrategy(AdaptiveMethod):
    """Adaptive control based on Thompson sampling

    Based on the algorithm presented in Figure 1 of
    https://arxiv.org/pdf/1703.08972.pdf.

    """

    def __init__(self, Q, R, A_star, B_star, sigma_w, rls_lam,
                 reg, tau, actual_error_multiplier):
        super().__init__(Q, R, A_star, B_star, sigma_w, rls_lam)
        self._reg = reg
        self._tau = tau
        self._actual_error_multiplier = actual_error_multiplier
        self._logger = logging.getLogger(__name__)

    def _get_logger(self):
        return self._logger

    def reset(self, rng):
        super().reset(rng)
        self._emp_cov = self._reg * np.eye(self._n + self._p)
        self._last_emp_cov = self._reg * np.eye(self._n + self._p)

    def _design_controller(self, states, inputs, transitions, rng):

        logger = self._get_logger()

        epoch_id = self._epoch_idx + 1 if self._has_primed else 0

        logger.debug("_design_controller(epoch={}): have {} points for regression".format(epoch_id, inputs.shape[0]))

        # do a least squares fit and design based on the nominal
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
        logger.info("_design_controller(epoch={}): actual weighted error is {}, eps is {}".format(epoch_id, actual_error, eps))

        def is_contained_in_confidence_set(A, B):
            theta_ab = np.hstack((A, B))
            this_delta = theta_ab - theta_nom
            return np.trace(this_delta.dot(emp_cov).dot(this_delta.T)) <= eps

        inv_sqrt_emp_cov = utils.pd_inv_sqrt(emp_cov)
        MAX_TRIES = 100000
        rng = self._get_rng(rng)
        success = False
        for rejection_idx in range(MAX_TRIES):
            eta = rng.normal(size=theta_nom.shape)
            eta *= np.power(rng.uniform(), 1/(theta_nom.shape[0] * theta_nom.shape[1])) / np.linalg.norm(eta, ord="fro")
            theta_tilde = theta_nom + np.sqrt(eps) * eta.dot(inv_sqrt_emp_cov)
            A_tilde = theta_tilde[:, :self._n]
            B_tilde = theta_tilde[:, self._n:]
            if is_contained_in_confidence_set(A_tilde, B_tilde):
                A_ts = A_tilde
                B_ts = B_tilde
                success = True
                break

        if not success:
            logger.warn("_design_controller(epoch={}): was unable to rejection sample after {} attempts".format(epoch_id, MAX_TRIES))
            raise Exception("this is a very low probability event")

        else:
            logger.info("_design_controller(epoch={}): took {} attempts to rejection sample".format(epoch_id, rejection_idx + 1))

        _, K = utils.dlqr(A_ts, B_ts, self._Q, self._R)
        self._current_K = K

        # compute the infinite horizon cost of this controller
        Jnom = utils.LQR_cost(self._A_star, self._B_star, self._current_K, self._Q, self._R, self._sigma_w)

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
        assert self._tau > min_epoch_time, "make tau larger, or min_epoch_time smaller"

        if self._iteration_within_epoch_idx <= min_epoch_time:
            return False

        # TODO(stephentu): what is the best numerical recipe for this
        # calculation?
        if (np.linalg.det(self._emp_cov) > 2 * np.linalg.det(self._last_emp_cov)) or \
                (self._iteration_within_epoch_idx >= self._tau):
            # condition triggered
            return True
        else:
            # keep going
            return False

    def _get_input(self, state, rng):
        rng = self._get_rng(rng)
        ctrl_input = self._current_K.dot(state)
        return ctrl_input

def _main():
    import examples
    A_star, B_star = examples.unstable_laplacian_dynamics()

    # define costs
    Q = 1e-3 * np.eye(3)
    R = np.eye(3)

    # initial controller
    _, K_init = utils.dlqr(A_star, B_star, 1e-3*np.eye(3), np.eye(3))

    rng = np.random

    env = TSStrategy(Q=Q,
                     R=R,
                     A_star=A_star,
                     B_star=B_star,
                     sigma_w=1,
                     reg=1e-5,
                     tau=500,
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