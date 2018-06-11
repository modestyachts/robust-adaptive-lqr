"""nominal.py

"""

import numpy as np
import utils
import logging
import math

from adaptive import AdaptiveMethod


class NominalStrategy(AdaptiveMethod):
    """Adaptive control based on nominal estimates of the dynamics

    """

    def __init__(self,
                 Q,
                 R,
                 A_star,
                 B_star,
                 sigma_w,
                 rls_lam,
                 sigma_explore,
                 reg,
                 epoch_multiplier,
                 epoch_schedule='linear'):
        super().__init__(Q, R, A_star, B_star, sigma_w, rls_lam)
        self._sigma_explore = sigma_explore
        self._reg = reg
        self._epoch_multiplier = epoch_multiplier
        if not epoch_schedule in ('linear', 'exponential'):
            raise ValueError("invalid epoch_schedule: {}".format(epoch_schedule))
        self._epoch_schedule = epoch_schedule
        self._logger = logging.getLogger(__name__)

    def _get_logger(self):
        return self._logger

    def reset(self, rng):
        super().reset(rng)
        self._explore_stddev_history = []

    def _on_iteration_completion(self):
        self._explore_stddev_history.append(self._explore_stddev())

    def _design_controller(self, states, inputs, transitions, rng):

        logger = self._get_logger()

        # do a least squares fit and controller based on the nominal
        Anom, Bnom, _ = utils.solve_least_squares(states, inputs, transitions, reg=self._reg)
        _, K = utils.dlqr(Anom, Bnom, self._Q, self._R)
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

    def _epoch_length(self):
        if self._epoch_schedule == 'linear':
            return self._epoch_multiplier * (self._epoch_idx + 1)
        else:
            return self._epoch_multiplier * math.pow(2, self._epoch_idx)

    def _explore_stddev(self):
        if self._epoch_schedule == 'linear':
            sigma_explore_decay = 1/math.pow(self._epoch_idx + 1, 1/3)
            return self._sigma_explore * sigma_explore_decay
        else:
            sigma_explore_decay = 1/math.pow(2, self._epoch_idx/6)
            return self._sigma_explore * sigma_explore_decay

    def _should_terminate_epoch(self):
        if (self._iteration_within_epoch_idx >= self._epoch_length()):
            return True
        else:
            return False

    def _get_input(self, state, rng):
        rng = self._get_rng(rng)
        ctrl_input = self._current_K.dot(state)
        explore_input = self._explore_stddev() * rng.normal(size=(self._p,))
        return ctrl_input + explore_input


def _main():
    import examples
    A_star, B_star = examples.unstable_laplacian_dynamics()

    # define costs
    Q = 1e-3 * np.eye(3)
    R = np.eye(3)

    # initial controller
    _, K_init = utils.dlqr(A_star, B_star, 1e-3*np.eye(3), np.eye(3))

    rng = np.random

    env = NominalStrategy(Q=Q,
                          R=R,
                          A_star=A_star,
                          B_star=B_star,
                          sigma_w=1,
                          sigma_explore=0.1,
                          reg=1e-5,
                          epoch_multiplier=10, 
                          rls_lam=None)

    env.reset(rng)
    env.prime(100, K_init, 0.1, rng)
    for idx in range(500):
        env.step(rng)

if __name__ == '__main__':
    logging.basicConfig(level=logging.DEBUG)
    np.set_printoptions(linewidth=200)
    _main()
