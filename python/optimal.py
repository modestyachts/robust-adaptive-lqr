"""optimal.py

"""

import numpy as np
import utils
import logging

from adaptive import AdaptiveMethod


class OptimalStrategy(AdaptiveMethod):

    def __init__(self,
                 Q,
                 R,
                 A_star,
                 B_star,
                 sigma_w):
        super().__init__(Q, R, A_star, B_star, sigma_w, None)
        self._logger = logging.getLogger(__name__)

    def _get_logger(self):
        return self._logger

    def _design_controller(self, states, inputs, transitions, rng):
        P, self._optimal_K = utils.dlqr(self._A_star, self._B_star, self._Q, self._R)
        return (self._A_star, self._B_star, (self._sigma_w**2) * np.trace(P))

    def _should_terminate_epoch(self):
        return False

    def _get_input(self, state, rng):
        return self._optimal_K.dot(state)
