"""sls.py

An implementation of the robust adaptive controller.
Both FIR SLS version with CVXPY and the common 
Lyapunov relaxation.


"""

import numpy as np
import cvxpy as cvx
import utils
import logging
import math
import scipy.linalg

from abc import ABC, abstractmethod
from adaptive import AdaptiveMethod


class SLSInfeasibleException(Exception):
    def __init__(self, msg=None):
        super().__init__(msg)

def make_state_space_controller(Phi_x, Phi_u, n, p):
    """
    Converts FIR transfer functions to a state
    space realization of the dynamic controller,
    mapping states to inputs.

    """
    assert len(Phi_x.shape) == 2
    assert len(Phi_u.shape) == 2

    assert Phi_x.shape[1] == n
    assert Phi_u.shape[1] == n

    nT, _ = Phi_x.shape
    pT, _ = Phi_u.shape

    assert (nT % n) == 0
    assert (pT % p) == 0

    T = nT // n
    assert T == (pT // p)

    # See Theorem 2 of:
    # https://nikolaimatni.github.io/papers/sls_state_space.pdf

    Z = np.diag(np.ones(n*(T-2)), k=-n)
    assert Z.shape == ((T-1)*n, (T-1)*n)

    calI = np.zeros((n*(T-1), n))
    calI[:n, :] = np.eye(n)

    Rhat = np.hstack([Phi_x[n*k:n*(k+1), :] for k in range(1, T)])
    Mhat = np.hstack([Phi_u[p*k:p*(k+1), :] for k in range(1, T)])

    M1 = Phi_u[:p, :]
    R1 = Phi_x[:n, :]

    A = Z - calI.dot(Rhat)
    B = -calI
    C = M1.dot(Rhat) - Mhat
    D = M1

    return (A, B, C, D)


def h2_squared_norm(A, B, Phi_x, Phi_u, Q, R, sigma_w):
    """
    Gets the squared infinite horizon LQR cost for system
    (A,B) in feedback with the controller defined by Phi_x
    and Phi_u. 

    """

    n, p = B.shape

    A_k, B_k, C_k, D_k = make_state_space_controller(Phi_x, Phi_u, n, p)

    A_cl = np.block([
        [A + B.dot(D_k), B.dot(C_k)],
        [B_k, A_k]
    ])

    Q_sqrt = utils.psd_sqrt(Q)
    R_sqrt = utils.psd_sqrt(R)

    C_cl = np.block([
        [Q_sqrt, np.zeros((n, A_k.shape[0]))],
        [R_sqrt.dot(D_k), R_sqrt.dot(C_k)]
    ])

    B_cl = np.vstack((np.eye(n), np.zeros((A_k.shape[0], n))))

    P = utils.solve_discrete_lyapunov(A_cl.T, B_cl.dot(B_cl.T))

    return (sigma_w ** 2) * np.trace(C_cl.dot(P).dot(C_cl.T))


def _assert_AB_consistent(A, B):
    assert len(A.shape) == 2 and A.shape[0] == A.shape[1]
    assert len(B.shape) == 2
    assert A.shape[0] == B.shape[0]


def _assert_ABCD_consistent(A, B, C, D):
    _assert_AB_consistent(A, B)

    assert len(C.shape) == 2
    assert len(D.shape) == 2

    assert C.shape[1] == A.shape[0]
    assert C.shape[0] == D.shape[0]
    assert D.shape[1] == B.shape[1]


def roll_forward(A, B, K, x0, psi0, sigma_w, horizon, rng=None):
    """Apply an LTI controller K = (A_k,B_k,C_k,D_k)

    Roll the true system (A, B) forward with the SS realization of the LTI
    controller given. horizon is the length of the trajectory, and
    sigma_w is the stddev of the Gaussian process noise.

    """

    if rng is None:
        rng = np.random

    _assert_AB_consistent(A, B)

    A_k, B_k, C_k, D_k = K
    _assert_ABCD_consistent(A_k, B_k, C_k, D_k)

    state_dim, input_dim = B.shape
    psi_dim = A_k.shape[0]

    assert C_k.shape[0] == input_dim
    assert B_k.shape[1] == state_dim

    if x0 is None:
        x0 = np.zeros((state_dim,))
    if psi0 is None:
        psi0 = np.zeros((psi_dim,))

    assert x0.shape == (state_dim,)
    assert psi0.shape == (psi_dim,)

    process = sigma_w*rng.normal(size=(horizon, state_dim))
    xt = np.array(x0)
    psit = np.array(psi0)

    states = np.zeros((horizon+1, state_dim))
    inputs = np.zeros((horizon, input_dim))
    controller_states = np.zeros((horizon+1, psi_dim))

    states[0, :] = x0
    controller_states[0, :] = psi0

    for t in range(horizon):
        psitp1 = A_k.dot(psit) + B_k.dot(xt)
        ut = C_k.dot(psit) + D_k.dot(xt)
        xtp1 = A.dot(xt) + B.dot(ut) + process[t]
        inputs[t, :] = ut
        states[t+1, :] = xtp1
        controller_states[t+1, :] = psitp1
        xt = xtp1
        psit = psitp1

    return states, inputs, controller_states


def sls_synth(Q, R, Ahat, Bhat, eps_A, eps_B, T, gamma, alpha, logger=None):
    """
    Solves the SLS synthesis problem for length T FIR filters
    using CVXPY

    """

    assert len(Q.shape) == 2 and Q.shape[0] == Q.shape[1]
    assert len(R.shape) == 2 and R.shape[0] == R.shape[1]
    assert len(Ahat.shape) == 2 and Ahat.shape[0] == Ahat.shape[1]
    assert len(Bhat.shape) == 2 and Bhat.shape[0] == Ahat.shape[0]
    assert Q.shape[0] == Ahat.shape[0]
    assert R.shape[0] == Bhat.shape[1]
    assert eps_A >= 0
    assert eps_B >= 0
    assert T >= 1
    assert gamma > 0 and gamma < 1
    assert alpha > 0 and alpha < 1

    if logger is None:
        logger = logging.getLogger(__name__)

    n, p = Bhat.shape

    Q_sqrt = utils.psd_sqrt(Q)
    R_sqrt = utils.psd_sqrt(R)

    # Phi_x = \sum_{k=1}^{T} Phi_x[k] z^{-k}
    Phi_x = cvx.Variable(T*n, n, name="Phi_x")

    # Phi_u = \sum_{k=1}^{T} Phi_u[k] z^{-k}
    Phi_u = cvx.Variable(T*p, n, name="Phi_u")

    # htwo_cost
    htwo_cost = cvx.Variable(name="htwo_cost")

    # subspace constraint:
    # [zI - Ah, -Bh] * [Phi_x; Phi_u] = I
    #
    # Note that:
    # z Phi_x = \sum_{k=0}^{T-1} Phi_x[k+1] z^{-k}
    #
    # This means that:
    # 1) Phi_x[1] = I
    # 2) Phi_x[k+1] = Ah*Phi_x[k] + Bh*Phi_u[k] for k=1, ..., T-1
    # 3) Ah*Phi_x[T] + Bh*Phi_u[T] = 0

    constr = []

    constr.append(Phi_x[:n, :] == np.eye(n))
    for k in range(T-1):
        constr.append(Phi_x[n*(k+1):n*(k+1+1), :] == Ahat*Phi_x[n*k:n*(k+1), :] + Bhat*Phi_u[p*k:p*(k+1), :])
    constr.append(Ahat*Phi_x[n*(T-1):, :] + Bhat*Phi_u[p*(T-1):, :] == 0)

    # H2 constraint:
    # By Parseval's identity, this is equal (up to constants) to
    #
    # frobenius_norm(
    #   [ Q_sqrt*Phi_x[1] ;
    #     ...
    #     Q_sqrt*Phi_x[T] ;
    #     R_sqrt*Phi_u[1] ;
    #     ...
    #     R_sqrt*Phi_u[T]
    #   ]
    # ) <= htwo_cost
    # TODO: what is the best way to implement this in cvxpy?
    constr.append(
        cvx.norm(
            cvx.bmat(
                [[Q_sqrt*Phi_x[n*k:n*(k+1), :]] for k in range(T)] +
                [[R_sqrt*Phi_u[p*k:p*(k+1), :]] for k in range(T)]),
            'fro') <= htwo_cost)

    # H-infinity constraint
    #
    # We want to enforce ||H(z)||_inf <= gamma, where
    #
    #   H(z) = \sum_{k=1}^{T} [ mult_x * Phi_x[k] ; mult_u * Phi_u[k] ] z^{-k}.
    #
    # Here, each of the FIR coefficients has size (n+p) x n. Since n+p>n, we enforce
    # the constraint on the transpose system H^T(z). The LMI constraint
    # for this comes from Theorem 5.8 of
    # Positive trigonometric polynomials and signal processing applications (2007) by
    # B. Dumitrescu.
    #
    # Here is a table to map the variable names in the text to this program
    #
    #       Text          Program                   Comment
    # -------------------------------------------------------------
    #         p             n                   Output dim
    #         m             n+p                 Input dim
    #         n             T                   FIR horizon
    #       p(n+1)          n(T+1)              SDP variable size
    #      p(n+1) x m       n(T+1) x (n+p)

    mult_x = eps_A/np.sqrt(alpha)
    mult_u = eps_B/np.sqrt(1-alpha)

    # Hbar has size (T+1)*n x (n+p)
    Hbar = cvx.bmat(
        [[np.zeros((n, n)), np.zeros((n, p))]] +
        [[mult_x*Phi_x[n*k:n*(k+1), :].T, mult_u*Phi_u[p*k:p*(k+1), :].T] for k in range(T)])

    Q = cvx.Semidef(n*(T+1), name="Q")

    # Constraint (5.44)

    # Case k==0: the block diag of Q has to sum to gamma^2 * eye(n)
    gamma_sq = gamma ** 2
    constr.append(
        sum([Q[n*t:n*(t+1), n*t:n*(t+1)] for t in range(T+1)]) == gamma_sq*np.eye(n))

    # Case k>0: the block off-diag of Q has to sum to zero
    for k in range(1, T+1):
        constr.append(
            sum([Q[n*t:n*(t+1), n*(t+k):n*(t+1+k)] for t in range(T+1-k)]) == np.zeros((n, n)))

    # Constraint (5.45)
    constr.append(
        cvx.bmat([
            [Q, Hbar],
            [Hbar.T, np.eye(n+p)]]) == cvx.Semidef(n*(T+1) + (n+p)))

    prob = cvx.Problem(cvx.Minimize(htwo_cost), constr)
    prob.solve(solver=cvx.SCS)

    if prob.status == cvx.OPTIMAL:
        logging.debug("successfully solved!")
        Phi_x = np.array(Phi_x.value)
        Phi_u = np.array(Phi_u.value)
        return (True, prob.value, Phi_x, Phi_u)
    else:
        logging.debug("could not solve: {}".format(prob.status))
        return (False, None, None, None)


def sls_common_lyapunov(A, B, Q, R, eps_A, eps_B, tau, logger=None):
    """
    Solves the common Lyapunov relaxation to the robust 
    synthesis problem.

    Taken from
    lstd-lqr/blob/master/code/policy_iteration.ipynb
    learning-lqr/experiments/matlab/sls_synth_yalmip/common_lyap_synth_var2_alpha.m

    """

    if logger is None:
        logger = logging.getLogger(__name__)

    d, p = B.shape
    X = cvx.Symmetric(d)   # inverse Lyapunov function
    Z = cvx.Variable(p, d) # -K*X
    W_11 = cvx.Symmetric(d)
    W_12 = cvx.Variable(d, p)
    W_22 = cvx.Symmetric(p)
    alph = cvx.Variable()  # scalar for tuning the H_inf constraint

    constraints = []

    # H2 cost: trace(W)=H2 cost
    mat1 = cvx.bmat([
            [X, X, Z.T],
            [X, W_11, W_12],
            [Z, W_12.T, W_22]])
    constraints.append(mat1 == cvx.Semidef(2*d + p))

    # H_infinity constraint
    mat2 = cvx.bmat([
            [X-np.eye(d), (A*X+B*Z), np.zeros((d, d)), np.zeros((d, p))],
            [(X*A.T+Z.T*B.T), X, eps_A*X, eps_B*Z.T],
            [np.zeros((d, d)), eps_A*X, alph*(tau**2)*np.eye(d), np.zeros((d, p))],
            [np.zeros((p, d)), eps_B*Z, np.zeros((p, d)), (1-alph)*(tau**2)*np.eye(p)]])
    constraints.append(mat2 == cvx.Semidef(3*d + p))

    # constrain alpha to be in [0,1]:
    constraints.append(alph >= 0)
    constraints.append(alph <= 1)

    # Solve!
    objective = cvx.Minimize(cvx.trace(Q*W_11) + cvx.trace(R*W_22))
    prob = cvx.Problem(objective, constraints)
    try:
        obj = prob.solve(solver=cvx.MOSEK)
    except cvx.SolverError:
        logger.warn("SolverError encountered")
        return (False, None, None, None)

    if prob.status == cvx.OPTIMAL:
        logging.debug("common_lyapunov: found optimal solution")

        X_value = np.array(X.value)
        P_value = scipy.linalg.solve(X_value, np.eye(d), sym_pos=True)

        # NOTE: the K returned here is meant to be used
        # as A + BK **NOT** A - BK
        K_value = np.array(Z.value).dot(P_value)

        return (True, obj, P_value, K_value)

    else:
        logging.debug("common_lyapunov: could not solve (status={})".format(prob.status))

        return (False, None, None, None)

class SLS_Implementation(ABC):

    @abstractmethod
    def open(self):
        """

        """

        pass

    @abstractmethod
    def synth(self, Q, R, Ahat, Bhat, eps_A, eps_B, truncation_length, gamma, alpha, logger):
        """

        """

        pass

class SLS_CVXPY(SLS_Implementation):

    def open(self):
        pass

    def synth(self, Q, R, Ahat, Bhat, eps_A, eps_B, truncation_length, gamma, alpha, logger):
        return sls_synth(Q, R, Ahat, Bhat, eps_A, eps_B, truncation_length, gamma, alpha, logger)

class SLS_FIRStrategy(AdaptiveMethod):
    """Adaptive control based on FIR truncated SLS

    """

    def __init__(self, Q, R, A_star, B_star, sigma_w, rls_lam,
                 sigma_explore, reg, epoch_multiplier,
                 truncation_length, actual_error_multiplier,
                 use_gamma=0.98, sls_impl=None):
        super().__init__(Q, R, A_star, B_star, sigma_w, rls_lam)
        self._sigma_explore = sigma_explore
        self._reg = reg
        self._epoch_multiplier = epoch_multiplier
        # TODO(stephentu):
        # the truncation length should grow with time, but for now
        # we keep it constant
        # Additionally, gamma should be searched over as an optimization
        # variable. For how, we fix the value.
        # Finally, the optimization problem should be modified
        # to involve the variable V as in https://arxiv.org/abs/1805.09388
        self._truncation_length = truncation_length
        self._actual_error_multiplier = actual_error_multiplier
        self._sls_impl = sls_impl if sls_impl is not None else SLS_CVXPY()
        self._logger = logging.getLogger(__name__)
        self._use_gamma = use_gamma
        self._controller_state = None

    def _get_logger(self):
        return self._logger

    def reset(self, rng):
        super().reset(rng)
        self._sls_impl.open()
        self._midway_infeasible = 0

    def _design_controller(self, states, inputs, transitions, rng):

        logger = self._get_logger()

        Anom, Bnom, _ = utils.solve_least_squares(states, inputs, transitions, reg=self._reg)
        eps_A = np.linalg.norm(Anom - self._A_star, ord=2)
        eps_B = np.linalg.norm(Bnom - self._B_star, ord=2)

        effective_eps_A = self._actual_error_multiplier * eps_A
        effective_eps_B = self._actual_error_multiplier * eps_B

        epoch_id = self._epoch_idx + 1 if self._has_primed else 0

        logger.info("_design_controller(epoch={}): effective_eps_A={}, effective_eps_B={}".format(epoch_id, effective_eps_A, effective_eps_B))

        # if SLS is not feasible, we fallback to the current
        # control policy if it exists, otherwise we throw an SLSInfeasibleException
        if self._use_gamma is None:
            # bisect for gamma
            logger.info("_design_controller(epoch={}): bisecting for gamma".format(epoch_id))

            INF = 1e12

            def fn(gamma):
                is_feasible, obj, _, _ = self._sls_impl.synth(self._Q, self._R, Anom, Bnom,
                    effective_eps_A, effective_eps_B, self._truncation_length,
                    gamma=gamma, alpha=0.5, logger=logger)
                if not is_feasible:
                    return INF
                else:
                    return 1/(1-gamma) * obj

            disp_lvl = 3 if logger.isEnabledFor(logging.DEBUG) else 0
            gamma_star, _, error_flag, _ = scipy.optimize.fminbound(fn, 0, 1 - 1e-5, xtol=1e-2, maxfun=20, full_output=True, disp=disp_lvl)
            if error_flag:
                logger.warn("_design_controller(epoch={}): maxfun exceeded during bisection, gamma_star={}".format(epoch_id, gamma_star))
            logger.info("_design_controller(epoch={}): using gamma_star={}".format(epoch_id, gamma_star))
            is_feasible, _, Phi_x, Phi_u = self._sls_impl.synth(self._Q, self._R, Anom, Bnom,
                    effective_eps_A, effective_eps_B, self._truncation_length,
                    gamma=gamma_star, alpha=0.5, logger=logger)

        else:
            assert self._use_gamma > 0 and self._use_gamma < 1
            logger.info("_design_controller(epoch={}): using fixed gamma={}".format(epoch_id, self._use_gamma))
            is_feasible, _, Phi_x, Phi_u = self._sls_impl.synth(self._Q, self._R, Anom, Bnom,
                    effective_eps_A, effective_eps_B, self._truncation_length,
                    gamma=self._use_gamma, alpha=0.5, logger=logger)

        if not is_feasible:
            logger.info("_design_controller(epoch={}): SLS was not feasible...".format(epoch_id))

            try:
                self._current_K
                # keep current controller
                assert self._current_K is not None
                logger.warn("_design_controller(epoch={}): SLS not feasible: keeping current controller".format(epoch_id))
                self._midway_infeasible += 1
            except AttributeError:
                logger.warn("_design_controller(epoch={}): SLS not feasible: no existing controller to fallback on, effective_eps_A={}, effective_eps_B={}".format(epoch_id, effective_eps_A, effective_eps_B))
                raise SLSInfeasibleException()

        else:
            logger.info("_design_controller(epoch={}): SLS was feasible. updating controller".format(epoch_id))
            self._Phi_x = Phi_x
            self._Phi_u = Phi_u
            self._current_K = make_state_space_controller(Phi_x, Phi_u, self._n, self._p)

        # compute the infinite horizon cost of this controller
        Jnom = h2_squared_norm(self._A_star,
                               self._B_star,
                               self._Phi_x,
                               self._Phi_u,
                               self._Q,
                               self._R,
                               self._sigma_w)

        return Anom, Bnom, Jnom

    def _should_terminate_epoch(self):

        if (self._iteration_within_epoch_idx >=
                self._epoch_multiplier * (self._epoch_idx + 1)):
            logger = self._get_logger()
            logger.debug("terminating epoch... exploration noise will now have stddev {}".format(
                self._sigma_explore * 1/math.pow(self._epoch_idx + 2, 1/3)))
            return True
        else:
            return False

    def _get_input(self, state, rng):

        rng = self._get_rng(rng)

        A_k, B_k, C_k, D_k = self._current_K
        psit = self._controller_state
        if psit is None:
            psit = np.zeros((A_k.shape[0],))
        psitp1 = A_k.dot(psit) + B_k.dot(state)
        ctrl_input = C_k.dot(psit) + D_k.dot(state)
        self._controller_state = psitp1

        sigma_explore_decay = 1/math.pow(self._epoch_idx + 1, 1/3)
        explore_input = self._sigma_explore * sigma_explore_decay * rng.normal(size=(self._p,))
        return ctrl_input + explore_input


class SLS_CommonLyapunovStrategy(AdaptiveMethod):
    """
    Adaptive control based on common Lyapunov relaxation
    of robust control problem

    """

    def __init__(self, Q, R, A_star, B_star, sigma_w, rls_lam,
                 sigma_explore, reg, epoch_multiplier, actual_error_multiplier):
        super().__init__(Q, R, A_star, B_star, sigma_w, rls_lam)
        self._sigma_explore = sigma_explore
        self._reg = reg
        self._epoch_multiplier = epoch_multiplier
        self._actual_error_multiplier = actual_error_multiplier
        self._logger = logging.getLogger(__name__)
        self._midway_infeasible = 0

    def reset(self, rng):
        super().reset(rng)
        self._midway_infeasible = 0

    def _get_logger(self):
        return self._logger

    def _design_controller(self, states, inputs, transitions, rng):
        logger = self._get_logger()

        Anom, Bnom, _ = utils.solve_least_squares(states, inputs, transitions, reg=self._reg)
        eps_A = np.linalg.norm(Anom - self._A_star, ord=2)
        eps_B = np.linalg.norm(Bnom - self._B_star, ord=2)

        effective_eps_A = self._actual_error_multiplier * eps_A
        effective_eps_B = self._actual_error_multiplier * eps_B

        epoch_id = self._epoch_idx + 1 if self._has_primed else 0

        logger.info("_design_controller(epoch={}): effective_eps_A={}, effective_eps_B={}".format(epoch_id, effective_eps_A, effective_eps_B))

        is_feasible, _, _, K = sls_common_lyapunov(
                Anom, Bnom, self._Q, self._R,
                effective_eps_A, effective_eps_B, tau=0.999, logger=logger)

        if not is_feasible:

            try:
                self._current_K
                # keep current controller
                assert self._current_K is not None
                logger.warn("_design_controller(epoch={}): SLS not feasible: keeping current controller".format(epoch_id))
                self._midway_infeasible += 1
            except AttributeError:
                logger.warn("_design_controller(epoch={}): SLS not feasible: no existing controller to fallback on, effective_eps_A={}, effective_eps_B={}".format(epoch_id, effective_eps_A, effective_eps_B))
                raise SLSInfeasibleException()

        else:
            logger.info("_design_controller(epoch={}): SLS was feasible. updating controller".format(epoch_id))
            self._current_K = K

        # compute the infinite horizon cost of this controller
        Jnom = utils.LQR_cost(self._A_star, self._B_star, self._current_K, self._Q, self._R, self._sigma_w)

        return Anom, Bnom, Jnom

    def _should_terminate_epoch(self):

        if (self._iteration_within_epoch_idx >=
                self._epoch_multiplier * (self._epoch_idx + 1)):
            logger = self._get_logger()
            logger.debug("terminating epoch... exploration noise will now have stddev {}".format(
                self._sigma_explore * 1/math.pow(self._epoch_idx + 2, 1/3)))
            return True
        else:
            return False

    def _get_input(self, state, rng):

        rng = self._get_rng(rng)
        ctrl_input = self._current_K.dot(state)
        sigma_explore_decay = 1/math.pow(self._epoch_idx + 1, 1/3)
        explore_input = self._sigma_explore * sigma_explore_decay * rng.normal(size=(self._p,))
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

    env = SLS_FIRStrategy(Q=Q,
                          R=R,
                          A_star=A_star,
                          B_star=B_star,
                          sigma_w=1,
                          sigma_explore=0.1,
                          reg=1e-5,
                          epoch_multiplier=10,
                          truncation_length=12,
                          actual_error_multiplier=1, 
                          rls_lam=None)

    env.reset(rng)
    env.prime(250, K_init, 0.5, rng)
    for idx in range(500):
        env.step(rng)

    env = SLS_CommonLyapunovStrategy(Q=Q,
                                     R=R,
                                     A_star=A_star,
                                     B_star=B_star,
                                     sigma_w=1,
                                     sigma_explore=0.1,
                                     reg=1e-5,
                                     epoch_multiplier=10,
                                     actual_error_multiplier=1, 
                                     rls_lam=None)

    env.reset(rng)
    env.prime(250, K_init, 0.5, rng)
    for idx in range(500):
        env.step(rng)

if __name__ == '__main__':
    logging.basicConfig(level=logging.DEBUG)
    np.set_printoptions(linewidth=200)
    _main()

