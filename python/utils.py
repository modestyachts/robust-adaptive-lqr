"""utils.py


"""


import numpy as np
import scipy.linalg
import scipy.optimize


class RecursiveLeastSquaresEstimator(object):

    def __init__(self, n, p, lam):
        assert lam > 0
        self._n = n
        self._p = p
        self._lam = lam
        self._Mcur = self._lam * np.eye(self._n + self._p)
        self._Mcur_inv = 1/self._lam * np.eye(self._n + self._p)
        self._Qcur = np.zeros((self._n + self._p, self._n))
        self._estimate_A = np.zeros((self._n, self._n))
        self._estimate_B = np.zeros((self._n, self._p))

    def update(self, state, inp, transition):
        zn = np.hstack((state, inp))
        Mcur_inv_dot_zn = self._Mcur_inv.dot(zn)
        Mnext_inv = self._Mcur_inv - np.outer(Mcur_inv_dot_zn, Mcur_inv_dot_zn)/(1 + zn.dot(Mcur_inv_dot_zn))
        self._Mcur_inv = Mnext_inv
        self._Mcur += np.outer(zn, zn)
        self._Qcur += np.outer(zn, transition)

        estimate_transposed = self._Mcur_inv.dot(self._Qcur)
        estimate = estimate_transposed.T
        self._estimate_A = estimate[:, :self._n]
        self._estimate_B = estimate[:, self._n:]

    def get_estimate(self):
        return (self._estimate_A, self._estimate_B, self._Mcur)


def spectral_radius(A):
    assert len(A.shape) == 2 and A.shape[0] == A.shape[1]
    return max(np.abs(np.linalg.eigvals(A)))


def psd_sqrt(P):
    assert len(P.shape) == 2
    assert P.shape[0] == P.shape[1]
    w, v = np.linalg.eigh(P)
    assert (w >= 0).all()
    return v.dot(np.diag(np.sqrt(w))).dot(v.T)


def pd_inv_sqrt(P):
    assert len(P.shape) == 2
    assert P.shape[0] == P.shape[1]
    w, v = np.linalg.eigh(P)
    TOL = 1e-8
    assert (w >= TOL).all()
    return v.dot(np.diag(1/np.sqrt(w))).dot(v.T)


def dlqr(A,B,Q=None,R=None):
    """Solve the discrete time lqr controller.

    x[k+1] = A x[k] + B u[k]
    cost = sum x[k].T*Q*x[k] + u[k].T*R*u[k]
    """
    #ref Bertsekas, p.151
    if Q is None:
        Q = np.eye(A.shape[0])
    if R is None:
        R = np.eye(B.shape[1])

    P = scipy.linalg.solve_discrete_are(A, B, Q, R)
    K = -scipy.linalg.solve(B.T.dot(P).dot(B) + R, B.T.dot(P).dot(A), sym_pos=True)

    A_c = A + B.dot(K)
    TOL = 1e-5
    if spectral_radius(A_c) >= 1 + TOL:
        print("WARNING: spectral radius of closed loop is:", spectral_radius(A_c))

    return P, K


def LQR_cost(A, B, K, Q, R, sigma_w):
    """Compute infinite time horizon average LQR cost.

    Returns +inf if A+BK is not stable


    """

    L = A + B.dot(K)
    if spectral_radius(L) >= 1:
        return np.inf

    M = Q + K.T.dot(R).dot(K)

    P = solve_discrete_lyapunov(L, M)

    return (sigma_w ** 2) * np.trace(P)


def sample_2_to_2_ball(A, eps, rng=None):
    """Sample from the 2->2 ball around A with radius eps

    """

    if rng is None:
        rng = np.random

    Asample = np.array(A)
    Delta_A = rng.normal(size=Asample.shape)
    Delta_A = Delta_A / np.linalg.norm(Delta_A, ord=2) * eps
    Asample += Delta_A
    TOL = 1e-7
    assert (np.linalg.norm(Asample - A, ord=2) <= eps + TOL), str(np.linalg.norm(Asample - A, ord=2))
    return Asample


def solve_discrete_lyapunov(A, Q, method=None):
    """Solve A^T P A - P + Q = 0

    """

    # newer versions of scipy solve A P A^T - P + Q = 0,
    # while older ones solve A^T P A - P + Q = 0. I do not
    # remember exactly which version of scipy made the change.

    # I am going to assume you have the newer version installed.
    # If the assertion below fails, please add an if statement
    # that branches on your version.

    P = scipy.linalg.solve_discrete_lyapunov(A.T, Q, method)

    assert np.allclose(A.T.dot(P).dot(A) - P, -Q)

    return P


def solve_least_squares(states, inputs, transitions, reg=0):
    """Solve for system dynamics from states and inputs

    """

    assert len(states.shape) == 2
    assert len(inputs.shape) == 2
    assert len(transitions.shape) == 2
    assert states.shape[0] == inputs.shape[0]
    assert states.shape == transitions.shape

    n, p = states.shape[1], inputs.shape[1]

    X = np.hstack((states, inputs))
    Y = transitions

    regI = reg * np.eye(n + p)

    Cov = X.T.dot(X) + regI

    # (n+p) x n
    Theta_hat = scipy.linalg.solve(Cov, X.T.dot(Y), sym_pos=True)
    # n x (n+p)
    Theta_hat = Theta_hat.T

    A_est = Theta_hat[:, :n]
    B_est = Theta_hat[:, n:]

    return A_est, B_est, Cov


def block_diagstack(A,B):
    # TODO use scipy?
    n,m = A.shape
    p,q = B.shape
    top = np.hstack([A, np.zeros([n,q])])
    bottom = np.hstack([np.zeros([p,m]), B])
    return np.vstack([top,bottom])

def solve_augmented_least_squares(states, inputs, transitions, A_z, B_z, reg=0):
    """Solve for system dynamics from states and inputs

    """

    assert len(states.shape) == 2
    assert len(inputs.shape) == 2
    assert len(transitions.shape) == 2
    assert states.shape[0] == inputs.shape[0]
    assert states.shape == transitions.shape

    _, p = states.shape[1], inputs.shape[1]
    n, p_z = B_z.shape
    p_d = p - p_z

    X = np.hstack((states, inputs))
    Y = transitions

    # TODO implement by indexing instead
    Ea = np.vstack([np.zeros([n,n]), np.eye(n)])
    Eb = np.vstack([np.zeros([p_z, p_d]), np.eye(p_d)])
    E2 = block_diagstack(Ea.T, Eb.T)

    Theta_0 = np.bmat([[A_z, np.eye(n), B_z, np.zeros([n, p_d])],
                       [np.zeros([n,n]), np.zeros([n,n]), 
                               np.zeros([n,p_z]), np.zeros([n,p_d])]])

    newY = Y.dot(Ea) - X.dot(Theta_0.T).dot(Ea)
    newX = X.dot(E2.T)

    regI = reg * np.eye(n + p_d)

    Cov = newX.T.dot(newX) + regI

    # (n+p) x n
    Theta_hat = scipy.linalg.solve(Cov, newX.T.dot(newY), sym_pos=True)
    # n x (n+p)
    Theta_hat = Theta_hat.T

    A_d_est = Theta_hat[:, :n]
    B_d_est = Theta_hat[:, n:]

    if p_d == 0: B_d_est = None

    return A_d_est, B_d_est, Cov


def augment_system(A_z, A_d, B_z, B_d):
    n, p_z = B_z.shape

    if B_d is not None:
        _, p_d = B_d.shape
    else:
        p_d = 0
        B_d = np.zeros([n, p_d])

    A = np.vstack([np.hstack([A_z, np.eye(n)]),
        np.hstack([np.zeros([n, n]), A_d])])

    B = np.vstack([np.hstack([B_z, np.zeros([n, p_d])]),
        np.hstack([np.zeros([n, p_z]), B_d])])

    #np.bmat([[A_z, np.eye(n)],
        #         [np.zeros([n, n]), A_d]])

    # B = np.bmat([[B_z, np.zeros([n, p_d])],
    #              [np.zeros([n, p_z]), B_d]])

    return A, B


def quad_form(Q, x):
    return x.dot(Q.dot(x))


def project_ball(M, Ahat, eps_A):
    """Project M onto the set { A : ||A - Ahat||_op <= eps_A }

    """

    # equivalent to
    # min_Delta || (M - Ahat) - Delta ||_F : ||Delta||_op <= eps_A

    assert len(M.shape) == 2
    assert len(Ahat.shape) == 2
    assert M.shape == Ahat.shape
    assert eps_A >= 0

    E = M - Ahat
    U, s, VT = np.linalg.svd(E)

    if max(s) <= eps_A:
        # already in the set, no work to do
        return (M, False)

    V = VT.T

    Delta_s = np.minimum(s, eps_A) # threshold
    Delta_lam = np.zeros_like(E)
    for i, sval in enumerate(Delta_s):
        Delta_lam[i, i] = sval
    Delta = U.dot(Delta_lam).dot(VT)

    ret = Ahat + Delta

    TOL = 1e-5
    assert np.linalg.norm(ret - Ahat, ord=2) <= eps_A + TOL

    return (ret, True)


def project_weighted_ball(M, theta_hat, cov, eps):
    """Project M onto the set { theta : Tr((theta - theta_hat) * cov * (theta - theta_hat).T) <= eps }

    We assume that cov is positive definite

    """

    assert len(M.shape) == 2
    assert M.shape == theta_hat.shape
    assert len(cov.shape) == 2
    assert cov.shape[0] == cov.shape[1]
    assert M.shape[1] == cov.shape[0]
    assert eps > 0

    TOL = 1e-5

    if not np.allclose(theta_hat, np.zeros_like(theta_hat)):
        ret = project_weighted_ball(M - theta_hat, np.zeros_like(theta_hat), cov, eps)
        ret += theta_hat
        assert np.trace((ret - theta_hat).dot(cov).dot((ret - theta_hat).T)) <= eps + TOL
        return ret

    # now we can treat theta_hat = 0

    # first check easy case:
    if np.trace(M.dot(cov).dot(M.T)) <= eps:
        return M

    # otherwise, solution takes form
    # theta_star = M (I + lam * cov)^{-1} for some lam > 0

    w, V = np.linalg.eigh(cov)

    # find lam such that
    # Tr( M * (I + lam * cov)^{-1} cov * (I + lam * cov)^{-1} M.T ) = eps

    MV = M.dot(V)
    VTMT_MV = MV.T.dot(MV)
    term2 = np.diag(VTMT_MV)

    def func(lam):
        assert lam >= 0
        term1 = (w / ((1 + lam * w) ** 2))
        return eps - np.sum(term1 * term2)

    assert func(0) <= 0
    lam_ub = 1
    while func(lam_ub) <= 0:
        lam_ub *= 2

    lam_star, results = scipy.optimize.brentq(func, 0, lam_ub, full_output=True)
    assert results.converged

    theta_star = MV.dot(np.diag(1/(1 + lam_star * w))).dot(V.T)

    assert np.trace(theta_star.dot(cov).dot(theta_star.T)) <= eps + TOL

    return theta_star

def _main():
    # for debugging
    pass


if __name__ == '__main__':
    _main()
