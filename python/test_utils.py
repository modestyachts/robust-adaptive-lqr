import numpy as np
import cvxpy as cvx

import utils
from utils import project_ball, project_weighted_ball, psd_sqrt


def test_project_weighted_ball():

    A = np.random.random(size=(2, 2))
    B = np.random.random(size=(2, 2))
    n = p = 2
    M = np.hstack((A, B))
    theta = cvx.Variable(n, n + p, name="theta")

    Q = np.random.random(size=(2*(n+p), n+p))
    cov = Q.T.dot(Q)
    sqrt_cov = psd_sqrt(cov)
    eps = 0.1

    constr = [cvx.norm(sqrt_cov * theta.T, "fro") <= np.sqrt(eps)]
    prob = cvx.Problem(cvx.Minimize(cvx.sum_squares(M - theta)), constr)
    prob.solve(solver=cvx.MOSEK, verbose=True)
    assert prob.status == cvx.OPTIMAL

    theta_star = project_weighted_ball(M, np.zeros_like(M), cov, eps)

    assert np.allclose(np.array(theta.value), theta_star, atol=1e-4)


def test_project_ball():

    M = np.random.random((3, 2))
    Ahat = np.random.random((3, 2))
    eps_A = 0.01

    # solve min ||M - A||_F : ||A - Ahat||_op <= eps_A

    A = cvx.Variable(3, 2, name="A")
    constr = [cvx.norm(A - Ahat, 2) <= eps_A]
    prob = cvx.Problem(cvx.Minimize(cvx.norm(M - A, "fro")), constr)
    prob.solve(solver=cvx.MOSEK)
    assert prob.status == cvx.OPTIMAL
    soln = np.array(A.value)
    assert np.allclose(soln, project_ball(M, Ahat, eps_A)[0])


def test_rls():

    rng = np.random.RandomState(657423)

    n, p = 3, 2

    A = rng.normal(size=(n, n))
    B = rng.normal(size=(n, p))
    _, K = utils.dlqr(A, B)
    assert utils.spectral_radius(A + B.dot(K)) <= 1

    lam = 1e-5

    rls = utils.RecursiveLeastSquaresEstimator(n, p, lam)

    states = []
    inputs = []
    transitions = []
    xcur = np.zeros((n,))
    for _ in range(100):
        ucur = K.dot(xcur) + rng.normal(size=(p,))
        xnext = A.dot(xcur) + B.dot(ucur) + rng.normal(size=(n,))
        states.append(xcur)
        inputs.append(ucur)
        transitions.append(xnext)
        rls.update(xcur, ucur, xnext)
        xcur = xnext

    # LS estimate
    Ahat_ls, Bhat_ls, Cov_ls = utils.solve_least_squares(
            np.array(states), np.array(inputs), np.array(transitions), reg=lam)

    # RLS estimate
    Ahat_rls, Bhat_rls, Cov_rls = rls.get_estimate()

    assert np.allclose(Ahat_ls, Ahat_rls)
    assert np.allclose(Bhat_ls, Bhat_rls)
    assert np.allclose(Cov_ls, Cov_rls)

    for _ in range(100):
        ucur = K.dot(xcur) + rng.normal(size=(p,))
        xnext = A.dot(xcur) + B.dot(ucur) + rng.normal(size=(n,))
        states.append(xcur)
        inputs.append(ucur)
        transitions.append(xnext)
        rls.update(xcur, ucur, xnext)
        xcur = xnext

    # LS estimate
    Ahat_ls, Bhat_ls, Cov_ls = utils.solve_least_squares(
            np.array(states), np.array(inputs), np.array(transitions), reg=lam)

    # RLS estimate
    Ahat_rls, Bhat_rls, Cov_rls = rls.get_estimate()

    assert np.allclose(Ahat_ls, Ahat_rls)
    assert np.allclose(Bhat_ls, Bhat_rls)
    assert np.allclose(Cov_ls, Cov_rls)
