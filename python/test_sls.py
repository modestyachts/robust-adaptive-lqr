import numpy as np
import utils


from sls import sls_synth, make_state_space_controller, sls_common_lyapunov, h2_squared_norm


def test_sls_synth():

    rng = np.random.RandomState(893754)

    Ahat = np.array([
        [1.01, 0.01, 0],
        [-0.01, 1.01, 0.01],
        [0, -0.01, 1.01]
    ])
    Bhat = np.eye(3)

    eps_A = 0.0001
    eps_B = 0.0001
    Ahat = utils.sample_2_to_2_ball(Ahat, eps_A, rng)
    Bhat = utils.sample_2_to_2_ball(Bhat, eps_B, rng)

    Q = np.eye(3)
    R = np.eye(3)

    alpha = 0.5
    gamma = 0.98

    n = 3
    p = 3

    T = 15

    is_feasible, sqrt_htwo_cost, Phi_x, Phi_u = sls_synth(Q, R, Ahat, Bhat, eps_A, eps_B, T, gamma, alpha)

    assert is_feasible, "should be feasible"

    P_nom, K_nom = utils.dlqr(Ahat, Bhat, Q, R)

    assert np.allclose(sqrt_htwo_cost ** 2, np.trace(P_nom))

    L = Ahat + Bhat.dot(K_nom)
    cur = np.eye(L.shape[0])
    coeffs = [np.array(cur)]
    for _ in range(T):
        cur = L.dot(cur)
        coeffs.append(np.array(cur))

    for idx in range(T):
        expected = coeffs[idx]
        actual = Phi_x[idx*n:(idx+1)*n, :]
        assert np.allclose(expected, actual, atol=1e-5)

    A_k, B_k, C_k, D_k = make_state_space_controller(Phi_x, Phi_u, n, p)

    A_cl = np.block([
        [Ahat + Bhat.dot(D_k), Bhat.dot(C_k)],
        [B_k, A_k]
    ])
    cur = np.eye(A_cl.shape[0])
    cl_coeffs = [np.eye(n)]
    for _ in range(T):
        cur = A_cl.dot(cur)
        cl_coeffs.append(np.array(cur[:n, :n]))

    for idx in range(T):
        expected = coeffs[idx]
        actual = cl_coeffs[idx]
        assert np.allclose(expected, actual, atol=1e-5)


def test_sls_common_lyapunov():

    rng = np.random.RandomState(237853)

    Ahat = np.array([
        [1.01, 0.01, 0],
        [-0.01, 1.01, 0.01],
        [0, -0.01, 1.01]
    ])
    Bhat = np.eye(3)

    eps_A = 0.0001
    eps_B = 0.0001
    Ahat = utils.sample_2_to_2_ball(Ahat, eps_A, rng)
    Bhat = utils.sample_2_to_2_ball(Bhat, eps_B, rng)

    Q = np.eye(3)
    R = np.eye(3)

    n = 3
    p = 3

    is_feasible, _, P, K = sls_common_lyapunov(Ahat, Bhat, Q, R, eps_A, eps_B, 0.999, None)

    assert is_feasible

    P_nom, K_nom = utils.dlqr(Ahat, Bhat, Q, R)

    # THIS FAILS
    #assert np.allclose(np.trace(P), np.trace(P_nom))

    assert np.allclose(K, K_nom, atol=1e-6)


def test_sls_h2_cost():

    rng = np.random.RandomState(805238)

    Astar = np.array([
        [1.01, 0.01, 0],
        [-0.01, 1.01, 0.01],
        [0, -0.01, 1.01]
    ])
    Bstar = np.eye(3)

    eps_A = 0.00001
    eps_B = 0.00001
    Ahat = utils.sample_2_to_2_ball(Astar, eps_A, rng)
    Bhat = utils.sample_2_to_2_ball(Bstar, eps_B, rng)

    Q = np.eye(3)
    R = np.eye(3)

    n = 3
    p = 3

    T = 15

    is_feasible, _, _, K_cl = sls_common_lyapunov(Ahat, Bhat, Q, R, eps_A, eps_B, 0.999, None)

    assert is_feasible

    P_star, K_star = utils.dlqr(Astar, Bstar, Q, R)
    J_star = np.trace(P_star)

    assert np.allclose(J_star, utils.LQR_cost(Astar, Bstar, K_star, Q, R, 1))
    assert np.allclose(J_star, utils.LQR_cost(Astar, Bstar, K_cl, Q, R, 1), atol=1e-6)

    is_feasible, _, Phi_x, Phi_u = sls_synth(Q, R, Ahat, Bhat, eps_A, eps_B, T, 0.999, 0.5)

    assert np.allclose(J_star, h2_squared_norm(Astar, Bstar, Phi_x, Phi_u, Q, R, 1), atol=1e-6)
