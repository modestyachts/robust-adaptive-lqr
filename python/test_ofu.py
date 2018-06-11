import numpy as np
import utils

from scipy.optimize import check_grad
from ofu import function_value, gradient


def test_gradient():

    Ahat = np.array([
        [1.01, 0.01, 0],
        [0.01, 1.01, 0.01],
        [0, 0.01, 1.01]
    ])
    Bhat = np.eye(3)
    Ahat = utils.sample_2_to_2_ball(Ahat, 0.1)
    Bhat = utils.sample_2_to_2_ball(Bhat, 0.1)

    Q = 1e-3 * np.eye(3)
    R = np.eye(3)

    n, p = Bhat.shape

    th = np.hstack((Ahat, Bhat))

    def func(x):
        x = x.reshape((n, n + p))
        A = x[:, :n]
        B = x[:, n:]
        return function_value(Q, R, A, B)

    def grad(x):
        x = x.reshape((n, n + p))
        A = x[:, :n]
        B = x[:, n:]
        g = gradient(Q, R, A, B)
        return g.flatten()

    err = check_grad(func, grad, th.flatten())
    assert err <= 1e-4, "numerical error of gradient is {}".format(err)
