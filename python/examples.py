import numpy as np
from scipy import linalg as ln

def chained_integrator_dynamics(dt=0.1, n=2, decay=1, amplification = 1, fullB=False):
    ''' forward euler discretization of
        dx_i/dt = x_{i+1}; dx_n/dt = u
        with added decay of states or amplification
        of integration terms '''
    if hasattr(decay, "__len__"):
        assert len(decay) == n, 'incorrect number of decay coefficients'
    else:
        decay = decay * np.ones(n)
    if hasattr(amplification, "__len__"):
        assert len(amplification) == n, 'incorrect number of amplification coefficients'
    else:
        amplification = amplification * np.ones(n)

    Astar = np.diag(decay) + dt * np.diag(amplification[:-1], k=1)
    if fullB:
        Bstar = np.eye(n)
    else:
        Bstar = np.zeros(n, dtype=float)
        Bstar[-1] = dt * amplification[-1]
        Bstar.reshape((n, 1))
    return Astar, Bstar

def transient_dynamics(diag_coeff=1.01, lowerdiag=1.5, n=3):
    ''' a state transition matrix with values along
        the diagonal and lower sub-diagonal  
    '''
    if hasattr(diag_coeff, "__len__"):
        assert len(diag_coeff) == n, 'incorrect number of diagonal coefficients'
    else:
        diag_coeff = diag_coeff * np.ones(n)
    if hasattr(lowerdiag, "__len__"):
        assert len(lowerdiag) == n-1, 'incorrect number of lower diagonal coefficients'
    else:
        lowerdiag = lowerdiag * np.ones(n-1)

    Astar = np.diag(diag_coeff) + np.diag(lowerdiag, k=-1)
    Bstar = np.eye(n)
    return Astar, Bstar

def inverted_pendulum_dynamics():
    ''' linearization of the inverted pendulum around its 
        equilibrium point
    '''
    mc = 1
    mp = 1

    h = 0.05 
    g = 9.81 
    tau = 1.0
    A_star = np.array([
        [1, 0, h, 0],
        [0, 1, 0, h],
        [0, g*h*mp/mc, 1, -h*tau/mc],
        [0, g*h*(1 + mp/mc), 0, -h*tau*(mc + mp)/(mc*mp) + 1] 
    ])   

    B_star = np.array([0, 0, h/mc, h/mc]).reshape((4, 1))
    return A_star, B_star

def unstable_laplacian_dynamics(n=3):
    Adj = generate_line_adjacency(n)
    node_dynamics = 0.03 * np.ones(n)
    node_dynamics[0] = 0.02; node_dynamics[-1] = 0.02
    Astar = generate_graph_dynamics(Adj, 0.01, node_dynamics=node_dynamics)
    Bstar = np.eye(n)
    return Astar, Bstar

def generate_line_adjacency(n, weights=1):
    if hasattr(weights, "__len__"):
        assert len(weights) == n-1, 'incorrect number of weights'
    else:
        weights = weights * np.ones(n-1)
    return np.diag(weights, k=1) + np.diag(weights, k=-1)

def generate_graph_dynamics(Adj, dt, node_dynamics=None):
    ''' forward euler discretization of dynamics in
        http://vcp.med.harvard.edu/papers/jg-lap-dyn.pdf '''
    D = np.diag(np.sum(Adj, 1))
    L = Adj - D
    A = np.eye(Adj.shape[0]) + dt * L
    if node_dynamics is not None:
        A = A + np.diag(node_dynamics)
    return A

