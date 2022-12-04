from cvxopt import solvers, matrix
import numpy as np
from qpsolvers import solve_qp

# solvers.options['feastol'] = 1e-8
# solvers.options['show_progress'] = False

# x = solve_qp(P, q, G, h)

# sol = solvers.qp(Q, p, A, b)

# TODO: MPC SOLVING
#  Input: Current state [x], reference trajectory [X,U], cost matrices Q, R, Q_f, MPC horizon T, linearized dynamics matrices A, B
#  Output: control u

def mpc_control(x, x_ref, u_ref, Q, R, Qf, T, A, B):
    """
    Input: current state x, reference trajectory x_ref, u_ref, cost matrices Q, R, Q_f,
            MPC horizon T, linearized dynamics matrices A, B
    Output: MPC control u
    """
    n = 4  # state dimension
    m = 2  # control dimension

    I = np.eye(n)

    xeq = x.reshape(n,1)
    ueq = u_ref[0:m, :]

    P = np.zeros([(m + n) * T, (m + n) * T])  # quadratic cost 2nd order term
    q = np.zeros([(m + n) * T, 1])  # quadratic cost 1st order term
    C = np.zeros([n * T, (m + n) * T])  # equality constraints LHS
    d = np.zeros([n * T, 1])  # equality constraints RHS

    for i in range(T):
        P[(m + n) * i:(m + n) * i + m, (m + n) * i:(m + n) * i + m] = R
        P[(m + n) * i + m:(m + n) * i + m + n, (m + n) * i + m:(m + n) * i + m + n] = Q

        q[(m + n) * i:(m + n) * i + m, :] = -np.matmul(R, u_ref[i*m:i*m+m, :] - ueq)
        q[(m + n) * i + m:(m + n) * i + m + n, :] = -np.matmul(Q, x_ref[i*n:i*n+n, :] - xeq)

        C[n * i: n * i + n,  (m + n) * i:  (m + n) * i + m] = B
        C[n * i: n * i + n, (m + n) * i+m:  (m + n) * i + m+n] = -I
        if i > 0:
            C[n * i: n * i + n, (m + n) * i - n:  (m + n) * i] = A

    P[-n:, -n:] = Qf
    q[-n:, :] = -np.matmul(Qf, x_ref[-n:, :] - xeq)

    res = solve_qp(P, q, C, d)

    u = res[0:m]  # control at the first time step

    return u

# TODO: linearization function
#  Input: state x, and control u,
#  output: A and B matrices, x equilibrium point xeq, u equilibrium point ueq

def linearize_dynamics(x, u):
    """
    Input: state x, control u
    Output: A and B matrices, x equilibrium point xeq, u equilibrium point ueq
    """

    return A_lin, B_lin, xeq, ueq


# for debug
if __name__ == "__main__":
    x0 = np.transpose(np.array([0, 0, 0, np.pi/2]))
    n = 4  # state dimension
    m = 2  # control dimension
    T = 5  # MPC horizon

    # cost functions
    Q = np.eye(n)
    R = np.eye(m)
    Qf = 10 * np.eye(n)

    A = np.random.rand(n, n)
    B = np.random.rand(n, m)

    u_ref = np.ones([m*T, 1])
    x_ref = 2*np.ones([n*T, 1])

    u = mpc_control(x0, x_ref, u_ref, Q, R, Qf, T, A, B)

