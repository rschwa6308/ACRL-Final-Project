import numpy as np
from qpsolvers import solve_qp
from rover import *

# MPC SOLVING
#  Input: Current state [x], reference trajectory [X,U], cost matrices Q, R, Q_f, MPC horizon T, linearized dynamics matrices A, B
#  Output: control u

def mpc_control(x, x_ref, u_ref, Q, R, Qf, T, A, B, xeq, ueq, rover):
    """
    Input: current state x, reference trajectory x_ref, u_ref, cost matrices Q, R, Q_f,
            MPC horizon T, linearized dynamics matrices A, B
    Output: MPC control u
    """
    n = 4  # state dimension
    m = 2  # control dimension

    I = np.eye(n)

    xeq = xeq.reshape(n,1)
    ueq = ueq.reshape(m,1)

    P = np.zeros([(m + n) * T, (m + n) * T])  # quadratic cost 2nd order term
    q = np.zeros([(m + n) * T, 1])  # quadratic cost 1st order term
    C = np.zeros([n * T, (m + n) * T])  # equality constraints LHS
    d = np.zeros([n * T, 1])  # equality constraints RHS
    G = np.zeros([6 * T, (m + n) * T])  # inequality constraints LHS
    h = np.zeros([6 * T, 1])  # inequality constraints RHS

    # state and control constraints from 'rover.py'
    # self.wheel_angle_limit = 0.5  # [radians]
    # self.wheel_angle_velocity_limit = 1  # [radians / sec]
    # self.velocity_limit = 1  # [m/s]
    for i in range(T):
        P[(m + n) * i:(m + n) * i + m, (m + n) * i:(m + n) * i + m] = R
        P[(m + n) * i + m:(m + n) * i + m + n, (m + n) * i + m:(m + n) * i + m + n] = Q
        q[(m + n) * i:(m + n) * i + m, :] = -np.matmul(R, u_ref[i*m:i*m+m, :] - ueq)
        q[(m + n) * i + m:(m + n) * i + m + n, :] = -np.matmul(Q, x_ref[i*n:i*n+n, :] - xeq)

        C[n * i: n * i + n,  (m + n) * i:  (m + n) * i + m] = B
        C[n * i: n * i + n, (m + n) * i+m:  (m + n) * i + m+n] = -I
        if i > 0:
            C[n * i: n * i + n, (m + n) * i - n:  (m + n) * i] = A

        G[6 * i, (m + n) * i + 3] = 1  # angle limit psi
        G[6 * i + 1, (m + n) * i + 4] = 1  # velocity limit v
        G[6 * i + 2, (m + n) * i + 5] = 1  # angle_velocity limit psi_dot
        G[6 * i + 3, (m + n) * i + 3] = -1
        G[6 * i + 4, (m + n) * i + 4] = -1
        G[6 * i + 5, (m + n) * i + 5] = -1

        h[6 * i, :] = 0.5
        h[6 * i + 1, :] = 1
        h[6 * i + 2, :] = 1
        h[6 * i + 3, :] = 0.5
        h[6 * i + 4, :] = 1
        h[6 * i + 5, :] = 1

    P[-n:, -n:] = Qf
    q[-n:, :] = -np.matmul(Qf, x_ref[-n:, :] - xeq)

    # need to reshape before passing to solver
    q = q.reshape(((m + n) * T, ))
    h = h.reshape((6 * T, ))
    d = d.reshape((n * T, ))

    res = solve_qp(P, q, G, h, C, d, solver='quadprog')
    # solvers.options['feastol'] = 1e-8
    # solvers.options['show_progress'] = False
    # res = solvers.qp(P, q, G, h, C, d)

    u = res[0:m]  # control at the first time step

    return u, res

def mpc_control_iterative(x, x_ref, u_ref, Q, R, Qf, T, res, x_eq, u_eq, rover):
    """
    MPC with iterative linearization
    Input: current state x, reference trajectory x_ref, u_ref, cost matrices Q, R, Q_f,
            MPC horizon T, control and state sequences res from the previous MPC, x_eq from the previous MPC
    Output: MPC control u
    """
    n = 4  # state dimension
    m = 2  # control dimension

    I = np.eye(n)

    # xeq = xeq.reshape(n,1)
    # ueq = ueq.reshape(m,1)

    P = np.zeros([(m + n) * T, (m + n) * T])  # quadratic cost 2nd order term
    q = np.zeros([(m + n) * T, 1])  # quadratic cost 1st order term
    C = np.zeros([n * T, (m + n) * T])  # equality constraints LHS
    d = np.zeros([n * T, 1])  # equality constraints RHS
    G = np.zeros([6 * T, (m + n) * T])  # inequality constraints LHS
    h = np.zeros([6 * T, 1])  # inequality constraints RHS

    # state and control constraints from 'rover.py'
    # self.wheel_angle_limit = 0.5  # [radians]
    # self.wheel_angle_velocity_limit = 1  # [radians / sec]
    # self.velocity_limit = 1  # [m/s]
    for i in range(T):
        ueq = res[(m + n) * i:(m + n) * i + m].reshape(-1, 1) + u_eq.reshape(-1, 1)
        if i == 0:
            xeq = x.reshape(-1, 1)
        else:
            xeq = res[(m + n) * i + m:(m + n) * (i + 1)].reshape(-1, 1) + x_eq.reshape(-1, 1)

        A, B, _, _ = linearize_dynamics(xeq, ueq)

        P[(m + n) * i:(m + n) * i + m, (m + n) * i:(m + n) * i + m] = R
        P[(m + n) * i + m:(m + n) * i + m + n, (m + n) * i + m:(m + n) * i + m + n] = Q
        q[(m + n) * i:(m + n) * i + m, :] = -np.matmul(R, u_ref[i*m:i*m+m, :] - ueq)
        q[(m + n) * i + m:(m + n) * i + m + n, :] = -np.matmul(Q, x_ref[i*n:i*n+n, :] - xeq)

        C[n * i: n * i + n,  (m + n) * i:  (m + n) * i + m] = B
        C[n * i: n * i + n, (m + n) * i+m:  (m + n) * i + m+n] = -I
        if i > 0:
            C[n * i: n * i + n, (m + n) * i - n:  (m + n) * i] = A

        G[6 * i, (m + n) * i + 3] = 1  # angle limit psi
        G[6 * i + 1, (m + n) * i + 4] = 1  # velocity limit v
        G[6 * i + 2, (m + n) * i + 5] = 1  # angle_velocity limit psi_dot
        G[6 * i + 3, (m + n) * i + 3] = -1
        G[6 * i + 4, (m + n) * i + 4] = -1
        G[6 * i + 5, (m + n) * i + 5] = -1

        h[6 * i, :] = 0.5
        h[6 * i + 1, :] = 1
        h[6 * i + 2, :] = 1
        h[6 * i + 3, :] = 0.5
        h[6 * i + 4, :] = 1
        h[6 * i + 5, :] = 1

    P[-n:, -n:] = Qf
    q[-n:, :] = -np.matmul(Qf, x_ref[-n:, :] - xeq)

    # need to reshape before passing to solver
    q = q.reshape(((m + n) * T, ))
    h = h.reshape((6 * T, ))
    d = d.reshape((n * T, ))

    res = solve_qp(P, q, G, h, C, d, solver='quadprog')
    # solvers.options['feastol'] = 1e-8
    # solvers.options['show_progress'] = False
    # res = solvers.qp(P, q, G, h, C, d)

    u = res[0:m]  # control at the first time step

    return u

# TODO: linearization function
#  Input: state x, and control u,
#  output: A and B matrices, x equilibrium point xeq, u equilibrium point ueq

def linearize_dynamics(x, u, dt=0.1, w_b=1.0):
    """
    Input: state x, control u, time step dt
    Output: A and B matrices, x equilibrium point xeq, u equilibrium point ueq
    """

    # Select equilibrium point
    xeq = x
    ueq = u

    # Extract state and control values
    v_t = ueq[0]
    theta_t = xeq[2]
    psi_t = xeq[3]

    # Construction linearization matrices
    A_lin = np.array([[1, 0, -v_t * dt * np.cos(psi_t) * np.sin(theta_t), -v_t * dt * np.sin(psi_t) * np.cos(theta_t)],
                      [0, 1,  v_t * dt * np.cos(psi_t) * np.cos(theta_t), -v_t * dt * np.sin(psi_t) * np.sin(theta_t)],
                      [0, 0, 1, v_t * dt * np.cos(psi_t) * w_b/2],
                      [0, 0, 0, 1]])
    
    B_lin = np.array([[dt * np.cos(psi_t) * np.cos(theta_t), 0],
                      [dt * np.cos(psi_t) * np.sin(theta_t), 0],
                      [dt * np.sin(psi_t) * w_b/2, 0],
                      [0, dt]])

    return A_lin, B_lin, xeq, ueq


# for debug
if __name__ == "__main__":
    # QP solver test. Passed

    # import qpsolvers
    # print(qpsolvers.available_solvers)
    #
    # M = np.array([[1., 2., 0.], [-8., 3., 2.], [0., 1., 1.]])
    # P = np.dot(M.T, M)  # quick way to build a symmetric matrix
    # q = np.dot(np.array([3., 2., 3.]), M).reshape((3,))
    # G = np.array([[1., 2., 1.], [2., 0., 1.], [-1., 2., -1.]])
    # h = np.array([3., 2., -2.]).reshape((3,))
    # A = np.array([1., 1., 1.])
    # b = np.array([1.])
    #
    # res = solve_qp(P, q, G, h, A, b, solver='quadprog')
    #
    # print(res)
    #
    # print(b.shape)
    # print(h.shape)

    # x0 = np.transpose(np.array([0, 0, 0, np.pi/2]))
    x0 = State(0, 0, 0, np.pi/2)
    rover = Rover(x0)
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
    
    xeq = x0.state.reshape(n,1)
    ueq = u_ref[0:m, :]

    u, res = mpc_control(x0.state, x_ref, u_ref, Q, R, Qf, T, A, B, xeq, ueq, rover)

    print(res)

    u = mpc_control_iterative(x0.state, x_ref, u_ref, Q, R, Qf, T, res, xeq, ueq, rover)

    print(u)


