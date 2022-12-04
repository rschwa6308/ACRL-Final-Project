from cvxopt import solvers, matrix

from qpsolvers import solve_qp

# solvers.options['feastol'] = 1e-8
# solvers.options['show_progress'] = False

# x = solve_qp(P, q, G, h)

# sol = solvers.qp(Q, p, A, b)

# TODO: MPC SOLVING
#  INPUT: Current state [x], reference trajectory [X,U], cost matrices Q, R, Qfinal
# Output: control u 

# TODO: linearization function 
# Input: state x, and control u, 
# output: A and B matricies


