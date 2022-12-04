from cvxopt import solvers, matrix

from qpsolvers import solve_qp

solvers.options['feastol'] = 1e-8
solvers.options['show_progress'] = False

x = solve_qp(P, q, G, h)

sol = solvers.qp(Q, p, A, b)