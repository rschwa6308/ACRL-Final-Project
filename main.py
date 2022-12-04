import numpy as np
from matplotlib import pyplot as plt
from controllers import *
from dynamics import *
from simulation import *
from observations import *


N = 100

x0 = np.array([0, 0, np.pi/2])

xs, ys, us = simulate(ackermann_dynamics, perfect_observations, sample_controller_3, x0, N)

pxs, pys, heading_angles = zip(*xs)
speeds = [u[0] for u in us]

# plot the trajectory
plt.plot(pxs, pys)
plt.axis("equal")

# plot some nice pose arrows
arrows = [(
    pxs[k], pys[k],
    speeds[k] * np.cos(heading_angles[k]),
    speeds[k] * np.sin(heading_angles[k])
) for k in range(N)]
arrows_select = arrows[::N//10] + [arrows[-1]]   # plot sparse selection of states (including final)
print(len(arrows_select))
plt.quiver(
    *zip(*arrows_select),
    color=["red"] + ["black"]*(len(arrows_select)-2) + ["green"]
)     

plt.show()
