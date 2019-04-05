import numpy as np
import matplotlib.pyplot as plt
import math

def make_sample_traj(NUM, dt=0.01, a=30.):
    """
    make sample trajectory
    Parameters
    ------------
    NUM : int
    dt : float
    a : float

    Returns
    ----------
    traj_xs : list
    traj_ys : list
    """
    DELAY = 2.
    traj_xs = [] 
    traj_ys = []

    for i in range(NUM):
        traj_xs.append(i * 0.1)
        traj_ys.append(a * math.sin(dt * i / DELAY))

    plt.plot(traj_xs, traj_ys)
    plt.show()

    return traj_xs, traj_ys

