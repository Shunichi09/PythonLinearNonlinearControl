import numpy as np
import matplotlib.pyplot as plt
import math

from mpc_func import MpcController
# from simulator_func import FirstOrderSystem
from control import matlab

def main():
    dt = 0.01
    simulation_time = 100 # in seconds
    iteration_num = int(simulation_time / dt)

    # you must be care about this matrix
    # these A and B are for continuos system if you want to use discret system matrix please skip this step
    tau = 0.53
    A = np.array([[1./tau, 0., 0., 0.],
                  [0., 1./tau, 0., 0.],
                  [1., 0., 0., 0.], 
                  [1., 0., 0., 0.]])
    B = np.array([[1./tau, 0.],
                  [0., 1./tau],
                  [0., 0.],
                  [0., 0.]])

    C = np.eye(4)
    D = np.zeros((4, 2))

    # create system
    sysc = matlab.ss(A, B, C, D)
    # discrete system
    sysd = matlab.c2d(sysc, dt)

    Ad = sysd.A
    Bd = sysd.B

    Q = np.diag([1., 1., 1., 1.])
    R = np.diag([1., 1.])
    pre_step = 3

    # make controller
    controller = MpcController(Ad, Bd, Q, R, pre_step)
    controller.initialize_controller()

    # make simulator
    # plant = FirstOrderSystem(tau)

    """
    for i in range(iteration_num):
    """

    # states = plant.states
    # controller.calc_input

if __name__ == "__main__":
    main()







