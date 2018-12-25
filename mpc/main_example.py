import numpy as np
import matplotlib.pyplot as plt
import math

from mpc_func import MpcController
from control import matlab

class FirstOrderSystem():
    """FirstOrderSystemWithStates

    Attributes
    -----------
    states : float
        system states
    A : numpy.ndarray
        system matrix
    B : numpy.ndarray
        control matrix
    C : numpy.ndarray
        observation matrix
    history_state : list
        time history of state
    """
    def __init__(self, A, B, C, D=None, init_states=None):
        """
        Parameters
        -----------
        A : numpy.ndarray
            system matrix
        B : numpy.ndarray
            control matrix
        C : numpy.ndarray
            observation matrix
        C : numpy.ndarray
            directly matrix
        init_state : float, optional
            initial state of system default is None
        history_xs : list
            time history of system states
        """

        if init_states is not None:
            self.states = init_states
        
        self.A = A
        self.B = B
        self.C = C

        if D is not None:
            self.D = D

        self.xs = np.zeros(self.A.shape[0])

        self.history_xs = [init_states]

    def update_state(self, us, dt=0.01):
        """calculating input
        Parameters
        ------------
        u : float
            input of system in some cases this means the reference
        dt : float in seconds, optional
            sampling time of simulation, default is 0.01 [s]
        """
        temp = self.xs.reshape(-1, 1)

        # solve Runge-Kutta
        k0 = dt * (np.dot(self.A, temp) + np.dot(self.B, us)) 
        k1 = dt * (np.dot(self.A, temp + k0/2.) + np.dot(self.B, us))
        k2 = dt * (np.dot(self.A, temp + k1/2.) + np.dot(self.B, us))
        k3 = dt * (np.dot(self.A, temp + k2) + np.dot(self.B, us))

        self.xs +=  ((k0 + 2 * k1 + 2 * k2 + k3) / 6.).flatten()

        # for oylar
        # self.state += k0

        # save
        self.history_xs.append(self.xs)

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

    # make simulator with coninuous matrix
    plant = FirstOrderSystem(A, B, C)

    # create system
    sysc = matlab.ss(A, B, C, D)
    # discrete system
    sysd = matlab.c2d(sysc, dt)

    Ad = sysd.A
    Bd = sysd.B

    # evaluation function weight
    Q = np.diag([1., 1., 1., 1.])
    R = np.diag([1., 1.])
    pre_step = 3

    # make controller with discreted matrix
    controller = MpcController(Ad, Bd, Q, R, pre_step)
    controller.initialize_controller()

    xs = np.array([0., 0., 0., 0.])

    for i in range(iteration_num):
        controller.calc_input(xs)

    # states = plant.states
    # controller.calc_input

if __name__ == "__main__":
    main()







