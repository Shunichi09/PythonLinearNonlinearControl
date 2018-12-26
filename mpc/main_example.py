import numpy as np
import matplotlib.pyplot as plt
import math
import copy

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

        self.A = A
        self.B = B
        self.C = C

        if D is not None:
            self.D = D

        self.xs = np.zeros(self.A.shape[0])

        if init_states is not None:
            self.xs = copy.deepcopy(init_states)

        self.history_xs = [init_states]

    def update_state(self, u, dt=0.01):
        """calculating input
        Parameters
        ------------
        u : float
            input of system in some cases this means the reference
        dt : float in seconds, optional
            sampling time of simulation, default is 0.01 [s]
        """
        temp_x = self.xs.reshape(-1, 1)
        temp_u = u.reshape(-1, 1)

        # solve Runge-Kutta
        k0 = dt * (np.dot(self.A, temp_x) + np.dot(self.B, temp_u)) 
        k1 = dt * (np.dot(self.A, temp_x + k0/2.) + np.dot(self.B, temp_u))
        k2 = dt * (np.dot(self.A, temp_x + k1/2.) + np.dot(self.B, temp_u))
        k3 = dt * (np.dot(self.A, temp_x + k2) + np.dot(self.B, temp_u))

        # self.xs +=  ((k0 + 2 * k1 + 2 * k2 + k3) / 6.).flatten()

        # for oylar
        self.xs += k0.flatten()

        # print("xs = {0}".format(self.xs))
        # a = input()
        # save
        save_states = copy.deepcopy(self.xs)
        self.history_xs.append(save_states)
        # print(self.history_xs)

def main():
    dt = 0.01
    simulation_time = 300 # in seconds
    iteration_num = int(simulation_time / dt)

    # you must be care about this matrix
    # these A and B are for continuos system if you want to use discret system matrix please skip this step
    tau = 0.63
    A = np.array([[-1./tau, 0., 0., 0.],
                  [0., -1./tau, 0., 0.],
                  [1., 0., 0., 0.], 
                  [0., 1., 0., 0.]])
    B = np.array([[1./tau, 0.],
                  [0., 1./tau],
                  [0., 0.],
                  [0., 0.]])

    C = np.eye(4)
    D = np.zeros((4, 2))

    # make simulator with coninuous matrix
    init_xs = np.array([0., 0., -3000., 50.])
    plant = FirstOrderSystem(A, B, C, init_states=init_xs)

    # create system
    sysc = matlab.ss(A, B, C, D)
    # discrete system
    sysd = matlab.c2d(sysc, dt)

    Ad = sysd.A
    Bd = sysd.B

    # evaluation function weight
    Q = np.diag([1., 1., 1., 1.])
    R = np.diag([100., 100.])
    pre_step = 3

    # make controller with discreted matrix
    controller = MpcController(Ad, Bd, Q, R, pre_step)
    controller.initialize_controller()

    for i in range(iteration_num):
        print("simulation time = {0}".format(i))
        reference = np.array([0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.])
        controller.calc_input(plant.xs, reference)
        
        states = plant.xs
        opt_u = controller.calc_input(states, reference)
        plant.update_state(opt_u)

    history_states = np.array(plant.history_xs)

    print(history_states[:, 2])

    plt.plot(np.arange(0, simulation_time+0.01, dt), history_states[:, 0])
    plt.plot(np.arange(0, simulation_time+0.01, dt), history_states[:, 1])
    plt.plot(np.arange(0, simulation_time+0.01, dt), history_states[:, 2], linestyle="dashed")
    plt.plot(np.arange(0, simulation_time+0.01, dt), history_states[:, 3])
    plt.show()

if __name__ == "__main__":
    main()







