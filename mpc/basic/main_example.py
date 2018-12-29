import numpy as np
import matplotlib.pyplot as plt
import math
import copy

from mpc_func_with_scipy import MpcController as MpcController_scipy
from mpc_func_with_cvxopt import MpcController as MpcController_cvxopt
from control import matlab

class FirstOrderSystem():
    """FirstOrderSystemWithStates

    Attributes
    -----------
    xs : numpy.ndarray
        system states
    A : numpy.ndarray
        system matrix
    B : numpy.ndarray
        control matrix
    C : numpy.ndarray
        observation matrix
    history_xs : list
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
        D : numpy.ndarray
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
        u : numpy.ndarray
            inputs of system in some cases this means the reference
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

        self.xs +=  ((k0 + 2 * k1 + 2 * k2 + k3) / 6.).flatten()

        # for oylar
        # self.xs += k0.flatten()
        # print("xs = {0}".format(self.xs))

        # save
        save_states = copy.deepcopy(self.xs)
        self.history_xs.append(save_states)

def main():
    dt = 0.05
    simulation_time = 30 # in seconds
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
    init_xs = np.array([0., 0., 0., 0.])
    plant = FirstOrderSystem(A, B, C, init_states=init_xs)

    # create system
    sysc = matlab.ss(A, B, C, D)
    # discrete system
    sysd = matlab.c2d(sysc, dt)

    Ad = sysd.A
    Bd = sysd.B

    # evaluation function weight
    Q = np.diag([1., 1., 1., 1.])
    R = np.diag([1., 1.])
    pre_step = 10

    # make controller with discreted matrix
    # please check the solver, if you want to use the scipy, set the MpcController_scipy
    controller = MpcController_cvxopt(Ad, Bd, Q, R, pre_step,
                               dt_input_upper=np.array([0.25 * dt, 0.25 * dt]), dt_input_lower=np.array([-0.5 * dt, -0.5 * dt]),
                               input_upper=np.array([1. ,3.]), input_lower=np.array([-1., -3.]))

    controller.initialize_controller()

    for i in range(iteration_num):
        print("simulation time = {0}".format(i))
        reference = np.array([[0., 0., -5., 7.5] for _ in range(pre_step)]).flatten()   
        states = plant.xs
        opt_u = controller.calc_input(states, reference)
        plant.update_state(opt_u, dt=dt)

    history_states = np.array(plant.history_xs)

    time_history_fig = plt.figure()
    x_fig = time_history_fig.add_subplot(411)
    y_fig = time_history_fig.add_subplot(412)
    v_x_fig = time_history_fig.add_subplot(413)
    v_y_fig = time_history_fig.add_subplot(414)

    v_x_fig.plot(np.arange(0, simulation_time+0.01, dt), history_states[:, 0])
    v_x_fig.plot(np.arange(0, simulation_time+0.01, dt), [0. for _ in range(iteration_num+1)], linestyle="dashed")
    v_x_fig.set_xlabel("time [s]")
    v_x_fig.set_ylabel("v_x")

    v_y_fig.plot(np.arange(0, simulation_time+0.01, dt), history_states[:, 1])
    v_y_fig.plot(np.arange(0, simulation_time+0.01, dt), [0. for _ in range(iteration_num+1)], linestyle="dashed")
    v_y_fig.set_xlabel("time [s]")
    v_y_fig.set_ylabel("v_y")

    x_fig.plot(np.arange(0, simulation_time+0.01, dt), history_states[:, 2])
    x_fig.plot(np.arange(0, simulation_time+0.01, dt), [-5. for _ in range(iteration_num+1)], linestyle="dashed")
    x_fig.set_xlabel("time [s]")
    x_fig.set_ylabel("x")

    y_fig.plot(np.arange(0, simulation_time+0.01, dt), history_states[:, 3])
    y_fig.plot(np.arange(0, simulation_time+0.01, dt), [7.5 for _ in range(iteration_num+1)], linestyle="dashed")
    y_fig.set_xlabel("time [s]")
    y_fig.set_ylabel("y")
    time_history_fig.tight_layout()
    plt.show()

    history_us = np.array(controller.history_us)
    input_history_fig = plt.figure()
    u_1_fig = input_history_fig.add_subplot(211)
    u_2_fig = input_history_fig.add_subplot(212)

    u_1_fig.plot(np.arange(0, simulation_time+0.01, dt), history_us[:, 0])
    u_1_fig.set_xlabel("time [s]")
    u_1_fig.set_ylabel("u_x")
    
    u_2_fig.plot(np.arange(0, simulation_time+0.01, dt), history_us[:, 1])
    u_2_fig.set_xlabel("time [s]")
    u_2_fig.set_ylabel("u_y")
    input_history_fig.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()