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

        self.xs +=  ((k0 + 2 * k1 + 2 * k2 + k3) / 6.).flatten()

        # for oylar
        # self.xs += k0.flatten()

        # print("xs = {0}".format(self.xs))
        # a = input()
        # save
        save_states = copy.deepcopy(self.xs)
        self.history_xs.append(save_states)
        # print(self.history_xs)

def main():
    dt = 0.05
    simulation_time = 50 # in seconds
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
    plant_cvxopt = FirstOrderSystem(A, B, C, init_states=init_xs)
    plant_scipy = FirstOrderSystem(A, B, C, init_states=init_xs)

    # create system
    sysc = matlab.ss(A, B, C, D)
    # discrete system
    sysd = matlab.c2d(sysc, dt)

    Ad = sysd.A
    Bd = sysd.B

    # evaluation function weight
    Q = np.diag([1., 1., 10., 10.])
    R = np.diag([0.01, 0.01])
    pre_step = 5

    # make controller with discreted matrix
    # please check the solver, if you want to use the scipy, set the MpcController_scipy
    controller_cvxopt = MpcController_cvxopt(Ad, Bd, Q, R, pre_step,
                               dt_input_upper=np.array([0.25 * dt, 0.25 * dt]), dt_input_lower=np.array([-0.5 * dt, -0.5 * dt]),
                               input_upper=np.array([1. ,3.]), input_lower=np.array([-1., -3.]))
    
    controller_scipy = MpcController_scipy(Ad, Bd, Q, R, pre_step,
                               dt_input_upper=np.array([0.25 * dt, 0.25 * dt]), dt_input_lower=np.array([-0.5 * dt, -0.5 * dt]),
                               input_upper=np.array([1. ,3.]), input_lower=np.array([-1., -3.]))

    controller_cvxopt.initialize_controller()
    controller_scipy.initialize_controller()

    for i in range(iteration_num):
        print("simulation time = {0}".format(i))
        reference = np.array([[0., 0., -5., 7.5] for _ in range(pre_step)]).flatten()   

        states_cvxopt = plant_cvxopt.xs
        states_scipy = plant_scipy.xs

        opt_u_cvxopt = controller_cvxopt.calc_input(states_cvxopt, reference)
        opt_u_scipy = controller_scipy.calc_input(states_scipy, reference)

        plant_cvxopt.update_state(opt_u_cvxopt)
        plant_scipy.update_state(opt_u_scipy)

    history_states_cvxopt = np.array(plant_cvxopt.history_xs)
    history_states_scipy = np.array(plant_scipy.history_xs)

    time_history_fig = plt.figure(dpi=75)
    x_fig = time_history_fig.add_subplot(411)
    y_fig = time_history_fig.add_subplot(412)
    v_x_fig = time_history_fig.add_subplot(413)
    v_y_fig = time_history_fig.add_subplot(414)

    v_x_fig.plot(np.arange(0, simulation_time+0.01, dt), history_states_cvxopt[:, 0], label="cvxopt")
    v_x_fig.plot(np.arange(0, simulation_time+0.01, dt), history_states_scipy[:, 0], label="scipy", linestyle="dashdot")
    v_x_fig.plot(np.arange(0, simulation_time+0.01, dt), [0. for _ in range(iteration_num+1)], linestyle="dashed")
    v_x_fig.set_xlabel("time [s]")
    v_x_fig.set_ylabel("v_x")
    v_x_fig.legend()

    v_y_fig.plot(np.arange(0, simulation_time+0.01, dt), history_states_cvxopt[:, 1], label="cvxopt")
    v_y_fig.plot(np.arange(0, simulation_time+0.01, dt), history_states_scipy[:, 1], label="scipy", linestyle="dashdot")
    v_y_fig.plot(np.arange(0, simulation_time+0.01, dt), [0. for _ in range(iteration_num+1)], linestyle="dashed")
    v_y_fig.set_xlabel("time [s]")
    v_y_fig.set_ylabel("v_y")
    v_y_fig.legend()

    x_fig.plot(np.arange(0, simulation_time+0.01, dt), history_states_cvxopt[:, 2], label="cvxopt")
    x_fig.plot(np.arange(0, simulation_time+0.01, dt), history_states_scipy[:, 2], label="scipy", linestyle="dashdot")
    x_fig.plot(np.arange(0, simulation_time+0.01, dt), [-5. for _ in range(iteration_num+1)], linestyle="dashed")
    x_fig.set_xlabel("time [s]")
    x_fig.set_ylabel("x")

    y_fig.plot(np.arange(0, simulation_time+0.01, dt),  history_states_cvxopt[:, 3], label ="cvxopt")
    y_fig.plot(np.arange(0, simulation_time+0.01, dt),  history_states_scipy[:, 3], label="scipy", linestyle="dashdot")
    y_fig.plot(np.arange(0, simulation_time+0.01, dt), [7.5 for _ in range(iteration_num+1)], linestyle="dashed")
    y_fig.set_xlabel("time [s]")
    y_fig.set_ylabel("y")
    time_history_fig.tight_layout()
    plt.show()

    history_us_cvxopt = np.array(controller_cvxopt.history_us)
    history_us_scipy = np.array(controller_scipy.history_us)

    input_history_fig = plt.figure(dpi=75)
    u_1_fig = input_history_fig.add_subplot(211)
    u_2_fig = input_history_fig.add_subplot(212)

    u_1_fig.plot(np.arange(0, simulation_time+0.01, dt), history_us_cvxopt[:, 0], label="cvxopt")
    u_1_fig.plot(np.arange(0, simulation_time+0.01, dt), history_us_scipy[:, 0], label="scipy", linestyle="dashdot")
    u_1_fig.set_xlabel("time [s]")
    u_1_fig.set_ylabel("u_x")
    u_1_fig.legend()
    
    u_2_fig.plot(np.arange(0, simulation_time+0.01, dt), history_us_cvxopt[:, 1], label="cvxopt")
    u_2_fig.plot(np.arange(0, simulation_time+0.01, dt), history_us_scipy[:, 1], label="scipy", linestyle="dashdot")
    u_2_fig.set_xlabel("time [s]")
    u_2_fig.set_ylabel("u_y")
    u_2_fig.legend()
    input_history_fig.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()