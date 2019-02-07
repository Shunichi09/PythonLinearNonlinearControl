import numpy as np
import matplotlib.pyplot as plt
import math
import copy

from mpc_func_with_cvxopt import MpcController as MpcController_cvxopt
from animation import AnimDrawer
from control import matlab
from coordinate_trans import coordinate_transformation_in_angle, coordinate_transformation_in_position

class WheeledSystem():
    """SampleSystem, this is the simulator
        Kinematic model of car

    Attributes
    -----------
    xs : numpy.ndarray
        system states, [x, y, phi, beta]
    history_xs : list
        time history of state
    """
    def __init__(self, init_states=None):
        """
        Palameters
        -----------
        init_state : float, optional, shape(3, )
            initial state of system default is None
        """
        self.NUM_STATE = 4
        self.xs = np.zeros(self.NUM_STATE)

        self.FRONT_WHEELE_BASE = 1.0
        self.REAR_WHEELE_BASE = 1.0

        if init_states is not None:
            self.xs = copy.deepcopy(init_states)

        self.history_xs = [init_states]

    def update_state(self, us, dt=0.01):
        """
        Palameters
        ------------
        u : numpy.ndarray
            inputs of system in some cases this means the reference
        dt : float in seconds, optional
            sampling time of simulation, default is 0.01 [s]
        """
        # for theta 1, theta 1 dot, theta 2, theta 2 dot
        k0 = [0.0 for _ in range(self.NUM_STATE)]
        k1 = [0.0 for _ in range(self.NUM_STATE)]
        k2 = [0.0 for _ in range(self.NUM_STATE)]
        k3 = [0.0 for _ in range(self.NUM_STATE)]

        functions = [self._func_x_1, self._func_x_2, self._func_x_3, self._func_x_4]

        # solve Runge-Kutta
        for i, func in enumerate(functions):
            k0[i] = dt * func(self.xs[0], self.xs[1], self.xs[2], self.xs[3], us[0], us[1])

        for i, func in enumerate(functions):
            k1[i] = dt * func(self.xs[0] + k0[0]/2., self.xs[1] + k0[1]/2., self.xs[2] + k0[2]/2.,  self.xs[3] + k0[3]/2, us[0], us[1])
        
        for i, func in enumerate(functions):
            k2[i] = dt * func(self.xs[0] + k1[0]/2., self.xs[1] + k1[1]/2., self.xs[2] + k1[2]/2., self.xs[3] + k1[3]/2., us[0], us[1])
        
        for i, func in enumerate(functions):
            k3[i] =  dt * func(self.xs[0] + k2[0], self.xs[1] + k2[1], self.xs[2] + k2[2], self.xs[3] + k2[3], us[0], us[1])
        
        self.xs[0] += (k0[0] + 2. * k1[0] + 2. * k2[0] + k3[0]) / 6.
        self.xs[1] += (k0[1] + 2. * k1[1] + 2. * k2[1] + k3[1]) / 6.
        self.xs[2] += (k0[2] + 2. * k1[2] + 2. * k2[2] + k3[2]) / 6.
        self.xs[3] += (k0[3] + 2. * k1[3] + 2. * k2[3] + k3[3]) / 6.
    
        # save
        save_states = copy.deepcopy(self.xs)
        self.history_xs.append(save_states)
        print(self.xs)

    def _func_x_1(self, y_1, y_2, y_3, y_4, u_1, u_2):
        """
        Parameters
        ------------
        y_1 : float
        y_2 : float
        y_3 : float
        u_1 : float
            system input
        u_2 : float
            system input
        """
        y_dot = u_1 * math.cos(y_3 + y_4)
        return y_dot
    
    def _func_x_2(self, y_1, y_2, y_3, y_4, u_1, u_2):
        """
        Parameters
        ------------
        y_1 : float
        y_2 : float
        y_3 : float
        u_1 : float
            system input
        u_2 : float
            system input
        """
        y_dot = u_1 * math.sin(y_3 + y_4)
        return y_dot
    
    def _func_x_3(self, y_1, y_2, y_3, y_4, u_1, u_2):
        """
        Parameters
        ------------
        y_1 : float
        y_2 : float
        y_3 : float
        u_1 : float
            system input
        u_2 : float
            system input
        """
        y_dot = u_1 / self.REAR_WHEELE_BASE * math.sin(y_4)
        return y_dot

    def _func_x_4(self, y_1, y_2, y_3, y_4, u_1, u_2):
        """
        """
        y_dot = math.atan2(self.REAR_WHEELE_BASE * math.tan(u_2) ,self.REAR_WHEELE_BASE + self.FRONT_WHEELE_BASE)

        return y_dot

def main():
    dt = 0.016
    simulation_time = 10 # in seconds
    iteration_num = int(simulation_time / dt)

    # you must be care about this matrix
    # these A and B are for continuos system if you want to use discret system matrix please skip this step
    # lineared car system
    WHEEL_BASE = 2.2
    tau = 0.01

    V = 5.0 # initialize

    
    delta_r = 0.

    A12 = (V / WHEEL_BASE) / (math.cos(delta_r)**2)
    A22 = (1. - 1. / tau)

    Ad = np.array([[1., V,   0.], 
                   [0., 1., A12],
                   [0., 0., A22]]) * dt

    Bd = np.array([[0.], [0.], [1. / tau]]) * dt

    W_D_0 = - (V / WHEEL_BASE) * delta_r / (math.cos(delta_r)**2)

    W_D = np.array([[0.], [W_D_0], [0.]]) * dt

    # make simulator with coninuous matrix
    init_xs_lead = np.array([5., 0., 0. ,0.])
    init_xs_follow = np.array([0., 0., 0., 0.])
    lead_car = WheeledSystem(init_states=init_xs_lead)
    follow_car = WheeledSystem(init_states=init_xs_follow)

    # evaluation function weight
    Q = np.diag([1., 1., 1.])
    R = np.diag([5.])
    pre_step = 15

    # make controller with discreted matrix
    # please check the solver, if you want to use the scipy, set the MpcController_scipy
    lead_controller = MpcController_cvxopt(Ad, Bd, W_D, Q, R, pre_step,
                               dt_input_upper=np.array([30 * dt]), dt_input_lower=np.array([-30 * dt]),
                               input_upper=np.array([30.]), input_lower=np.array([-30.]))

    follow_controller = MpcController_cvxopt(Ad, Bd, W_D, Q, R, pre_step,
                               dt_input_upper=np.array([30 * dt]), dt_input_lower=np.array([-30 * dt]),
                               input_upper=np.array([30.]), input_lower=np.array([-30.]))

    lead_controller.initialize_controller()
    follow_controller.initialize_controller()

    # reference
    lead_reference = np.array([[0., 0., 0.] for _ in range(pre_step)]).flatten()
    ref = np.array([[0.], [0.]])

    for i in range(iteration_num):
        print("simulation time = {0}".format(i))

        # make lead car's move
        if i > int(iteration_num / 3):
            ref = np.array([[0.], [4.]])

        ## lead
        # world traj
        lead_states = lead_car.xs

        # transformation
        relative_ref = coordinate_transformation_in_position(ref, lead_states[:2])
        relative_ref = coordinate_transformation_in_angle(relative_ref, lead_states[2])

        # make ref
        lead_reference = np.array([[ref[1, 0], 0., 0.] for _ in range(pre_step)]).flatten()

        alpha = math.atan2(relative_ref[1], relative_ref[0])
        R = np.linalg.norm(relative_ref) / 2 * math.sin(alpha)

        print(R)
        input()

        V = 7.0 
        delta_r = math.atan2(WHEEL_BASE, R)

        A12 = (V / WHEEL_BASE) / (math.cos(delta_r)**2)
        A22 = (1. - 1. / tau)

        Ad = np.array([[1., V,   0.], 
                    [0., 1., A12],
                    [0., 0., A22]]) * dt

        Bd = np.array([[0.], [0.], [1. / tau]]) * dt

        W_D_0 = - (V / WHEEL_BASE) * delta_r / (math.cos(delta_r)**2)

        W_D = np.array([[0.], [W_D_0], [0.]]) * dt

        # update system matrix
        lead_controller.update_system_model(Ad, Bd, W_D)

        lead_opt_u = lead_controller.calc_input(np.zeros(3), lead_reference)
        lead_opt_u = np.hstack((np.array([V]), lead_opt_u))

        
        ## follow
        # make follow car
        follow_reference = np.array([lead_states[1:] for _ in range(pre_step)]).flatten()
        follow_states = follow_car.xs
       
        follow_opt_u = follow_controller.calc_input(follow_states[1:], follow_reference)
        follow_opt_u = np.hstack((np.array([V]), follow_opt_u))
        
        lead_car.update_state(lead_opt_u, dt=dt)
        follow_car.update_state(follow_opt_u, dt=dt)

    # figures and animation
    lead_history_states = np.array(lead_car.history_xs)
    follow_history_states = np.array(follow_car.history_xs)

    time_history_fig = plt.figure()
    x_fig = time_history_fig.add_subplot(311)
    y_fig = time_history_fig.add_subplot(312)
    theta_fig = time_history_fig.add_subplot(313)

    car_traj_fig = plt.figure()
    traj_fig = car_traj_fig.add_subplot(111)
    traj_fig.set_aspect('equal')

    x_fig.plot(np.arange(0, simulation_time+0.01, dt), lead_history_states[:, 0], label="lead")
    x_fig.plot(np.arange(0, simulation_time+0.01, dt), follow_history_states[:, 0], label="follow")
    x_fig.set_xlabel("time [s]")
    x_fig.set_ylabel("x")
    x_fig.legend()

    y_fig.plot(np.arange(0, simulation_time+0.01, dt), lead_history_states[:, 1], label="lead")
    y_fig.plot(np.arange(0, simulation_time+0.01, dt), follow_history_states[:, 1], label="follow")
    y_fig.plot(np.arange(0, simulation_time+0.01, dt), [4. for _ in range(iteration_num+1)], linestyle="dashed")
    y_fig.set_xlabel("time [s]")
    y_fig.set_ylabel("y")
    y_fig.legend()

    theta_fig.plot(np.arange(0, simulation_time+0.01, dt), lead_history_states[:, 2], label="lead")
    theta_fig.plot(np.arange(0, simulation_time+0.01, dt), follow_history_states[:, 2], label="follow")
    theta_fig.plot(np.arange(0, simulation_time+0.01, dt), [0. for _ in range(iteration_num+1)], linestyle="dashed")
    theta_fig.set_xlabel("time [s]")
    theta_fig.set_ylabel("theta")
    theta_fig.legend()

    time_history_fig.tight_layout()

    traj_fig.plot(lead_history_states[:, 0], lead_history_states[:, 1], label="lead")
    traj_fig.plot(follow_history_states[:, 0], follow_history_states[:, 1], label="follow")
    traj_fig.set_xlabel("x")
    traj_fig.set_ylabel("y")
    traj_fig.legend()
    plt.show()

    lead_history_us = np.array(lead_controller.history_us)
    follow_history_us = np.array(follow_controller.history_us)
    input_history_fig = plt.figure()
    u_1_fig = input_history_fig.add_subplot(111)

    u_1_fig.plot(np.arange(0, simulation_time+0.01, dt), lead_history_us[:, 0], label="lead")
    u_1_fig.plot(np.arange(0, simulation_time+0.01, dt), follow_history_us[:, 0], label="follow")
    u_1_fig.set_xlabel("time [s]")
    u_1_fig.set_ylabel("u_omega")
    
    input_history_fig.tight_layout()
    plt.show()

    animdrawer = AnimDrawer([lead_history_states, follow_history_states])
    animdrawer.draw_anim()

if __name__ == "__main__":
    main()