import numpy as np
import matplotlib.pyplot as plt
import math
import copy

# from mpc_func_with_cvxopt import MpcController as MpcController_cvxopt
from iterative_MPC import IterativeMpcController
from animation import AnimDrawer
# from control import matlab
from coordinate_trans import coordinate_transformation_in_angle, coordinate_transformation_in_position
from traj_func import make_sample_traj
from func_curvature import calc_curvatures

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

        self.tau = 0.01

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
        # y_dot = u_1 * math.cos(y_3 + y_4)
        y_dot = u_1 * math.cos(y_3)

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
        # y_dot = u_1 * math.sin(y_3 + y_4)
        y_dot = u_1 * math.sin(y_3)

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
        # y_dot = u_1 / self.REAR_WHEELE_BASE * math.sin(y_4)
        y_dot = u_1 * math.tan(y_4) / (self.REAR_WHEELE_BASE + self.FRONT_WHEELE_BASE)

        return y_dot

    def _func_x_4(self, y_1, y_2, y_3, y_4, u_1, u_2):
        """Ad, Bd, W_D, Q, R
        ParAd, Bd, W_D, Q, R
        ---Ad, Bd, W_D, Q, R
        y_1 : float
        y_2 : float
        y_3 : float
        u_1 : float
            system input
        u_2 : float
            system input
        """
        # y_dot = math.atan2(self.REAR_WHEELE_BASE * math.tan(u_2) ,self.REAR_WHEELE_BASE + self.FRONT_WHEELE_BASE)
        y_dot = - 1. / self.tau * (y_4 - u_2)

        return y_dot

class SystemModel():
    """
    Attributes
    -----------
    WHEEL_BASE : float
        wheel base of the car
    Ad_s : list
        list of system model matrix Ad
    Bd_s : list
        list of system model matrix Bd
    W_D_s : list
        list of system model matrix W_D_s
    Q : numpy.ndarray
    R : numpy.ndarray
    """
    def __init__(self, tau = 0.01, dt = 0.01):
        """
        Parameters
        -----------
        tau : time constant, optional
        dt : sampling time, optional
        """
        self.dt = dt
        self.tau = tau
        self.WHEEL_BASE = 2.2

        self.Ad_s = []
        self.Bd_s = []
        self.W_D_s = []

    def calc_predict_sytem_model(self, V, curvatures, predict_step):
        """
        calc next predict systemo models
        V : float
        curvatures : list
            this is the next curvature's list
        predict_step : int
            predict step of MPC
        """
        for i in range(predict_step):
            delta_r = math.atan2(self.WHEEL_BASE, 1. / curvatures[i])

            A12 = (V / self.WHEEL_BASE) / (math.cos(delta_r)**2)
            A22 = (1. - 1. / self.tau * self.dt)

            Ad = np.array([[1., V * self.dt,   0.], 
                           [0., 1., A12 * self.dt],
                           [0., 0., A22]])

            Bd = np.array([[0.], [0.], [1. / self.tau]]) * self.dt

            # -v*curvature + v/L*(tan(delta_r)-delta_r*cos_delta_r_squared_inv);
            # W_D_0 = V / self.WHEEL_BASE * (delta_r / (math.cos(delta_r)**2)
            W_D_0 = -V * curvatures[i] + (V / self.WHEEL_BASE) * (math.tan(delta_r) - delta_r / (math.cos(delta_r)**2))

            W_D = np.array([[0.], [W_D_0], [0.]]) * self.dt

            self.Ad_s.append(Ad)
            self.Bd_s.append(Bd)
            self.W_D_s.append(W_D)

        # return self.Ad_s, self.Bd_s, self.W_D_s

def search_nearest_point(points, base_point):
    """
    Parameters
    -----------
    points : numpy.ndarray, shape is (2, N)
    base_point : numpy.ndarray, shape is (2, 1)

    Returns
    -------
    nearest_index : 
    nearest_point : 
    """
    distance_mat = np.sqrt(np.sum((points - base_point)**2, axis=0))

    index_min = np.argmin(distance_mat)

    return index_min, points[:, index_min]


def main():
    # parameters
    dt = 0.01
    simulation_time = 20 # in seconds
    PREDICT_STEP = 20
    iteration_num = int(simulation_time / dt)

    # make simulator with coninuous matrix
    init_xs_lead = np.array([0., 0., math.pi/4, 0.])
    init_xs_follow = np.array([0., 0., math.pi/4, 0.])
    lead_car = WheeledSystem(init_states=init_xs_lead)
    follow_car = WheeledSystem(init_states=init_xs_follow)

    # make system model
    lead_car_system_model = SystemModel()
    follow_car_system_model = SystemModel()

    # reference
    traj_ref_xs, traj_ref_ys = make_sample_traj(int(simulation_time/dt))
    traj_ref = np.array([traj_ref_xs, traj_ref_ys])

    # nearest point
    index_min, nearest_point = search_nearest_point(traj_ref, lead_car.xs[:2].reshape(2, 1))

    # get traj's curvature
    NUM_SKIP = 5
    angles, curvatures = calc_curvatures(traj_ref[:, index_min:index_min + PREDICT_STEP + 2 * NUM_SKIP], PREDICT_STEP, NUM_SKIP)

    # evaluation function weight
    Q = np.diag([100., 1., 1.])
    R = np.diag([0.01])

    # System model update
    V = 4.0 # in pratical we should calc from the state
    lead_car_system_model.calc_predict_sytem_model(V, curvatures, PREDICT_STEP)
    follow_car_system_model.calc_predict_sytem_model(V, curvatures, PREDICT_STEP)

    # make controller with discreted matrix
    lead_controller = IterativeMpcController(lead_car_system_model, Q, R, PREDICT_STEP,
                               dt_input_upper=np.array([30 * dt]), dt_input_lower=np.array([-30 * dt]),
                               input_upper=np.array([30.]), input_lower=np.array([-30.]))

    follow_controller = IterativeMpcController(follow_car_system_model, Q, R, PREDICT_STEP,
                               dt_input_upper=np.array([30 * dt]), dt_input_lower=np.array([-30 * dt]),
                               input_upper=np.array([30.]), input_lower=np.array([-30.]))

    # initialize
    lead_controller.initialize_controller()
    follow_controller.initialize_controller()
    
    for i in range(iteration_num):
        print("simulation time = {0}".format(i))

        ## lead
        # world traj
        lead_states = lead_car.xs

        # nearest point
        index_min, nearest_point = search_nearest_point(traj_ref, lead_car.xs[:2].reshape(2, 1))
        # end check
        if len(traj_ref_ys) <= index_min + PREDICT_STEP + 2 * NUM_SKIP:
            print("break")
            break            

        # get traj's curvature
        angles, curvatures = calc_curvatures(traj_ref[:, index_min:index_min + PREDICT_STEP + 2 * NUM_SKIP], PREDICT_STEP, NUM_SKIP)

        # System model update
        V = 4.0 # in pratical we should calc from the state
        lead_car_system_model.calc_predict_sytem_model(V, curvatures, PREDICT_STEP)

        # transformation
        # car
        relative_car_position = coordinate_transformation_in_position(lead_states[:2].reshape(2, 1), nearest_point)
        relative_car_position = coordinate_transformation_in_angle(relative_car_position, angles[0])

        relative_car_angle = lead_states[2] - angles[0]
        relative_car_state = np.hstack((relative_car_position[1], relative_car_angle, lead_states[-1]))

        # traj_ref
        relative_traj = coordinate_transformation_in_position(traj_ref[:, index_min:index_min + PREDICT_STEP], nearest_point)
        relative_traj = coordinate_transformation_in_angle(relative_traj, angles[0])
        relative_ref_angle = np.array(angles) - angles[0]

        # make ref
        lead_reference = np.array([[relative_traj[1, -1], relative_ref_angle[i], 0.] for i in range(PREDICT_STEP)]).flatten()

        print("relative car state = {}".format(relative_car_state))
        print("nearest point = {}".format(nearest_point))
        # input()

        # update system matrix
        lead_controller.update_system_model(lead_car_system_model)

        lead_opt_u = lead_controller.calc_input(relative_car_state, lead_reference)

        lead_opt_u = np.hstack((np.array([V]), lead_opt_u))

        print("opt_u = {}".format(lead_opt_u))
        # input()
        
        lead_car.update_state(lead_opt_u, dt=dt)
        follow_car.update_state(lead_opt_u, dt=dt)

    # figures and animation
    lead_history_states = np.array(lead_car.history_xs)
    follow_history_states = np.array(follow_car.history_xs)

    """
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
    """

    animdrawer = AnimDrawer([lead_history_states, follow_history_states, traj_ref])
    animdrawer.draw_anim()

if __name__ == "__main__":
    main()