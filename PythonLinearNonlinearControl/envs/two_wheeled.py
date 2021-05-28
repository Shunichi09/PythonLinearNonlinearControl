import numpy as np
from matplotlib.axes import Axes
import matplotlib.pyplot as plt

from .env import Env
from ..plotters.plot_objs import circle_with_angle, square, circle


def step_two_wheeled_env(curr_x, u, dt, method="Oylar"):
    """ step two wheeled enviroment

    Args:
        curr_x (numpy.ndarray): current state, shape(state_size, )
        u (numpy.ndarray): input, shape(input_size, )
        dt (float): sampling time
    Returns:
        next_x (numpy.ndarray): next state, shape(state_size. )

    Notes:
        TODO: deal with another method, like Runge Kutta
    """
    B = np.array([[np.cos(curr_x[-1]), 0.],
                  [np.sin(curr_x[-1]), 0.],
                  [0., 1.]])

    x_dot = np.matmul(B, u[:, np.newaxis])

    next_x = x_dot.flatten() * dt + curr_x

    return next_x


class TwoWheeledConstEnv(Env):
    """ Two wheeled robot with constant goal Env
    """

    def __init__(self):
        """
        """
        self.config = {"state_size": 3,
                       "input_size": 2,
                       "dt": 0.01,
                       "max_step": 500,
                       "input_lower_bound": (-1.5, -3.14),
                       "input_upper_bound": (1.5, 3.14),
                       "car_size": 0.2,
                       "wheel_size": (0.075, 0.015)
                       }

        super(TwoWheeledConstEnv, self).__init__(self.config)

    def reset(self, init_x=None):
        """ reset state

        Returns:
            init_x (numpy.ndarray): initial state, shape(state_size, )  
            info (dict): information
        """
        self.step_count = 0

        noise = np.clip(np.random.randn(3), -0.1, 0.1)
        noise *= 0.1
        self.curr_x = np.zeros(self.config["state_size"]) + noise

        if init_x is not None:
            self.curr_x = init_x

        # goal
        self.g_x = np.array([2.5, 2.5, 0.])

        # clear memory
        self.history_x = []
        self.history_g_x = []

        return self.curr_x, {"goal_state": self.g_x}

    def step(self, u):
        """ step environments

        Args:
            u (numpy.ndarray) : input, shape(input_size, )
        Returns:
            next_x (numpy.ndarray): next state, shape(state_size, ) 
            cost (float): costs
            done (bool): end the simulation or not
            info (dict): information 
        """
        # clip action
        u = np.clip(u,
                    self.config["input_lower_bound"],
                    self.config["input_upper_bound"])

        # step
        next_x = step_two_wheeled_env(self.curr_x, u, self.config["dt"])

        # TODO: costs
        costs = 0.
        costs += 0.1 * np.sum(u**2)
        costs += np.sum(((self.curr_x - self.g_x)**2) * np.array([5., 5., 1.]))

        # save history
        self.history_x.append(next_x.flatten())
        self.history_g_x.append(self.g_x.flatten())

        # update
        self.curr_x = next_x.flatten()
        # update costs
        self.step_count += 1

        return next_x.flatten(), costs, \
            self.step_count > self.config["max_step"], \
            {"goal_state": self.g_x}

    def plot_func(self, to_plot, i=None, history_x=None, history_g_x=None):
        """ plot cartpole object function

        Args:
            to_plot (axis or imgs): plotted objects
            i (int): frame count 
            history_x (numpy.ndarray): history of state, shape(iters, state)
            history_g_x (numpy.ndarray): history of goal state,
                                         shape(iters, state)

        Returns:
            None or imgs : imgs order is ["cart_img", "pole_img"]
        """
        if isinstance(to_plot, Axes):
            imgs = {}  # create new imgs

            imgs["car"] = to_plot.plot([], [], c="k")[0]
            imgs["car_angle"] = to_plot.plot([], [], c="k")[0]
            imgs["left_tire"] = to_plot.plot([], [], c="k", linewidth=5)[0]
            imgs["right_tire"] = to_plot.plot([], [], c="k", linewidth=5)[0]
            imgs["goal"] = to_plot.plot([], [], marker="*",
                                        c="b", markersize=10)[0]
            imgs["traj"] = to_plot.plot([], [], c="b", linestyle="dashed")[0]

            # set axis
            to_plot.set_xlim([-1., 3.])
            to_plot.set_ylim([-1., 3.])

            return imgs

        # set imgs
        # car imgs
        car_x, car_y, car_angle_x, car_angle_y, \
            left_tire_x, left_tire_y, right_tire_x, right_tire_y = \
            self._plot_car(history_x[i])

        to_plot["car"].set_data(car_x, car_y)
        to_plot["car_angle"].set_data(car_angle_x, car_angle_y)
        to_plot["left_tire"].set_data(left_tire_x, left_tire_y,)
        to_plot["right_tire"].set_data(right_tire_x, right_tire_y)

        # goal and trajs
        to_plot["goal"].set_data(history_g_x[i, 0], history_g_x[i, 1])
        to_plot["traj"].set_data(history_x[:i, 0], history_x[:i, 1])

    def _plot_car(self, curr_x):
        """ plot car fucntions
        """
        # cart
        car_x, car_y, car_angle_x, car_angle_y = \
            circle_with_angle(curr_x[0], curr_x[1],
                              self.config["car_size"], curr_x[2])

        # left tire
        center_x = (self.config["car_size"]
                    + self.config["wheel_size"][1]) \
            * np.cos(curr_x[2]-np.pi/2.) + curr_x[0]
        center_y = (self.config["car_size"]
                    + self.config["wheel_size"][1]) \
            * np.sin(curr_x[2]-np.pi/2.) + curr_x[1]

        left_tire_x, left_tire_y = \
            square(center_x, center_y,
                   self.config["wheel_size"], curr_x[2])

        # right tire
        center_x = (self.config["car_size"]
                    + self.config["wheel_size"][1]) \
            * np.cos(curr_x[2]+np.pi/2.) + curr_x[0]
        center_y = (self.config["car_size"]
                    + self.config["wheel_size"][1]) \
            * np.sin(curr_x[2]+np.pi/2.) + curr_x[1]

        right_tire_x, right_tire_y = \
            square(center_x, center_y,
                   self.config["wheel_size"], curr_x[2])

        return car_x, car_y, car_angle_x, car_angle_y,\
            left_tire_x, left_tire_y,\
            right_tire_x, right_tire_y


class TwoWheeledTrackEnv(Env):
    """ Two wheeled robot with constant goal Env
    """

    def __init__(self):
        """
        """
        self.config = {"state_size": 3,
                       "input_size": 2,
                       "dt": 0.01,
                       "max_step": 1000,
                       "input_lower_bound": (-1.5, -3.14),
                       "input_upper_bound": (1.5, 3.14),
                       "car_size": 0.2,
                       "wheel_size": (0.075, 0.015)
                       }

        super(TwoWheeledTrackEnv, self).__init__(self.config)

    @staticmethod
    def make_road(linelength=3., circle_radius=1.):
        """ make track

        Returns:
            road (numpy.ndarray): road info, shape(n_point, 3) x, y, angle
        """
        # line
        # not include start points
        line = np.linspace(-1.5, 1.5, num=51, endpoint=False)[1:]
        line_1 = np.stack((line, np.zeros(50)), axis=1)
        line_2 = np.stack((line[::-1], np.zeros(50)+circle_radius*2.), axis=1)

        # circle
        circle_1_x, circle_1_y = circle(linelength/2., circle_radius,
                                        circle_radius, start=-np.pi/2., end=np.pi/2., n_point=50)
        circle_1 = np.stack((circle_1_x, circle_1_y), axis=1)

        circle_2_x, circle_2_y = circle(-linelength/2., circle_radius,
                                        circle_radius, start=np.pi/2., end=3*np.pi/2., n_point=50)
        circle_2 = np.stack((circle_2_x, circle_2_y), axis=1)

        road_pos = np.concatenate((line_1, circle_1, line_2, circle_2), axis=0)

        # calc road angle
        road_diff = road_pos[1:] - road_pos[:-1]
        road_angle = np.arctan2(road_diff[:, 1], road_diff[:, 0])
        road_angle = np.concatenate((np.zeros(1), road_angle))

        road = np.concatenate((road_pos, road_angle[:, np.newaxis]), axis=1)

        return np.tile(road, (3, 1))

    def reset(self, init_x=None):
        """ reset state

        Returns:
            init_x (numpy.ndarray): initial state, shape(state_size, )  
            info (dict): information
        """
        self.step_count = 0

        noise = np.clip(np.random.randn(3), -0.1, 0.1)
        noise *= 0.01
        self.curr_x = np.zeros(self.config["state_size"]) + noise

        if init_x is not None:
            self.curr_x = init_x

        # goal
        self.g_traj = self.make_road()

        # clear memory
        self.history_x = []
        self.history_g_x = []

        return self.curr_x, {"goal_state": self.g_traj}

    def step(self, u):
        """ step environments

        Args:
            u (numpy.ndarray) : input, shape(input_size, )
        Returns:
            next_x (numpy.ndarray): next state, shape(state_size, ) 
            cost (float): costs
            done (bool): end the simulation or not
            info (dict): information 
        """
        # clip action
        u = np.clip(u,
                    self.config["input_lower_bound"],
                    self.config["input_upper_bound"])

        # step
        next_x = step_two_wheeled_env(self.curr_x, u, self.config["dt"])

        costs = 0.
        costs += 0.1 * np.sum(u**2)
        costs += np.min(np.linalg.norm(self.curr_x - self.g_traj, axis=1))

        # save history
        self.history_x.append(next_x.flatten())

        # update
        self.curr_x = next_x.flatten()
        # update costs
        self.step_count += 1

        return next_x.flatten(), costs, \
            self.step_count > self.config["max_step"], \
            {"goal_state": self.g_traj}

    def plot_func(self, to_plot, i=None, history_x=None, history_g_x=None):
        """ plot cartpole object function

        Args:
            to_plot (axis or imgs): plotted objects
            i (int): frame count 
            history_x (numpy.ndarray): history of state, shape(iters, state)
            history_g_x (numpy.ndarray): history of goal state,
                                         shape(iters, state)

        Returns:
            None or imgs : imgs order is ["cart_img", "pole_img"]
        """
        if isinstance(to_plot, Axes):
            imgs = {}  # create new imgs

            imgs["car"] = to_plot.plot([], [], c="k")[0]
            imgs["car_angle"] = to_plot.plot([], [], c="k")[0]
            imgs["left_tire"] = to_plot.plot([], [], c="k", linewidth=5)[0]
            imgs["right_tire"] = to_plot.plot([], [], c="k", linewidth=5)[0]
            imgs["goal"] = to_plot.plot([], [], marker="*",
                                        c="b", markersize=10)[0]
            imgs["traj"] = to_plot.plot([], [], c="b", linestyle="dashed")[0]

            # set axis
            to_plot.set_xlim([-3., 3.])
            to_plot.set_ylim([-1., 3.])

            # plot road
            to_plot.plot(self.g_traj[:, 0], self.g_traj[:, 1],
                         c="k", linestyle="dashed")

            return imgs

        # set imgs
        # car imgs
        car_x, car_y, car_angle_x, car_angle_y, \
            left_tire_x, left_tire_y, right_tire_x, right_tire_y = \
            self._plot_car(history_x[i])

        to_plot["car"].set_data(car_x, car_y)
        to_plot["car_angle"].set_data(car_angle_x, car_angle_y)
        to_plot["left_tire"].set_data(left_tire_x, left_tire_y,)
        to_plot["right_tire"].set_data(right_tire_x, right_tire_y)

        # goal and trajs
        to_plot["goal"].set_data(history_g_x[i, 0], history_g_x[i, 1])
        to_plot["traj"].set_data(history_x[:i, 0], history_x[:i, 1])

    def _plot_car(self, curr_x):
        """ plot car fucntions
        """
        # cart
        car_x, car_y, car_angle_x, car_angle_y = \
            circle_with_angle(curr_x[0], curr_x[1],
                              self.config["car_size"], curr_x[2])

        # left tire
        center_x = (self.config["car_size"]
                    + self.config["wheel_size"][1]) \
            * np.cos(curr_x[2]-np.pi/2.) + curr_x[0]
        center_y = (self.config["car_size"]
                    + self.config["wheel_size"][1]) \
            * np.sin(curr_x[2]-np.pi/2.) + curr_x[1]

        left_tire_x, left_tire_y = \
            square(center_x, center_y,
                   self.config["wheel_size"], curr_x[2])

        # right tire
        center_x = (self.config["car_size"]
                    + self.config["wheel_size"][1]) \
            * np.cos(curr_x[2]+np.pi/2.) + curr_x[0]
        center_y = (self.config["car_size"]
                    + self.config["wheel_size"][1]) \
            * np.sin(curr_x[2]+np.pi/2.) + curr_x[1]

        right_tire_x, right_tire_y = \
            square(center_x, center_y,
                   self.config["wheel_size"], curr_x[2])

        return car_x, car_y, car_angle_x, car_angle_y,\
            left_tire_x, left_tire_y,\
            right_tire_x, right_tire_y
