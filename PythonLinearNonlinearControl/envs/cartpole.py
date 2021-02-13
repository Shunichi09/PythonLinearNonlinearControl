import numpy as np
from matplotlib.axes import Axes

from .env import Env
from ..plotters.plot_objs import square


class CartPoleEnv(Env):
    """ Cartpole Environment

    Ref : 
        https://ocw.mit.edu/courses/
        electrical-engineering-and-computer-science/
        6-832-underactuated-robotics-spring-2009/readings/
        MIT6_832s09_read_ch03.pdf
    """

    def __init__(self):
        """
        """
        self.config = {"state_size": 4,
                       "input_size": 1,
                       "dt": 0.02,
                       "max_step": 500,
                       "input_lower_bound": [-3.],
                       "input_upper_bound": [3.],
                       "mp": 0.2,
                       "mc": 1.,
                       "l": 0.5,
                       "g": 9.81,
                       "cart_size": (0.15, 0.1),
                       }

        super(CartPoleEnv, self).__init__(self.config)

    def reset(self, init_x=None):
        """ reset state

        Returns:
            init_x (numpy.ndarray): initial state, shape(state_size, )  
            info (dict): information
        """
        self.step_count = 0

        theta = np.random.randn(1)
        self.curr_x = np.array([0., 0., theta[0], 0.])

        if init_x is not None:
            self.curr_x = init_x

        # goal
        self.g_x = np.array([0., 0., -np.pi, 0.])

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
        if self.config["input_lower_bound"] is not None:
            u = np.clip(u,
                        self.config["input_lower_bound"],
                        self.config["input_upper_bound"])

        # step
        # x
        d_x0 = self.curr_x[1]
        # v_x
        d_x1 = (u[0] + self.config["mp"] * np.sin(self.curr_x[2])
                * (self.config["l"] * (self.curr_x[3]**2)
                   + self.config["g"] * np.cos(self.curr_x[2]))) \
            / (self.config["mc"] + self.config["mp"]
               * (np.sin(self.curr_x[2])**2))
        # theta
        d_x2 = self.curr_x[3]

        # v_theta
        d_x3 = (-u[0] * np.cos(self.curr_x[2])
                - self.config["mp"] * self.config["l"] * (self.curr_x[3]**2)
                * np.cos(self.curr_x[2]) * np.sin(self.curr_x[2])
                - (self.config["mc"] + self.config["mp"]) * self.config["g"]
                * np.sin(self.curr_x[2])) \
            / (self.config["l"] * (self.config["mc"] + self.config["mp"]
                                   * (np.sin(self.curr_x[2])**2)))

        next_x = self.curr_x +\
            np.array([d_x0, d_x1, d_x2, d_x3]) * self.config["dt"]

        # TODO: costs
        costs = 0.
        costs += 0.1 * np.sum(u**2)
        costs += 6. * self.curr_x[0]**2 \
            + 12. * (np.cos(self.curr_x[2]) + 1.)**2 \
            + 0.1 * self.curr_x[1]**2 \
            + 0.1 * self.curr_x[3]**2

        # save history
        self.history_x.append(next_x.flatten())
        self.history_g_x.append(self.g_x.flatten())

        # update
        self.curr_x = next_x.flatten().copy()
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

            imgs["cart"] = to_plot.plot([], [], c="k")[0]
            imgs["pole"] = to_plot.plot([], [], c="k", linewidth=5)[0]
            imgs["center"] = to_plot.plot([], [], marker="o", c="k",
                                          markersize=10)[0]
            # centerline
            to_plot.plot(np.linspace(-1., 1., num=50), np.zeros(50),
                         c="k", linestyle="dashed")

            # set axis
            to_plot.set_xlim([-1., 1.])
            to_plot.set_ylim([-0.55, 1.5])

            return imgs

        # set imgs
        cart_x, cart_y, pole_x, pole_y = \
            self._plot_cartpole(history_x[i])

        to_plot["cart"].set_data(cart_x, cart_y)
        to_plot["pole"].set_data(pole_x, pole_y)
        to_plot["center"].set_data(history_x[i][0], 0.)

    def _plot_cartpole(self, curr_x):
        """ plot cartpole fucntions

        Args:
            curr_x (numpy.ndarray): current catpole state
        Returns:
            cart_x (numpy.ndarray): x data of cart
            cart_y (numpy.ndarray): y data of cart 
            pole_x (numpy.ndarray): x data of pole 
            pole_y (numpy.ndarray): y data of pole 
        """
        # cart
        cart_x, cart_y = square(curr_x[0], 0.,
                                self.config["cart_size"], 0.)

        # pole
        pole_x = np.array([curr_x[0], curr_x[0] + self.config["l"]
                           * np.cos(curr_x[2]-np.pi/2)])
        pole_y = np.array([0., self.config["l"]
                           * np.sin(curr_x[2]-np.pi/2)])

        return cart_x, cart_y, pole_x, pole_y
