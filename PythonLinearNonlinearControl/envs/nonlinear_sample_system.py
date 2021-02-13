import numpy as np
import scipy
from scipy import integrate
from .env import Env
from ..common.utils import update_state_with_Runge_Kutta


class NonlinearSampleSystemEnv(Env):
    """ Nonlinear Sample Env
    """

    def __init__(self):
        """
        """
        self.config = {"state_size": 2,
                       "input_size": 1,
                       "dt": 0.01,
                       "max_step": 2000,
                       "input_lower_bound": [-0.5],
                       "input_upper_bound": [0.5],
                       }

        super(NonlinearSampleSystemEnv, self).__init__(self.config)

    def reset(self, init_x=np.array([2., 0.])):
        """ reset state
        Returns:
            init_x (numpy.ndarray): initial state, shape(state_size, )  
            info (dict): information
        """
        self.step_count = 0

        self.curr_x = np.zeros(self.config["state_size"])

        if init_x is not None:
            self.curr_x = init_x

        # goal
        self.g_x = np.array([0., 0.])

        # clear memory
        self.history_x = []
        self.history_g_x = []

        return self.curr_x, {"goal_state": self.g_x}

    def step(self, u):
        """
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

        functions = [self._func_x_1, self._func_x_2]

        next_x = update_state_with_Runge_Kutta(self.curr_x, u,
                                               functions, self.config["dt"],
                                               batch=False)

        # cost
        cost = 0
        cost = np.sum(u**2)
        cost += np.sum((self.curr_x - self.g_x)**2)

        # save history
        self.history_x.append(next_x.flatten())
        self.history_g_x.append(self.g_x.flatten())

        # update
        self.curr_x = next_x.flatten()
        # update costs
        self.step_count += 1

        return next_x.flatten(), cost, \
            self.step_count > self.config["max_step"], \
            {"goal_state": self.g_x}

    def _func_x_1(self, x, u):
        x_dot = x[1]
        return x_dot

    def _func_x_2(self, x, u):
        x_dot = (1. - x[0]**2 - x[1]**2) * x[1] - x[0] + u[0]
        return x_dot

    def plot_func(self, to_plot, i=None, history_x=None, history_g_x=None):
        """
        """
        raise ValueError("NonlinearSampleSystemEnv does not have animation")
