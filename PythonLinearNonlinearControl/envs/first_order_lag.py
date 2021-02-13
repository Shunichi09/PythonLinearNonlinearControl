import numpy as np
import scipy
from scipy import integrate
from .env import Env


class FirstOrderLagEnv(Env):
    """ First Order Lag System Env
    """

    def __init__(self, tau=0.63):
        """
        """
        self.config = {"state_size": 4,
                       "input_size": 2,
                       "dt": 0.05,
                       "max_step": 500,
                       "input_lower_bound": [-0.5, -0.5],
                       "input_upper_bound": [0.5, 0.5],
                       }

        super(FirstOrderLagEnv, self).__init__(self.config)

        # to get discrete system matrix
        self.A, self.B = self._to_state_space(tau, dt=self.config["dt"])

    @staticmethod
    def _to_state_space(tau, dt=0.05):
        """
        Args:
            tau (float): time constant
            dt (float): discrte time
        Returns:
            A (numpy.ndarray): discrete A matrix 
            B (numpy.ndarray): discrete B matrix 
        """
        # continuous
        Ac = np.array([[-1./tau, 0., 0., 0.],
                       [0., -1./tau, 0., 0.],
                       [1., 0., 0., 0.],
                       [0., 1., 0., 0.]])
        Bc = np.array([[1./tau, 0.],
                       [0., 1./tau],
                       [0., 0.],
                       [0., 0.]])
        # to discrete system
        A = scipy.linalg.expm(dt*Ac)
        # B = np.matmul(np.matmul(scipy.linalg.expm(Ac*dt) -
        #                         scipy.linalg.expm(Ac*0.), np.linalg.inv(Ac)),\
        #               Bc)
        B = np.zeros_like(Bc)
        for m in range(Bc.shape[0]):
            for n in range(Bc.shape[1]):
                integrate_fn =\
                    lambda tau: np.matmul(scipy.linalg.expm(Ac*tau), Bc)[m, n]
                sol = integrate.quad(integrate_fn, 0, dt)
                B[m, n] = sol[0]

        return A, B

    def reset(self, init_x=None):
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
        self.g_x = np.array([0., 0, -2., 3.])

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

        next_x = np.matmul(self.A, self.curr_x[:, np.newaxis]) \
            + np.matmul(self.B, u[:, np.newaxis])

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

    def plot_func(self, to_plot, i=None, history_x=None, history_g_x=None):
        """
        """
        raise ValueError("FirstOrderLag does not have animation")
