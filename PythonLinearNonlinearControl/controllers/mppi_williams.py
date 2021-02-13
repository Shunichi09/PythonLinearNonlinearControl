from logging import getLogger

import numpy as np
import scipy.stats as stats

from .controller import Controller
from ..envs.cost import calc_cost

logger = getLogger(__name__)


class MPPIWilliams(Controller):
    """ Model Predictive Path Integral for linear and nonlinear method

    Attributes:
        history_u (list[numpy.ndarray]): time history of optimal input
    Ref:
        G. Williams et al., "Information theoretic MPC
        for model-based reinforcement learning,"
        2017 IEEE International Conference on Robotics and Automation (ICRA),
        Singapore, 2017, pp. 1714-1721.
    """

    def __init__(self, config, model):
        super(MPPIWilliams, self).__init__(config, model)

        # model
        self.model = model

        # general parameters
        self.pred_len = config.PRED_LEN
        self.input_size = config.INPUT_SIZE

        # mppi parameters
        self.pop_size = config.opt_config["MPPIWilliams"]["popsize"]
        self.lam = config.opt_config["MPPIWilliams"]["lambda"]
        self.noise_sigma = config.opt_config["MPPIWilliams"]["noise_sigma"]
        self.opt_dim = self.input_size * self.pred_len

        # get bound
        self.input_upper_bounds = np.tile(config.INPUT_UPPER_BOUND,
                                          (self.pred_len, 1))
        self.input_lower_bounds = np.tile(config.INPUT_LOWER_BOUND,
                                          (self.pred_len, 1))

        # get cost func
        self.state_cost_fn = config.state_cost_fn
        self.terminal_state_cost_fn = config.terminal_state_cost_fn
        self.input_cost_fn = config.input_cost_fn

        # init mean
        self.prev_sol = np.tile((config.INPUT_UPPER_BOUND
                                 + config.INPUT_LOWER_BOUND) / 2.,
                                self.pred_len)
        self.prev_sol = self.prev_sol.reshape(self.pred_len, self.input_size)

        # save
        self.history_u = [np.zeros(self.input_size)]

    def clear_sol(self):
        """ clear prev sol
        """
        logger.debug("Clear Solution")
        self.prev_sol = \
            (self.input_upper_bounds + self.input_lower_bounds) / 2.
        self.prev_sol = self.prev_sol.reshape(self.pred_len, self.input_size)

    def calc_cost(self, curr_x, samples, g_xs):
        """ calculate the cost of input samples by using MPPI's eq

        Args:
            curr_x (numpy.ndarray): shape(state_size),
                current robot position
            samples (numpy.ndarray): shape(pop_size, opt_dim), 
                input samples
            g_xs (numpy.ndarray): shape(pred_len, state_size),
                goal states
        Returns:
            costs (numpy.ndarray): shape(pop_size, )
        """
        # get size
        pop_size = samples.shape[0]
        g_xs = np.tile(g_xs, (pop_size, 1, 1))

        # calc cost, pred_xs.shape = (pop_size, pred_len+1, state_size)
        pred_xs = self.model.predict_traj(curr_x, samples)

        # get particle cost
        costs = calc_cost(pred_xs, samples, g_xs,
                          self.state_cost_fn, None,
                          self.terminal_state_cost_fn)

        return costs

    def obtain_sol(self, curr_x, g_xs):
        """ calculate the optimal inputs

        Args:
            curr_x (numpy.ndarray): current state, shape(state_size, )
            g_xs (numpy.ndarrya): goal trajectory, shape(plan_len, state_size)
        Returns:
            opt_input (numpy.ndarray): optimal input, shape(input_size, )
        """
        # get noised inputs
        noise = np.random.normal(
            loc=0, scale=1.0, size=(self.pop_size, self.pred_len,
                                    self.input_size)) * self.noise_sigma

        noised_inputs = self.prev_sol + noise

        # clip actions
        noised_inputs = np.clip(
            noised_inputs, self.input_lower_bounds, self.input_upper_bounds)

        # calc cost
        costs = self.calc_cost(curr_x, noised_inputs, g_xs)

        costs += np.sum(np.sum(
            self.lam * self.prev_sol * noise / self.noise_sigma,
            axis=-1), axis=-1)

        # mppi update
        beta = np.min(costs)
        eta = np.sum(np.exp(- 1. / self.lam * (costs - beta)), axis=0) \
            + 1e-10

        # weight
        # eta.shape = (pred_len, input_size)
        weights = np.exp(- 1. / self.lam * (costs - beta)) / eta

        # update inputs
        sol = self.prev_sol \
            + np.sum(weights[:, np.newaxis, np.newaxis] * noise, axis=0)

        # update
        self.prev_sol[:-1] = sol[1:]
        self.prev_sol[-1] = sol[-1]  # last use the terminal input

        # log
        self.history_u.append(sol[0])

        return sol[0]

    def __str__(self):
        return "MPPIWilliams"
