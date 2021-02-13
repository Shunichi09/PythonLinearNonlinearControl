from logging import getLogger

import numpy as np
import scipy.stats as stats

from .controller import Controller
from ..envs.cost import calc_cost

logger = getLogger(__name__)


class MPPI(Controller):
    """ Model Predictive Path Integral for linear and nonlinear method

    Attributes:
        history_u (list[numpy.ndarray]): time history of optimal input
    Ref:
        Nagabandi, A., Konoglie, K., Levine, S., & Kumar, V. (2019). 
        Deep Dynamics Models for Learning Dexterous Manipulation.
        arXiv preprint arXiv:1909.11652.
    """

    def __init__(self, config, model):
        super(MPPI, self).__init__(config, model)

        # model
        self.model = model

        # general parameters
        self.pred_len = config.PRED_LEN
        self.input_size = config.INPUT_SIZE

        # mppi parameters
        self.beta = config.opt_config["MPPI"]["beta"]
        self.pop_size = config.opt_config["MPPI"]["popsize"]
        self.kappa = config.opt_config["MPPI"]["kappa"]
        self.noise_sigma = config.opt_config["MPPI"]["noise_sigma"]
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
        noised_inputs = noise.copy()

        for t in range(self.pred_len):
            if t > 0:
                noised_inputs[:, t, :] = self.beta \
                    * (self.prev_sol[t, :]
                       + noise[:, t, :]) \
                    + (1 - self.beta) \
                    * noised_inputs[:, t-1, :]
            else:
                noised_inputs[:, t, :] = self.beta \
                    * (self.prev_sol[t, :]
                       + noise[:, t, :]) \
                    + (1 - self.beta) \
                    * self.history_u[-1]

        # clip actions
        noised_inputs = np.clip(
            noised_inputs, self.input_lower_bounds, self.input_upper_bounds)

        # calc cost
        costs = self.calc_cost(curr_x, noised_inputs, g_xs)
        rewards = -costs

        # mppi update
        # normalize and get sum of reward
        # exp_rewards.shape = (N, )
        exp_rewards = np.exp(self.kappa * (rewards - np.max(rewards)))
        denom = np.sum(exp_rewards) + 1e-10  # avoid numeric error

        # weight actions
        weighted_inputs = exp_rewards[:, np.newaxis, np.newaxis] \
            * noised_inputs
        sol = np.sum(weighted_inputs, 0) / denom

        # update
        self.prev_sol[:-1] = sol[1:]
        self.prev_sol[-1] = sol[-1]  # last use the terminal input

        # log
        self.history_u.append(sol[0])

        return sol[0]

    def __str__(self):
        return "MPPI"
