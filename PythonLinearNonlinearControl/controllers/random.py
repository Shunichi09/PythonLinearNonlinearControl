from logging import getLogger

import numpy as np
import scipy.stats as stats

from .controller import Controller
from ..envs.cost import calc_cost

logger = getLogger(__name__)


class RandomShooting(Controller):
    """ Random Shooting Method for linear and nonlinear method

    Attributes:
        history_u (list[numpy.ndarray]): time history of optimal input
    Ref:
        Chua, K., Calandra, R., McAllister, R., & Levine, S. (2018).
        Deep reinforcement learning in a handful of trials 
        using probabilistic dynamics models.
        In Advances in Neural Information Processing Systems (pp. 4754-4765).
    """

    def __init__(self, config, model):
        super(RandomShooting, self).__init__(config, model)

        # model
        self.model = model

        # general parameters
        self.pred_len = config.PRED_LEN
        self.input_size = config.INPUT_SIZE

        # cem parameters
        self.pop_size = config.opt_config["Random"]["popsize"]
        self.opt_dim = self.input_size * self.pred_len

        # get bound
        self.input_upper_bounds = np.tile(config.INPUT_UPPER_BOUND,
                                          self.pred_len)
        self.input_lower_bounds = np.tile(config.INPUT_LOWER_BOUND,
                                          self.pred_len)

        # get cost func
        self.state_cost_fn = config.state_cost_fn
        self.terminal_state_cost_fn = config.terminal_state_cost_fn
        self.input_cost_fn = config.input_cost_fn

        # save
        self.history_u = []

    def obtain_sol(self, curr_x, g_xs):
        """ calculate the optimal inputs

        Args:
            curr_x (numpy.ndarray): current state, shape(state_size, )
            g_xs (numpy.ndarrya): goal trajectory, shape(plan_len, state_size)
        Returns:
            opt_input (numpy.ndarray): optimal input, shape(input_size, )
        """
        # set different seed
        np.random.seed()

        samples = np.random.uniform(self.input_lower_bounds,
                                    self.input_upper_bounds,
                                    [self.pop_size, self.opt_dim])
        # calc cost
        costs = self.calc_cost(curr_x,
                               samples.reshape(self.pop_size,
                                               self.pred_len,
                                               self.input_size),
                               g_xs)
        # solution
        sol = samples[np.argmin(costs)]

        return sol.reshape(self.pred_len, self.input_size).copy()[0]

    def __str__(self):
        return "RandomShooting"
