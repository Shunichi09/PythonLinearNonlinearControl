from logging import getLogger

import numpy as np
import scipy.stats as stats

from .controller import Controller
from ..envs.cost import calc_cost

logger = getLogger(__name__)


class CEM(Controller):
    """ Cross Entropy Method for linear and nonlinear method

    Attributes:
        history_u (list[numpy.ndarray]): time history of optimal input
    Ref:
        Chua, K., Calandra, R., McAllister, R., & Levine, S. (2018).
        Deep reinforcement learning in a handful of trials 
        using probabilistic dynamics models.
        In Advances in Neural Information Processing Systems (pp. 4754-4765).
    """

    def __init__(self, config, model):
        super(CEM, self).__init__(config, model)

        # model
        self.model = model

        # general parameters
        self.pred_len = config.PRED_LEN
        self.input_size = config.INPUT_SIZE

        # cem parameters
        self.alpha = config.opt_config["CEM"]["alpha"]
        self.pop_size = config.opt_config["CEM"]["popsize"]
        self.max_iters = config.opt_config["CEM"]["max_iters"]
        self.num_elites = config.opt_config["CEM"]["num_elites"]
        self.epsilon = config.opt_config["CEM"]["threshold"]
        self.init_var = config.opt_config["CEM"]["init_var"]
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

        # init mean
        self.init_mean = np.tile((config.INPUT_UPPER_BOUND
                                  + config.INPUT_LOWER_BOUND) / 2.,
                                 self.pred_len)
        self.prev_sol = self.init_mean.copy()
        # init variance
        var = np.ones_like(config.INPUT_UPPER_BOUND) \
            * config.opt_config["CEM"]["init_var"]
        self.init_var = np.tile(var, self.pred_len)

        # save
        self.history_u = []

    def clear_sol(self):
        """ clear prev sol
        """
        logger.debug("Clear Sol")
        self.prev_sol = self.init_mean.copy()

    def obtain_sol(self, curr_x, g_xs):
        """ calculate the optimal inputs

        Args:
            curr_x (numpy.ndarray): current state, shape(state_size, )
            g_xs (numpy.ndarrya): goal trajectory, shape(plan_len, state_size)
        Returns:
            opt_input (numpy.ndarray): optimal input, shape(input_size, )
        """
        # initialize
        opt_count = 0

        # get configuration
        mean = self.prev_sol.flatten().copy()
        var = self.init_var.flatten().copy()

        # make distribution
        X = stats.truncnorm(-1, 1,
                            loc=np.zeros_like(mean),
                            scale=np.ones_like(mean))

        while (opt_count < self.max_iters) and np.max(var) > self.epsilon:
            # constrained
            lb_dist = mean - self.input_lower_bounds
            ub_dist = self.input_upper_bounds - mean
            constrained_var = np.minimum(np.minimum(np.square(lb_dist),
                                                    np.square(ub_dist)),
                                         var)

            # sample
            samples = X.rvs(size=[self.pop_size, self.opt_dim]) \
                * np.sqrt(constrained_var) \
                + mean

            # calc cost
            # samples.shape = (pop_size, opt_dim)
            costs = self.calc_cost(curr_x,
                                   samples.reshape(self.pop_size,
                                                   self.pred_len,
                                                   self.input_size),
                                   g_xs)

            # sort cost
            elites = samples[np.argsort(costs)][:self.num_elites]

            # new mean
            new_mean = np.mean(elites, axis=0)
            new_var = np.var(elites, axis=0)

            # soft update
            mean = self.alpha * mean + (1. - self.alpha) * new_mean
            var = self.alpha * var + (1. - self.alpha) * new_var

            logger.debug("Var = {}".format(np.max(var)))
            logger.debug("Costs = {}".format(np.mean(costs)))
            opt_count += 1

        sol = mean.copy()
        self.prev_sol = np.concatenate((mean[self.input_size:],
                                        np.zeros(self.input_size)))

        return sol.reshape(self.pred_len, self.input_size).copy()[0]

    def __str__(self):
        return "CEM"
