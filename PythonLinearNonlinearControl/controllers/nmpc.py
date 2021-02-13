from logging import getLogger

import numpy as np
import scipy.stats as stats

from .controller import Controller
from ..envs.cost import calc_cost

logger = getLogger(__name__)


class NMPC(Controller):
    def __init__(self, config, model):
        """ Nonlinear Model Predictive Control using pure gradient algorithm
        """
        super(NMPC, self).__init__(config, model)

        # model
        self.model = model

        # get cost func
        self.state_cost_fn = config.state_cost_fn
        self.terminal_state_cost_fn = config.terminal_state_cost_fn
        self.input_cost_fn = config.input_cost_fn

        # controller parameters
        self.threshold = config.opt_config["NMPC"]["threshold"]
        self.max_iters = config.opt_config["NMPC"]["max_iters"]
        self.learning_rate = config.opt_config["NMPC"]["learning_rate"]

        # general parameters
        self.pred_len = config.PRED_LEN
        self.input_size = config.INPUT_SIZE
        self.dt = config.DT

        # initialize
        self.prev_sol = np.zeros((self.pred_len, self.input_size))

    def obtain_sol(self, curr_x, g_xs):
        """ calculate the optimal inputs
        Args:
            curr_x (numpy.ndarray): current state, shape(state_size, )
            g_xs (numpy.ndarrya): goal trajectory, shape(plan_len, state_size)
        Returns:
            opt_input (numpy.ndarray): optimal input, shape(input_size, )
        """
        sol = self.prev_sol.copy()
        count = 0

        while True:
            # shape(pred_len+1, state_size)
            pred_xs = self.model.predict_traj(curr_x, sol)
            # shape(pred_len, state_size)
            pred_lams = self.model.predict_adjoint_traj(pred_xs, sol, g_xs)

            F_hat = self.config.gradient_hamiltonian_input(
                pred_xs, pred_lams, sol, g_xs)

            if np.linalg.norm(F_hat) < self.threshold:
                break

            if count > self.max_iters:
                logger.debug(" break max iteartion at F : `{}".format(
                    np.linalg.norm(F_hat)))
                break

            sol -= self.learning_rate * F_hat
            count += 1

        # update us for next optimization
        self.prev_sol = np.concatenate(
            (sol[1:], np.zeros((1, self.input_size))), axis=0)

        return sol[0]
