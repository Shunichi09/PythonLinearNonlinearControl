from logging import getLogger

import numpy as np
import scipy.stats as stats

from .controller import Controller
from ..envs.cost import calc_cost
from ..common.utils import line_search

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
        self.optimizer_mode = config.opt_config["NMPC"]["optimizer_mode"]

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
        # use for Conjugate method
        conjugate_d = None
        conjugate_prev_d = None
        conjugate_s = None
        conjugate_beta = None

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

            if self.optimizer_mode == "conjugate":
                conjugate_d = F_hat.flatten()

                if conjugate_prev_d is None:  # initial
                    conjugate_s = conjugate_d
                    conjugate_prev_d = conjugate_d
                    F_hat = conjugate_s.reshape(F_hat.shape)
                else:
                    prev_d = np.dot(conjugate_prev_d, conjugate_prev_d)
                    d = np.dot(conjugate_d, conjugate_d - conjugate_prev_d)
                    conjugate_beta = (d + 1e-6) / (prev_d + 1e-6)

                    conjugate_s = conjugate_d + conjugate_beta * conjugate_s
                    conjugate_prev_d = conjugate_d
                    F_hat = conjugate_s.reshape(F_hat.shape)

            def compute_eval_val(u):
                pred_xs = self.model.predict_traj(curr_x, u)
                state_cost = np.sum(self.config.state_cost_fn(
                    pred_xs[1:-1], g_xs[1:-1]))
                input_cost = np.sum(self.config.input_cost_fn(u))
                terminal_cost = np.sum(
                    self.config.terminal_state_cost_fn(pred_xs[-1], g_xs[-1]))
                return state_cost + input_cost + terminal_cost

            alpha = line_search(F_hat, sol,
                                compute_eval_val, init_alpha=self.learning_rate)

            sol -= alpha * F_hat
            count += 1

        # update us for next optimization
        self.prev_sol = np.concatenate(
            (sol[1:], np.zeros((1, self.input_size))), axis=0)

        return sol[0]
