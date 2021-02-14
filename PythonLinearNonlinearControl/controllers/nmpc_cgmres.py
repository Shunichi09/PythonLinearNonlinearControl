from logging import getLogger

import numpy as np
import scipy.stats as stats

from .controller import Controller
from ..envs.cost import calc_cost
from ..common.utils import line_search

logger = getLogger(__name__)


class NMPCCGMRES(Controller):
    def __init__(self, config, model):
        """ Nonlinear Model Predictive Control using cgmres
        """
        super(NMPCCGMRES, self).__init__(config, model)

        # model
        self.model = model

        # get cost func
        self.state_cost_fn = config.state_cost_fn
        self.terminal_state_cost_fn = config.terminal_state_cost_fn
        self.input_cost_fn = config.input_cost_fn

        # general parameters
        self.pred_len = config.PRED_LEN
        self.input_size = config.INPUT_SIZE
        self.dt = config.DT

        # controller parameters
        self.threshold = config.opt_config["NMPCCGMRES"]["threshold"]
        self.zeta = config.opt_config["NMPCCGMRES"]["zeta"]
        self.delta = config.opt_config["NMPCCGMRES"]["delta"]
        self.alpha = config.opt_config["NMPCCGMRES"]["alpha"]
        self.tf = config.opt_config["NMPCCGMRES"]["tf"]
        self.divide_num = config.PRED_LEN
        self.with_constraint = config.opt_config["NMPCCGMRES"]["constraint"]
        if not self.with_constraint:
            raise NotImplementedError
        # 3 means u, dummy_u, raw
        self.max_iters = 3 * self.input_size * self.divide_num

        # initialize
        self.prev_sol = np.zeros((self.pred_len, self.input_size))
        self.opt_count = 1
        # add smaller than constraints value
        input_constraint = np.abs([config.INPUT_LOWER_BOUND])
        self.prev_dummy_sol = np.ones(
            (self.pred_len, self.input_size)) * input_constraint - 1e-3
        # add bigger than 0.01 to avoid computational error
        self.prev_raw = np.zeros(
            (self.pred_len, self.input_size)) + 0.01 + 1e-3

    def _compute_f(self, curr_x, sol, g_xs, dummy_sol=None, raw=None):
        # shape(pred_len+1, state_size)
        pred_xs = self.model.predict_traj(curr_x, sol)
        # shape(pred_len, state_size)
        pred_lams = self.model.predict_adjoint_traj(pred_xs, sol, g_xs)

        if self.with_constraint:
            F = self.config.gradient_hamiltonian_input_with_constraint(
                pred_xs, pred_lams, sol, g_xs, dummy_sol, raw)

            return F
        else:
            raise NotImplementedError

    def obtain_sol(self, curr_x, g_xs):
        """ calculate the optimal inputs
        Args:
            curr_x (numpy.ndarray): current state, shape(state_size, )
            g_xs (numpy.ndarrya): goal trajectory, shape(plan_len, state_size)
        Returns:
            opt_input (numpy.ndarray): optimal input, shape(input_size, )
        """
        sol = self.prev_sol.copy()
        dummy_sol = self.prev_dummy_sol.copy()
        raw = self.prev_raw.copy()

        # compute delta t
        time = self.dt * self.opt_count
        dt = self.tf * (1. - np.exp(-self.alpha * time)) / \
            float(self.divide_num)
        self.model.dt = dt

        # compute Fxt
        x_dot = self.model.x_dot(curr_x, sol[0])
        dx = x_dot * self.delta
        Fxt = self._compute_f(curr_x+dx, sol, g_xs, dummy_sol, raw).flatten()

        # compute F
        F = self._compute_f(curr_x, sol, g_xs, dummy_sol, raw).flatten()
        right = - self.zeta * F - ((Fxt - F) / self.delta)

        # compute Fuxt
        du = sol * self.delta
        ddummy_u = dummy_sol * self.delta
        draw = raw * self.delta
        Fuxt = self._compute_f(curr_x+dx, sol+du, g_xs,
                               dummy_sol+ddummy_u, raw+draw).flatten()
        left = ((Fuxt - Fxt) / self.delta)

        r0 = right - left
        r0_norm = np.linalg.norm(r0)

        vs = np.zeros((self.max_iters, self.max_iters + 1))
        vs[:, 0] = r0 / r0_norm
        hs = np.zeros((self.max_iters + 1, self.max_iters + 1))
        e = np.zeros((self.max_iters + 1, 1))
        e[0] = 1.

        for i in range(self.max_iters):
            reshaped_vs = vs.reshape(
                (self.divide_num, 3, self.input_size, self.max_iters+1))
            du = reshaped_vs[:, 0, :, i] * self.delta
            ddummy_u = reshaped_vs[:, 1, :, i] * self.delta
            draw = reshaped_vs[:, 2, :, i] * self.delta

            Fuxt = self._compute_f(
                curr_x+dx, sol+du, g_xs, dummy_sol+ddummy_u, raw+draw).flatten()
            Av = ((Fuxt - Fxt) / self.delta)

            sum_Av = np.zeros(self.max_iters)

            for j in range(i + 1):
                hs[j, i] = np.dot(Av, vs[:, j])
                sum_Av = sum_Av + hs[j, i] * vs[:, j]

            v_est = Av - sum_Av

            hs[i+1, i] = np.linalg.norm(v_est)

            vs[:, i+1] = v_est / hs[i+1, i]

            inv_hs = np.linalg.pinv(hs[:i+1, :i])
            ys = np.dot(inv_hs, r0_norm * e[:i+1])

            judge_value = r0_norm * e[:i+1] - np.dot(hs[:i+1, :i], ys[:i])

            if np.linalg.norm(judge_value) < self.threshold or i == self.max_iters-1:
                update_value = np.dot(vs[:, :i-1], ys_pre[:i-1]).flatten()

                update_value = update_value.reshape(
                    (self.divide_num, 3, self.input_size))
                du_new = du + update_value[:, 0, :]
                ddummy_u_new = ddummy_u + update_value[:, 1, :]
                draw_new = draw + update_value[:, 2, :]
                break

            ys_pre = ys

        sol += du_new * self.delta
        dummy_sol += ddummy_u_new * self.delta
        raw += draw_new * self.delta

        F = self._compute_f(curr_x, sol, g_xs, dummy_sol, raw)
        logger.debug("check F = {0}".format(np.linalg.norm(F)))

        self.prev_sol = sol.copy()
        self.prev_dummy_sol = dummy_sol.copy()
        self.prev_raw = raw.copy()

        self.opt_count += 1

        return sol[0]
