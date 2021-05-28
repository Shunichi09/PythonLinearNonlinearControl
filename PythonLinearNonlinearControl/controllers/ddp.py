from logging import getLogger

import numpy as np
import scipy.stats as stats

from .controller import Controller
from ..envs.cost import calc_cost

logger = getLogger(__name__)


class DDP(Controller):
    """ Differential Dynamic Programming

    Ref:
        Tassa, Y., Erez, T., & Todorov, E. (2012). 
        In 2012 IEEE/RSJ International Conference on
        Intelligent Robots and Systems (pp. 4906-4913). and Study Wolf,
        https://github.com/studywolf/control, and
        https://github.com/anassinator/ilqr
    """

    def __init__(self, config, model):
        """
        """
        super(DDP, self).__init__(config, model)

        # model
        self.model = model

        # get cost func
        self.state_cost_fn = config.state_cost_fn
        self.terminal_state_cost_fn = config.terminal_state_cost_fn
        self.input_cost_fn = config.input_cost_fn
        self.gradient_cost_fn_state = config.gradient_cost_fn_state
        self.gradient_cost_fn_input = config.gradient_cost_fn_input
        self.hessian_cost_fn_state = config.hessian_cost_fn_state
        self.hessian_cost_fn_input = config.hessian_cost_fn_input
        self.hessian_cost_fn_input_state = \
            config.hessian_cost_fn_input_state

        # controller parameters
        self.max_iters = config.opt_config["DDP"]["max_iters"]
        self.init_mu = config.opt_config["DDP"]["init_mu"]
        self.mu = self.init_mu
        self.mu_min = config.opt_config["DDP"]["mu_min"]
        self.mu_max = config.opt_config["DDP"]["mu_max"]
        self.init_delta = config.opt_config["DDP"]["init_delta"]
        self.delta = self.init_delta
        self.threshold = config.opt_config["DDP"]["threshold"]

        # general parameters
        self.pred_len = config.PRED_LEN
        self.input_size = config.INPUT_SIZE
        self.dt = config.DT

        # cost parameters
        self.Q = config.Q
        self.R = config.R
        self.Sf = config.Sf

        # initialize
        self.prev_sol = np.zeros((self.pred_len, self.input_size))

    def clear_sol(self):
        """ clear prev sol
        """
        logger.debug("Clear Sol")
        self.prev_sol = np.zeros((self.pred_len, self.input_size))

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
        sol = self.prev_sol.copy()
        converged_sol = False
        update_sol = True
        self.mu = self.init_mu
        self.delta = self.init_delta

        # line search param
        alphas = 1.1**(-np.arange(10)**2)

        while opt_count < self.max_iters:
            accepted_sol = False

            # forward
            if update_sol == True:
                pred_xs, cost, f_x, f_u, f_xx, f_ux, f_uu,\
                    l_x, l_xx, l_u, l_uu, l_ux = \
                    self.forward(curr_x, g_xs, sol)
                update_sol = False

            try:
                # backward
                k, K = self.backward(f_x, f_u, f_xx, f_ux, f_uu,
                                     l_x, l_xx, l_u, l_uu, l_ux)

                # line search
                for alpha in alphas:
                    new_pred_xs, new_sol = \
                        self.calc_input(k, K, pred_xs, sol, alpha)

                    new_cost = calc_cost(new_pred_xs[np.newaxis, :, :],
                                         new_sol[np.newaxis, :, :],
                                         g_xs[np.newaxis, :, :],
                                         self.state_cost_fn,
                                         self.input_cost_fn,
                                         self.terminal_state_cost_fn)

                    if new_cost < cost:
                        if np.abs((cost - new_cost) / cost) < self.threshold:
                            converged_sol = True

                        cost = new_cost
                        pred_xs = new_pred_xs
                        sol = new_sol
                        update_sol = True

                        # decrease regularization term
                        self.delta = min(1.0, self.delta) / self.init_delta
                        self.mu *= self.delta
                        if self.mu <= self.mu_min:
                            self.mu = 0.0

                        # accept the solution
                        accepted_sol = True
                        break

            except np.linalg.LinAlgError as e:
                logger.debug("Non ans : {}".format(e))

            if not accepted_sol:
                # increase regularization term.
                self.delta = max(1.0, self.delta) * self.init_delta
                self.mu = max(self.mu_min, self.mu * self.delta)
                logger.debug("Update regularization term to {}"
                             .format(self.mu))
                if self.mu >= self.mu_max:
                    logger.debug("Reach Max regularization term")
                    break

            if converged_sol:
                logger.debug("Get converged sol")
                break

            opt_count += 1

        # update prev sol
        self.prev_sol[:-1] = sol[1:]
        self.prev_sol[-1] = sol[-1]  # last use the terminal input

        return sol[0]

    def calc_input(self, k, K, pred_xs, sol, alpha):
        """ calc input trajectory by using k and K

        Args:
            k (numpy.ndarray): gain, shape(pred_len, input_size)
            K (numpy.ndarray): gain, shape(pred_len, input_size, state_size)
            pred_xs (numpy.ndarray): predicted state,
                shape(pred_len+1, state_size)
            sol (numpy.ndarray): input trajectory, previous solutions
                shape(pred_len, input_size)
            alpha (float): param of line search
        Returns:
            new_pred_xs (numpy.ndarray): update state trajectory,
                shape(pred_len+1, state_size)
            new_sol (numpy.ndarray): update input trajectory,
                shape(pred_len, input_size)
        """
        # get size
        (pred_len, input_size, state_size) = K.shape
        # initialize
        new_pred_xs = np.zeros((pred_len+1, state_size))
        new_pred_xs[0] = pred_xs[0].copy()  # init state is same
        new_sol = np.zeros((pred_len, input_size))

        for t in range(pred_len):
            new_sol[t] = sol[t] \
                + alpha * k[t] \
                + np.dot(K[t], (new_pred_xs[t] - pred_xs[t]))
            new_pred_xs[t+1] = self.model.predict_next_state(new_pred_xs[t],
                                                             new_sol[t])

        return new_pred_xs, new_sol

    def forward(self, curr_x, g_xs, sol):
        """ forward step of iLQR

        Args:
            curr_x (numpy.ndarray): current state, shape(state_size, )
            g_xs (numpy.ndarrya): goal trajectory, shape(plan_len, state_size)
            sol (numpy.ndarray): solutions, shape(plan_len, input_size)
        Returns:
            f_x (numpy.ndarray): gradient of model with respecto to state,
                shape(pred_len, state_size, state_size)
            f_u (numpy.ndarray): gradient of model with respecto to input,
                shape(pred_len, state_size, input_size)
            f_xx (numpy.ndarray): gradient of model with respecto to state,
                shape(pred_len+1, state_size, state_size, state_size)
            f_ux (numpy.ndarray): gradient of model with respecto to input,
                shape(pred_len, state_size, input_size, state_size)
            f_uu (numpy.ndarray): gradient of model with respecto to input,
                shape(pred_len, state_size, input_size, input_size)
            l_x (numpy.ndarray): gradient of cost with respecto to state,
                shape(pred_len+1, state_size)
            l_u (numpy.ndarray): gradient of cost with respecto to input,
                shape(pred_len, input_size)
            l_xx (numpy.ndarray): hessian of cost with respecto to state,
                shape(pred_len+1, state_size, state_size)
            l_uu (numpy.ndarray): hessian of cost with respecto to input,
                shape(pred_len+1, input_size, input_size)
            l_ux (numpy.ndarray): hessian of cost with respect
                to state and input, shape(pred_len, input_size, state_size)
        """
        # simulate forward using the current control trajectory
        pred_xs = self.model.predict_traj(curr_x, sol)
        # check costs
        cost = self.calc_cost(curr_x,
                              sol[np.newaxis, :, :],
                              g_xs)

        # calc gradinet in batch
        f_x = self.model.calc_f_x(pred_xs[:-1], sol, self.dt)
        f_u = self.model.calc_f_u(pred_xs[:-1], sol, self.dt)
        # calc hessian in batch
        f_xx = self.model.calc_f_xx(pred_xs[:-1], sol, self.dt)
        f_ux = self.model.calc_f_ux(pred_xs[:-1], sol, self.dt)
        f_uu = self.model.calc_f_uu(pred_xs[:-1], sol, self.dt)

        # gradint of costs
        l_x, l_xx, l_u, l_uu, l_ux = \
            self._calc_gradient_hessian_cost(pred_xs, g_xs, sol)

        return pred_xs, cost, f_x, f_u, f_xx, f_ux, f_uu, \
            l_x, l_xx, l_u, l_uu, l_ux

    def _calc_gradient_hessian_cost(self, pred_xs, g_x, sol):
        """ calculate gradient and hessian of model and cost fn

        Args:
            pred_xs (numpy.ndarray): predict traj,
                shape(pred_len+1, state_size)
            sol (numpy.ndarray): input traj,
                shape(pred_len, input_size)
        Returns
            l_x (numpy.ndarray): gradient of cost,
                shape(pred_len+1, state_size)
            l_u (numpy.ndarray): gradient of cost,
                shape(pred_len, input_size)
            l_xx (numpy.ndarray): hessian of cost,
                shape(pred_len+1, state_size, state_size)
            l_uu (numpy.ndarray): hessian of cost,
                shape(pred_len+1, input_size, input_size)
            l_ux (numpy.ndarray): hessian of cost, 
                shape(pred_len, input_size, state_size)
        """
        # l_x.shape = (pred_len+1, state_size)
        l_x = self.gradient_cost_fn_state(pred_xs[:-1],
                                          g_x[:-1], terminal=False)
        terminal_l_x = \
            self.gradient_cost_fn_state(pred_xs[-1],
                                        g_x[-1], terminal=True)

        l_x = np.concatenate((l_x, terminal_l_x), axis=0)

        # l_u.shape = (pred_len, input_size)
        l_u = self.gradient_cost_fn_input(pred_xs[:-1], sol)

        # l_xx.shape = (pred_len+1, state_size, state_size)
        l_xx = self.hessian_cost_fn_state(pred_xs[:-1],
                                          g_x[:-1], terminal=False)
        terminal_l_xx = \
            self.hessian_cost_fn_state(pred_xs[-1],
                                       g_x[-1], terminal=True)

        l_xx = np.concatenate((l_xx, terminal_l_xx), axis=0)

        # l_uu.shape = (pred_len, input_size, input_size)
        l_uu = self.hessian_cost_fn_input(pred_xs[:-1], sol)

        # l_ux.shape = (pred_len, input_size, state_size)
        l_ux = self.hessian_cost_fn_input_state(pred_xs[:-1], sol)

        return l_x, l_xx, l_u, l_uu, l_ux

    def backward(self, f_x, f_u, f_xx, f_ux, f_uu, l_x, l_xx, l_u, l_uu, l_ux):
        """ backward step of iLQR

        Args:
            f_x (numpy.ndarray): gradient of model with respecto to state,
                shape(pred_len+1, state_size, state_size)
            f_u (numpy.ndarray): gradient of model with respecto to input,
                shape(pred_len, state_size, input_size)
            f_xx (numpy.ndarray): gradient of model with respecto to state,
                shape(pred_len+1, state_size, state_size, state_size)
            f_ux (numpy.ndarray): gradient of model with respecto to input,
                shape(pred_len, state_size, input_size, state_size)
            f_uu (numpy.ndarray): gradient of model with respecto to input,
                shape(pred_len, state_size, input_size, input_size)
            l_x (numpy.ndarray): gradient of cost with respecto to state,
                shape(pred_len+1, state_size)
            l_u (numpy.ndarray): gradient of cost with respecto to input,
                shape(pred_len, input_size)
            l_xx (numpy.ndarray): hessian of cost with respecto to state,
                shape(pred_len+1, state_size, state_size)
            l_uu (numpy.ndarray): hessian of cost with respecto to input,
                shape(pred_len, input_size, input_size)
            l_ux (numpy.ndarray): hessian of cost with respect
                to state and input, shape(pred_len, input_size, state_size)
        Returns:
            k (numpy.ndarray): gain, shape(pred_len, input_size)
            K (numpy.ndarray): gain, shape(pred_len, input_size, state_size)
        """
        # get size
        (_, state_size, _) = f_x.shape

        # initialzie
        V_x = l_x[-1]
        V_xx = l_xx[-1]
        k = np.zeros((self.pred_len, self.input_size))
        K = np.zeros((self.pred_len, self.input_size, state_size))

        for t in range(self.pred_len-1, -1, -1):
            # get Q val
            Q_x, Q_u, Q_xx, Q_ux, Q_uu = self._Q(f_x[t], f_u[t],
                                                 f_xx[t], f_ux[t], f_uu[t],
                                                 l_x[t],
                                                 l_u[t], l_xx[t], l_ux[t],
                                                 l_uu[t], V_x, V_xx)
            # calc gain
            k[t] = - np.linalg.solve(Q_uu, Q_u)
            K[t] = - np.linalg.solve(Q_uu, Q_ux)
            # update V_x val
            V_x = Q_x + np.dot(np.dot(K[t].T, Q_uu), k[t])
            V_x += np.dot(K[t].T, Q_u) + np.dot(Q_ux.T, k[t])
            # update V_xx val
            V_xx = Q_xx + np.dot(np.dot(K[t].T, Q_uu), K[t])
            V_xx += np.dot(K[t].T, Q_ux) + np.dot(Q_ux.T, K[t])
            V_xx = 0.5 * (V_xx + V_xx.T)  # to maintain symmetry.

        return k, K

    def _Q(self, f_x, f_u, f_xx, f_ux, f_uu,
           l_x, l_u, l_xx, l_ux, l_uu, V_x, V_xx):
        """ compute Q function valued

        Args:
            f_x (numpy.ndarray): gradient of model with respecto to state,
                shape(state_size, state_size)
            f_u (numpy.ndarray): gradient of model with respecto to input,
                shape(state_size, input_size)
            f_xx (numpy.ndarray): gradient of model with respecto to state,
                shape(state_size, state_size, state_size)
            f_ux (numpy.ndarray): gradient of model with respecto to input,
                shape(state_size, input_size, state_size)
            f_uu (numpy.ndarray): gradient of model with respecto to input,
                shape(state_size, input_size, input_size)
            l_x (numpy.ndarray): gradient of cost with respecto to state,
                shape(state_size, )
            l_u (numpy.ndarray): gradient of cost with respecto to input,
                shape(input_size, )
            l_xx (numpy.ndarray): hessian of cost with respecto to state,
                shape(state_size, state_size)
            l_uu (numpy.ndarray): hessian of cost with respecto to input,
                shape(input_size, input_size)
            l_ux (numpy.ndarray): hessian of cost with respect
                to state and input, shape(input_size, state_size)
            V_x (numpy.ndarray): gradient of Value function,
                shape(state_size, )
            V_xx (numpy.ndarray): hessian of Value function,
                shape(state_size, state_size)
        Returns:
            Q_x (numpy.ndarray): gradient of Q function, shape(state_size, )
            Q_u (numpy.ndarray): gradient of Q function, shape(input_size, )
            Q_xx (numpy.ndarray): hessian of Q fucntion,
                shape(state_size, state_size)
            Q_ux (numpy.ndarray): hessian of Q fucntion,
                shape(input_size, state_size)
            Q_uu (numpy.ndarray): hessian of Q fucntion,
                shape(input_size, input_size)
        """
        # get size
        state_size = len(l_x)

        Q_x = l_x + np.dot(f_x.T, V_x)
        Q_u = l_u + np.dot(f_u.T, V_x)
        Q_xx = l_xx + np.dot(np.dot(f_x.T, V_xx), f_x)

        reg = self.mu * np.eye(state_size)
        Q_ux = l_ux + np.dot(np.dot(f_u.T, (V_xx + reg)), f_x)
        Q_uu = l_uu + np.dot(np.dot(f_u.T, (V_xx + reg)), f_u)

        # tensor constraction
        Q_xx += np.tensordot(V_x, f_xx, axes=1)
        Q_ux += np.tensordot(V_x, f_ux, axes=1)
        Q_uu += np.tensordot(V_x, f_uu, axes=1)

        return Q_x, Q_u, Q_xx, Q_ux, Q_uu
