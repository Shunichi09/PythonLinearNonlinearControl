from logging import getLogger

import numpy as np
from scipy.optimize import minimize
from scipy.optimize import LinearConstraint

from .controller import Controller
from ..envs.cost import calc_cost

logger = getLogger(__name__)


class LinearMPC(Controller):
    """ Model Predictive Controller for linear model

    Attributes:
        A (numpy.ndarray): system matrix, shape(state_size, state_size)
        B (numpy.ndarray): input matrix, shape(state_size, input_size)
        Q (numpy.ndarray): cost function weight for states
        R (numpy.ndarray): cost function weight for states
        history_us (list[numpy.ndarray]): time history of optimal input
    Ref:
        Maciejowski, J. M. (2002). Predictive control: with constraints.
    """

    def __init__(self, config, model):
        """
        Args:
            model (Model): system matrix, shape(state_size, state_size)
            config (ConfigModule): input matrix, shape(state_size, input_size)
        """
        if config.TYPE != "Linear":
            raise ValueError("{} could be not applied to \
                              this controller".format(model))
        super(LinearMPC, self).__init__(config, model)
        # system parameters
        self.model = model
        self.A = model.A
        self.B = model.B
        self.state_size = config.STATE_SIZE
        self.input_size = config.INPUT_SIZE
        self.pred_len = config.PRED_LEN

        # get cost func
        self.state_cost_fn = config.state_cost_fn
        self.terminal_state_cost_fn = config.terminal_state_cost_fn
        self.input_cost_fn = config.input_cost_fn

        # cost parameters
        self.Q = config.Q
        self.R = config.R
        self.Qs = None
        self.Rs = None

        # constraints
        self.dt_input_lower_bound = config.DT_INPUT_LOWER_BOUND
        self.dt_input_upper_bound = config.DT_INPUT_UPPER_BOUND
        self.input_lower_bound = config.INPUT_LOWER_BOUND
        self.input_upper_bound = config.INPUT_UPPER_BOUND

        # setup controllers
        self.W = None
        self.omega = None
        self.F = None
        self.f = None
        self.setup()
        self.prev_sol = np.zeros(self.input_size*self.pred_len)

        # history
        self.history_u = [np.zeros(self.input_size)]

    def setup(self):
        """
        setup Model Predictive Control as a quadratic programming        
        """
        A_factorials = [self.A]
        self.phi_mat = self.A.copy()

        for _ in range(self.pred_len - 1):
            temp_mat = np.matmul(A_factorials[-1], self.A)
            self.phi_mat = np.vstack((self.phi_mat, temp_mat))
            A_factorials.append(temp_mat)  # after we use this factorials

        self.gamma_mat = self.B.copy()
        gammma_mat_temp = self.B.copy()

        for i in range(self.pred_len - 1):
            temp_1_mat = np.matmul(A_factorials[i], self.B)
            gammma_mat_temp = temp_1_mat + gammma_mat_temp
            self.gamma_mat = np.vstack((self.gamma_mat, gammma_mat_temp))

        self.theta_mat = self.gamma_mat.copy()

        for i in range(self.pred_len - 1):
            temp_mat = np.zeros_like(self.gamma_mat)
            temp_mat[int((i + 1)*self.state_size):, :] =\
                self.gamma_mat[:-int((i + 1)*self.state_size), :]

            self.theta_mat = np.hstack((self.theta_mat, temp_mat))

        # evaluation function weight
        diag_Qs = np.tile(np.diag(self.Q), self.pred_len)
        diag_Rs = np.tile(np.diag(self.R), self.pred_len)
        self.Qs = np.diag(diag_Qs)
        self.Rs = np.diag(diag_Rs)

        # constraints
        # about inputs
        if self.input_lower_bound is not None:
            self.F = np.zeros((self.input_size * 2,
                               self.pred_len * self.input_size))

            for i in range(self.input_size):
                self.F[i * 2: (i + 1) * 2, i] = np.array([1.,  -1.])
                temp_F = self.F.copy()

            for i in range(self.pred_len - 1):
                for j in range(self.input_size):
                    temp_F[j * 2: (j + 1) * 2,
                           ((i+1) * self.input_size) + j] = np.array([1., -1.])
                self.F = np.vstack((self.F, temp_F))

            self.F1 = self.F[:, :self.input_size]

            temp_f = []
            for i in range(self.input_size):
                temp_f.append(-1 * self.input_upper_bound[i])
                temp_f.append(self.input_lower_bound[i])

            self.f = np.tile(np.array(temp_f).flatten(), self.pred_len)

        # about dt_input constraints
        if self.dt_input_lower_bound is not None:
            self.W = np.zeros((2, self.pred_len * self.input_size))
            self.W[:, 0] = np.array([1.,  -1.])

            for i in range(self.pred_len * self.input_size - 1):
                temp_W = np.zeros((2, self.pred_len * self.input_size))
                temp_W[:, i+1] = np.array([1.,  -1.])
                self.W = np.vstack((self.W, temp_W))

            temp_omega = []

            for i in range(self.input_size):
                temp_omega.append(self.dt_input_upper_bound[i])
                temp_omega.append(-1. * self.dt_input_lower_bound[i])

            self.omega = np.tile(np.array(temp_omega).flatten(),
                                 self.pred_len)

    def obtain_sol(self, curr_x, g_xs):
        """ calculate the optimal inputs

        Args:
            curr_x (numpy.ndarray): current state, shape(state_size, )
            g_xs (numpy.ndarrya): goal trajectory,
                shape(plan_len+1, state_size)
        Returns:
            opt_input (numpy.ndarray): optimal input, shape(input_size, )
        """
        temp_1 = np.matmul(self.phi_mat, curr_x.reshape(-1, 1))
        temp_2 = np.matmul(self.gamma_mat, self.history_u[-1].reshape(-1, 1))

        error = g_xs[1:].reshape(-1, 1) - temp_1 - temp_2

        G = np.matmul(self.theta_mat.T, np.matmul(self.Qs, error))

        H = np.matmul(self.theta_mat.T, np.matmul(self.Qs, self.theta_mat)) \
            + self.Rs
        H = H * 0.5

        # constraints
        A = []
        b = []

        if self.W is not None:
            A.append(self.W)
            b.append(self.omega.reshape(-1, 1))

        if self.F is not None:
            b_F = - np.matmul(self.F1, self.history_u[-1].reshape(-1, 1)) \
                  - self.f.reshape(-1, 1)
            A.append(self.F)
            b.append(b_F)

        A = np.array(A).reshape(-1, self.input_size * self.pred_len)

        ub = np.array(b).flatten()

        # using cvxopt
        def optimized_func(dt_us):
            return (np.dot(dt_us, np.dot(H, dt_us.reshape(-1, 1)))
                    - np.dot(G.T, dt_us.reshape(-1, 1)))[0]

        # constraint
        lb = np.array([-np.inf for _ in range(len(ub))])  # one side cons
        cons = LinearConstraint(A, lb, ub)
        # solve
        opt_sol = minimize(optimized_func, self.prev_sol.flatten(),
                           constraints=[cons])
        opt_dt_us = opt_sol.x

        """ using cvxopt ver,
        if you want to solve more quick please use cvxopt instead of scipy
        
        # make cvxpy problem formulation
        P = 2*matrix(H)
        q = matrix(-1 * G)
        A = matrix(A)
        b = matrix(ub)

        # solve the problem
        opt_sol = solvers.qp(P, q, G=A, h=b)
        opt_dt_us = np.array(list(opt_sol['x']))
        """

        # to dt form
        opt_dt_u_seq = np.cumsum(opt_dt_us.reshape(self.pred_len,
                                                   self.input_size),
                                 axis=0)
        self.prev_sol = opt_dt_u_seq.copy()

        opt_u_seq = opt_dt_u_seq + self.history_u[-1]

        # save
        self.history_u.append(opt_u_seq[0])

        # check costs
        costs = self.calc_cost(curr_x,
                               opt_u_seq.reshape(1,
                                                 self.pred_len,
                                                 self.input_size),
                               g_xs)

        logger.debug("Cost = {}".format(costs))

        return opt_u_seq[0]

    def __str__(self):
        return "LinearMPC"
