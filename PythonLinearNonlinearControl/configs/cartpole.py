import numpy as np


class CartPoleConfigModule():
    # parameters
    ENV_NAME = "CartPole-v0"
    PLANNER_TYPE = "Const"
    TYPE = "Nonlinear"
    TASK_HORIZON = 500
    PRED_LEN = 50
    STATE_SIZE = 4
    INPUT_SIZE = 1
    DT = 0.02
    # cost parameters
    R = np.diag([0.01])  # 0.01 is worked for MPPI and CEM and MPPIWilliams
    # 1. is worked for iLQR
    TERMINAL_WEIGHT = 1.
    Q = None
    Sf = None
    # bounds
    INPUT_LOWER_BOUND = np.array([-3.])
    INPUT_UPPER_BOUND = np.array([3.])
    # parameters
    MP = 0.2
    MC = 1.
    L = 0.5
    G = 9.81
    CART_SIZE = (0.15, 0.1)

    def __init__(self):
        """ 
        """
        # opt configs
        self.opt_config = {
            "Random": {
                "popsize": 5000
            },
            "CEM": {
                "popsize": 500,
                "num_elites": 50,
                "max_iters": 15,
                "alpha": 0.3,
                "init_var": 9.,
                "threshold": 0.001
            },
            "MPPI": {
                "beta": 0.6,
                "popsize": 5000,
                "kappa": 0.9,
                "noise_sigma": 0.5,
            },
            "MPPIWilliams": {
                "popsize": 5000,
                "lambda": 1.,
                "noise_sigma": 0.9,
            },
            "iLQR": {
                "max_iter": 500,
                "init_mu": 1.,
                "mu_min": 1e-6,
                "mu_max": 1e10,
                "init_delta": 2.,
                "threshold": 1e-6,
            },
            "DDP": {
                "max_iter": 500,
                "init_mu": 1.,
                "mu_min": 1e-6,
                "mu_max": 1e10,
                "init_delta": 2.,
                "threshold": 1e-6,
            },
            "NMPC-CGMRES": {
            },
            "NMPC-Newton": {
            },
        }

    @staticmethod
    def input_cost_fn(u):
        """ input cost functions

        Args:
            u (numpy.ndarray): input, shape(pred_len, input_size)
                or shape(pop_size, pred_len, input_size)
        Returns:
            cost (numpy.ndarray): cost of input, shape(pred_len, input_size) or
                shape(pop_size, pred_len, input_size)
        """
        return (u**2) * np.diag(CartPoleConfigModule.R)

    @staticmethod
    def state_cost_fn(x, g_x):
        """ state cost function

        Args:
            x (numpy.ndarray): state, shape(pred_len, state_size)
                or shape(pop_size, pred_len, state_size)
            g_x (numpy.ndarray): goal state, shape(pred_len, state_size)
                or shape(pop_size, pred_len, state_size)
        Returns:
            cost (numpy.ndarray): cost of state, shape(pred_len, 1) or
                shape(pop_size, pred_len, 1)
        """

        if len(x.shape) > 2:
            return (6. * (x[:, :, 0]**2)
                    + 12. * ((np.cos(x[:, :, 2]) + 1.)**2)
                    + 0.1 * (x[:, :, 1]**2)
                    + 0.1 * (x[:, :, 3]**2))[:, :, np.newaxis]

        elif len(x.shape) > 1:
            return (6. * (x[:, 0]**2)
                    + 12. * ((np.cos(x[:, 2]) + 1.)**2)
                    + 0.1 * (x[:, 1]**2)
                    + 0.1 * (x[:, 3]**2))[:,  np.newaxis]

        return 6. * (x[0]**2) \
            + 12. * ((np.cos(x[2]) + 1.)**2) \
            + 0.1 * (x[1]**2) \
            + 0.1 * (x[3]**2)

    @staticmethod
    def terminal_state_cost_fn(terminal_x, terminal_g_x):
        """

        Args:
            terminal_x (numpy.ndarray): terminal state,
                shape(state_size, ) or shape(pop_size, state_size)
            terminal_g_x (numpy.ndarray): terminal goal state,
                shape(state_size, ) or shape(pop_size, state_size)
        Returns:
            cost (numpy.ndarray): cost of state, shape(pred_len, ) or
                shape(pop_size, pred_len)
        """

        if len(terminal_x.shape) > 1:
            return (6. * (terminal_x[:, 0]**2)
                    + 12. * ((np.cos(terminal_x[:, 2]) + 1.)**2)
                    + 0.1 * (terminal_x[:, 1]**2)
                    + 0.1 * (terminal_x[:, 3]**2))[:, np.newaxis] \
                * CartPoleConfigModule.TERMINAL_WEIGHT

        return (6. * (terminal_x[0]**2)
                + 12. * ((np.cos(terminal_x[2]) + 1.)**2)
                + 0.1 * (terminal_x[1]**2)
                + 0.1 * (terminal_x[3]**2)) \
            * CartPoleConfigModule.TERMINAL_WEIGHT

    @staticmethod
    def gradient_cost_fn_state(x, g_x, terminal=False):
        """ gradient of costs with respect to the state

        Args:
            x (numpy.ndarray): state, shape(pred_len, state_size)
            g_x (numpy.ndarray): goal state, shape(pred_len, state_size)

        Returns:
            l_x (numpy.ndarray): gradient of cost, shape(pred_len, state_size)
                or shape(1, state_size)
        """
        if not terminal:
            cost_dx0 = 12. * x[:, 0]
            cost_dx1 = 0.2 * x[:, 1]
            cost_dx2 = 24. * (1 + np.cos(x[:, 2])) * -np.sin(x[:, 2])
            cost_dx3 = 0.2 * x[:, 3]
            cost_dx = np.stack((cost_dx0, cost_dx1,
                                cost_dx2, cost_dx3), axis=1)
            return cost_dx

        cost_dx0 = 12. * x[0]
        cost_dx1 = 0.2 * x[1]
        cost_dx2 = 24. * (1 + np.cos(x[2])) * -np.sin(x[2])
        cost_dx3 = 0.2 * x[3]
        cost_dx = np.array([[cost_dx0, cost_dx1, cost_dx2, cost_dx3]])

        return cost_dx * CartPoleConfigModule.TERMINAL_WEIGHT

    @staticmethod
    def gradient_cost_fn_input(x, u):
        """ gradient of costs with respect to the input

        Args:
            x (numpy.ndarray): state, shape(pred_len, state_size)
            u (numpy.ndarray): goal state, shape(pred_len, input_size)
        Returns:
            l_u (numpy.ndarray): gradient of cost, shape(pred_len, input_size)
        """
        return 2. * u * np.diag(CartPoleConfigModule.R)

    @staticmethod
    def hessian_cost_fn_state(x, g_x, terminal=False):
        """ hessian costs with respect to the state

        Args:
            x (numpy.ndarray): state, shape(pred_len, state_size)
            g_x (numpy.ndarray): goal state, shape(pred_len, state_size)
        Returns:
            l_xx (numpy.ndarray): gradient of cost,
                shape(pred_len, state_size, state_size) or
                shape(1, state_size, state_size) or
        """
        if not terminal:
            (pred_len, state_size) = x.shape
            hessian = np.eye(state_size)
            hessian = np.tile(hessian, (pred_len, 1, 1))
            hessian[:, 0, 0] = 12.
            hessian[:, 1, 1] = 0.2
            hessian[:, 2, 2] = 24. * -np.sin(x[:, 2]) \
                * (-np.sin(x[:, 2])) \
                + 24. * (1. + np.cos(x[:, 2])) \
                * -np.cos(x[:, 2])
            hessian[:, 3, 3] = 0.2

            return hessian

        state_size = len(x)
        hessian = np.eye(state_size)
        hessian[0, 0] = 12.
        hessian[1, 1] = 0.2
        hessian[2, 2] = 24. * -np.sin(x[2]) \
            * (-np.sin(x[2])) \
            + 24. * (1. + np.cos(x[2])) \
            * -np.cos(x[2])
        hessian[3, 3] = 0.2

        return hessian[np.newaxis, :, :] * CartPoleConfigModule.TERMINAL_WEIGHT

    @staticmethod
    def hessian_cost_fn_input(x, u):
        """ hessian costs with respect to the input

        Args:
            x (numpy.ndarray): state, shape(pred_len, state_size)
            u (numpy.ndarray): goal state, shape(pred_len, input_size)
        Returns:
            l_uu (numpy.ndarray): gradient of cost,
                shape(pred_len, input_size, input_size)
        """
        (pred_len, _) = u.shape

        return np.tile(2.*CartPoleConfigModule.R, (pred_len, 1, 1))

    @staticmethod
    def hessian_cost_fn_input_state(x, u):
        """ hessian costs with respect to the state and input

        Args:
            x (numpy.ndarray): state, shape(pred_len, state_size)
            u (numpy.ndarray): goal state, shape(pred_len, input_size)
        Returns:
            l_ux (numpy.ndarray): gradient of cost ,
                shape(pred_len, input_size, state_size)
        """
        (_, state_size) = x.shape
        (pred_len, input_size) = u.shape

        return np.zeros((pred_len, input_size, state_size))
