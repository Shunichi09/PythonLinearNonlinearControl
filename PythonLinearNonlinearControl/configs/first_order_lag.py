import numpy as np


class FirstOrderLagConfigModule():
    # parameters
    ENV_NAME = "FirstOrderLag-v0"
    TYPE = "Linear"
    TASK_HORIZON = 1000
    PRED_LEN = 50
    STATE_SIZE = 4
    INPUT_SIZE = 2
    DT = 0.05
    # cost parameters
    R = np.eye(INPUT_SIZE)
    Q = np.eye(STATE_SIZE)
    Sf = np.eye(STATE_SIZE)
    # bounds
    INPUT_LOWER_BOUND = np.array([-0.5, -0.5])
    INPUT_UPPER_BOUND = np.array([0.5, 0.5])
    # DT_INPUT_LOWER_BOUND = np.array([-0.5 * DT, -0.5 * DT])
    # DT_INPUT_UPPER_BOUND = np.array([0.25 * DT, 0.25 * DT])
    DT_INPUT_LOWER_BOUND = None
    DT_INPUT_UPPER_BOUND = None

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
                "init_var": 1.,
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
            "MPC": {
            },
            "iLQR": {
                "max_iters": 500,
                "init_mu": 1.,
                "mu_min": 1e-6,
                "mu_max": 1e10,
                "init_delta": 2.,
                "threshold": 1e-6,
            },
            "DDP": {
                "max_iters": 500,
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
        return (u**2) * np.diag(FirstOrderLagConfigModule.R)

    @staticmethod
    def state_cost_fn(x, g_x):
        """ state cost function
        Args:
            x (numpy.ndarray): state, shape(pred_len, state_size)
                or shape(pop_size, pred_len, state_size)
            g_x (numpy.ndarray): goal state, shape(pred_len, state_size)
                or shape(pop_size, pred_len, state_size)
        Returns:
            cost (numpy.ndarray): cost of state, shape(pred_len, state_size) or
                shape(pop_size, pred_len, state_size)
        """
        return ((x - g_x)**2) * np.diag(FirstOrderLagConfigModule.Q)

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
        return ((terminal_x - terminal_g_x)**2) \
            * np.diag(FirstOrderLagConfigModule.Sf)

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
            return 2. * (x - g_x) * np.diag(FirstOrderLagConfigModule.Q)

        return (2. * (x - g_x)
                * np.diag(FirstOrderLagConfigModule.Sf))[np.newaxis, :]

    @staticmethod
    def gradient_cost_fn_input(x, u):
        """ gradient of costs with respect to the input

        Args:
            x (numpy.ndarray): state, shape(pred_len, state_size)
            u (numpy.ndarray): goal state, shape(pred_len, input_size)

        Returns:
            l_u (numpy.ndarray): gradient of cost, shape(pred_len, input_size)
        """
        return 2. * u * np.diag(FirstOrderLagConfigModule.R)

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
            (pred_len, _) = x.shape
            return np.tile(2.*FirstOrderLagConfigModule.Q, (pred_len, 1, 1))

        return np.tile(2.*FirstOrderLagConfigModule.Sf, (1, 1, 1))

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

        return np.tile(2.*FirstOrderLagConfigModule.R, (pred_len, 1, 1))

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
