import numpy as np
from matplotlib.axes import Axes

from ..plotters.plot_objs import square_with_angle, square
from ..common.utils import fit_angle_in_range


class TwoWheeledConfigModule():
    # parameters
    ENV_NAME = "TwoWheeled-v0"
    TYPE = "Nonlinear"
    N_AHEAD = 1
    TASK_HORIZON = 1000
    PRED_LEN = 20
    STATE_SIZE = 3
    INPUT_SIZE = 2
    DT = 0.01
    # cost parameters
    # for Const goal
    """
    R = np.diag([0.1, 0.1])
    Q = np.diag([1., 1., 0.01])
    Sf = np.diag([5., 5., 1.])
    """
    # for track goal
    """
    R = np.diag([0.01, 0.01])
    Q = np.diag([2.5, 2.5, 0.01])
    Sf = np.diag([2.5, 2.5, 0.01])
    """
    # for track goal to NMPC
    R = np.diag([1., 1.])
    Q = np.diag([0.001, 0.001, 0.001])
    Sf = np.diag([1., 1., 0.001])
    # bounds
    INPUT_LOWER_BOUND = np.array([-1.5, -3.14])
    INPUT_UPPER_BOUND = np.array([1.5, 3.14])
    # parameters
    CAR_SIZE = 0.2
    WHEELE_SIZE = (0.075, 0.015)

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
                "lambda": 1,
                "noise_sigma": 1.,
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
            "NMPC": {
                "threshold": 0.01,
                "max_iters": 5000,
                "learning_rate": 0.01,
                "optimizer_mode": "conjugate"
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
        return (u**2) * np.diag(TwoWheeledConfigModule.R)

    @staticmethod
    def fit_diff_in_range(diff_x):
        """ fit difference state in range(angle)

        Args:
            diff_x (numpy.ndarray): 
                shape(pop_size, pred_len, state_size) or
                shape(pred_len, state_size) or
                shape(state_size, )
        Returns:
            fitted_diff_x (numpy.ndarray): same shape as diff_x
        """
        if len(diff_x.shape) == 3:
            diff_x[:, :, -1] = fit_angle_in_range(diff_x[:, :, -1])
        elif len(diff_x.shape) == 2:
            diff_x[:, -1] = fit_angle_in_range(diff_x[:, -1])
        elif len(diff_x.shape) == 1:
            diff_x[-1] = fit_angle_in_range(diff_x[-1])

        return diff_x

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
        diff = TwoWheeledConfigModule.fit_diff_in_range(x - g_x)
        return ((diff)**2) * np.diag(TwoWheeledConfigModule.Q)

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
        terminal_diff = TwoWheeledConfigModule.fit_diff_in_range(terminal_x
                                                                 - terminal_g_x)

        return ((terminal_diff)**2) * np.diag(TwoWheeledConfigModule.Sf)

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
        diff = TwoWheeledConfigModule.fit_diff_in_range(x - g_x)

        if not terminal:
            return 2. * (diff) * np.diag(TwoWheeledConfigModule.Q)

        return (2. * (diff)
                * np.diag(TwoWheeledConfigModule.Sf))[np.newaxis, :]

    @staticmethod
    def gradient_cost_fn_input(x, u):
        """ gradient of costs with respect to the input

        Args:
            x (numpy.ndarray): state, shape(pred_len, state_size)
            u (numpy.ndarray): goal state, shape(pred_len, input_size)

        Returns:
            l_u (numpy.ndarray): gradient of cost, shape(pred_len, input_size)
        """
        return 2. * u * np.diag(TwoWheeledConfigModule.R)

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
            return np.tile(2.*TwoWheeledConfigModule.Q, (pred_len, 1, 1))

        return np.tile(2.*TwoWheeledConfigModule.Sf, (1, 1, 1))

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

        return np.tile(2.*TwoWheeledConfigModule.R, (pred_len, 1, 1))

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

    @staticmethod
    def gradient_hamiltonian_input(x, lam, u, g_x):
        """

        Args:
            x (numpy.ndarray): shape(pred_len+1, state_size)
            lam (numpy.ndarray): shape(pred_len, state_size)
            u (numpy.ndarray): shape(pred_len, input_size)
            g_xs (numpy.ndarray): shape(pred_len, state_size)

        Returns:
            F (numpy.ndarray), shape(pred_len, input_size)
        """
        if len(x.shape) == 1:
            input_size = u.shape[0]
            F = np.zeros(input_size)
            F[0] = u[0] * TwoWheeledConfigModule.R[0, 0] + \
                lam[0] * np.cos(x[2]) + lam[1] * np.sin(x[2])
            F[1] = u[1] * TwoWheeledConfigModule.R[1, 1] + lam[2]

            return F

        elif len(x.shape) == 2:
            pred_len, input_size = u.shape
            F = np.zeros((pred_len, input_size))

            for i in range(pred_len):
                F[i, 0] = u[i, 0] * TwoWheeledConfigModule.R[0, 0] + \
                    lam[i, 0] * np.cos(x[i, 2]) + lam[i, 1] * np.sin(x[i, 2])
                F[i, 1] = u[i, 1] * TwoWheeledConfigModule.R[1, 1] + lam[i, 2]

            return F
        else:
            raise NotImplementedError

    @staticmethod
    def gradient_hamiltonian_state(x, lam, u, g_x):
        """
        Args:
            x (numpy.ndarray): shape(pred_len+1, state_size)
            lam (numpy.ndarray): shape(pred_len, state_size)
            u (numpy.ndarray): shape(pred_len, input_size)
            g_xs (numpy.ndarray): shape(pred_len, state_size)

        Returns:
            lam_dot (numpy.ndarray), shape(state_size, )
        """
        if len(lam.shape) == 1:
            state_size = lam.shape[0]
            lam_dot = np.zeros(state_size)
            lam_dot[0] = \
                (x[0] - g_x[0]) * TwoWheeledConfigModule.Q[0, 0]
            lam_dot[1] = \
                (x[1] - g_x[1]) * TwoWheeledConfigModule.Q[1, 1]

            relative_angle = fit_angle_in_range(x[2] - g_x[2])
            lam_dot[2] = \
                relative_angle * TwoWheeledConfigModule.Q[2, 2] \
                - lam[0] * u[0] * np.sin(x[2]) \
                + lam[1] * u[0] * np.cos(x[2])

            return lam_dot

        elif len(lam.shape) == 2:
            pred_len, state_size = lam.shape
            lam_dot = np.zeros((pred_len, state_size))

            for i in range(pred_len):
                lam_dot[i, 0] = \
                    (x[i, 0] - g_x[i, 0]) * TwoWheeledConfigModule.Q[0, 0]
                lam_dot[i, 1] = \
                    (x[i, 1] - g_x[i, 1]) * TwoWheeledConfigModule.Q[1, 1]

                relative_angle = fit_angle_in_range(x[i, 2] - g_x[i, 2])
                lam_dot[i, 2] = \
                    relative_angle * TwoWheeledConfigModule.Q[2, 2] \
                    - lam[i, 0] * u[i, 0] * np.sin(x[i, 2]) \
                    + lam[i, 1] * u[i, 0] * np.cos(x[i, 2])

            return lam_dot
        else:
            raise NotImplementedError


class TwoWheeledExtendConfigModule(TwoWheeledConfigModule):
    PRED_LEN = 20

    def __init__(self):
        super().__init__()
        self.opt_config = {
            "NMPCCGMRES": {
                "threshold": 1e-3,
                "zeta": 5.,
                "delta": 0.01,
                "alpha": 0.5,
                "tf": 1.,
                "constraint": True
            },
            "NMPCNewton": {
                "threshold": 1e-3,
                "max_iteration": 500,
                "learning_rate": 1e-3
            }
        }

    @staticmethod
    def gradient_hamiltonian_input_with_constraint(x, lam, u, g_x, dummy_u, raw):
        """

        Args:
            x (numpy.ndarray): shape(pred_len+1, state_size)
            lam (numpy.ndarray): shape(pred_len, state_size)
            u (numpy.ndarray): shape(pred_len, input_size)
            g_xs (numpy.ndarray): shape(pred_len, state_size)
            dummy_u (numpy.ndarray): shape(pred_len, input_size)
            raw (numpy.ndarray): shape(pred_len, input_size), Lagrangian for constraints

        Returns:
            F (numpy.ndarray), shape(pred_len, 3)
        """
        if len(x.shape) == 1:
            vanilla_F = np.zeros(2)
            extend_F = np.zeros(2)  # 1 is the same as input size
            extend_C = np.zeros(2)

            vanilla_F[0] = u[0] + lam[0] * \
                np.cos(x[2]) + lam[1] * np.sin(x[2]) + 2. * raw[0] * u[0]
            vanilla_F[1] = u[1] + lam[2] + 2 * raw[1] * u[1]

            extend_F[0] = -0.01 + 2. * raw[0] * dummy_u[0]
            extend_F[1] = -0.01 + 2. * raw[1] * dummy_u[1]

            extend_C[0] = u[0]**2 + dummy_u[0]**2 - \
                TwoWheeledConfigModule.INPUT_LOWER_BOUND[0]**2
            extend_C[1] = u[1]**2 + dummy_u[1]**2 - \
                TwoWheeledConfigModule.INPUT_LOWER_BOUND[1]**2

            F = np.concatenate([vanilla_F, extend_F, extend_C])

        elif len(x.shape) == 2:
            pred_len, _ = u.shape
            vanilla_F = np.zeros((pred_len, 2))
            extend_F = np.zeros((pred_len, 2))  # 1 is the same as input size
            extend_C = np.zeros((pred_len, 2))

            for i in range(pred_len):
                vanilla_F[i, 0] = u[i, 0] + lam[i, 0] * \
                    np.cos(x[i, 2]) + lam[i, 1] * \
                    np.sin(x[i, 2]) + 2. * raw[i, 0] * u[i, 0]
                vanilla_F[i, 1] = u[i, 1] + lam[i, 2] + 2 * raw[i, 1] * u[i, 1]

                extend_F[i, 0] = -0.01 + 2. * raw[i, 0] * dummy_u[i, 0]
                extend_F[i, 1] = -0.01 + 2. * raw[i, 1] * dummy_u[i, 1]

                extend_C[i, 0] = u[i, 0]**2 + dummy_u[i, 0]**2 - \
                    TwoWheeledConfigModule.INPUT_LOWER_BOUND[0]**2
                extend_C[i, 1] = u[i, 1]**2 + dummy_u[i, 1]**2 - \
                    TwoWheeledConfigModule.INPUT_LOWER_BOUND[1]**2

            F = np.concatenate([vanilla_F, extend_F, extend_C], axis=1)

        return F
