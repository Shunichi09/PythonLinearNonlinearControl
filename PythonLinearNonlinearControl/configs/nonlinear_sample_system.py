import numpy as np


class NonlinearSampleSystemConfigModule():
    # parameters
    ENV_NAME = "NonlinearSampleSystem-v0"
    PLANNER_TYPE = "Const"
    TYPE = "Nonlinear"
    TASK_HORIZON = 2000
    PRED_LEN = 10
    STATE_SIZE = 2
    INPUT_SIZE = 1
    DT = 0.01
    R = np.diag([1.])
    Q = None
    Sf = None
    # bounds
    INPUT_LOWER_BOUND = np.array([-0.5])
    INPUT_UPPER_BOUND = np.array([0.5])

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
            }
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
        return (u**2) * np.diag(NonlinearSampleSystemConfigModule.R)

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
            return (0.5 * (x[:, :, 0]**2) +
                    0.5 * (x[:, :, 1]**2))[:, :, np.newaxis]

        elif len(x.shape) > 1:
            return (0.5 * (x[:, 0]**2) + 0.5 * (x[:, 1]**2))[:, np.newaxis]

        return 0.5 * (x[0]**2) + 0.5 * (x[1]**2)

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
            return (0.5 * (terminal_x[:, 0]**2) +
                    0.5 * (terminal_x[:, 1]**2))[:, np.newaxis]

        return 0.5 * (terminal_x[0]**2) + 0.5 * (terminal_x[1]**2)

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
            cost_dx0 = x[:, 0]
            cost_dx1 = x[:, 1]
            cost_dx = np.stack((cost_dx0, cost_dx1), axis=1)
            return cost_dx

        cost_dx0 = x[0]
        cost_dx1 = x[1]
        cost_dx = np.array([[cost_dx0, cost_dx1]])

        return cost_dx

    @staticmethod
    def gradient_cost_fn_input(x, u):
        """ gradient of costs with respect to the input

        Args:
            x (numpy.ndarray): state, shape(pred_len, state_size)
            u (numpy.ndarray): goal state, shape(pred_len, input_size)
        Returns:
            l_u (numpy.ndarray): gradient of cost, shape(pred_len, input_size)
        """
        return 2. * u * np.diag(NonlinearSampleSystemConfigModule.R)

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
            hessian[:, 0, 0] = 1.
            hessian[:, 1, 1] = 1.

            return hessian

        state_size = len(x)
        hessian = np.eye(state_size)
        hessian[0, 0] = 1.
        hessian[1, 1] = 1.

        return hessian[np.newaxis, :, :]

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

        return np.tile(NonlinearSampleSystemConfigModule.R, (pred_len, 1, 1))

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
            F[0] = u[0] + lam[1]

            return F

        elif len(x.shape) == 2:
            pred_len, input_size = u.shape
            F = np.zeros((pred_len, input_size))

            for i in range(pred_len):
                F[i, 0] = u[i, 0] + lam[i, 1]

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
            lam_dot[0] = x[0] - (2. * x[0] * x[1] + 1.) * lam[1]
            lam_dot[1] = x[1] + lam[0] + \
                (-3. * (x[1]**2) - x[0]**2 + 1.) * lam[1]

            return lam_dot

        elif len(lam.shape) == 2:
            pred_len, state_size = lam.shape
            lam_dot = np.zeros((pred_len, state_size))

            for i in range(pred_len):
                lam_dot[i, 0] = x[i, 0] - \
                    (2. * x[i, 0] * x[i, 1] + 1.) * lam[i, 1]
                lam_dot[i, 1] = x[i, 1] + lam[i, 0] + \
                    (-3. * (x[i, 1]**2) - x[i, 0]**2 + 1.) * lam[i, 1]

            return lam_dot

        else:
            raise NotImplementedError


class NonlinearSampleSystemExtendConfigModule(NonlinearSampleSystemConfigModule):
    def __init__(self):
        super().__init__()
        self.opt_config = {
            "NMPCCGMRES": {
                "threshold": 1e-3,
                "zeta": 100.,
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
            vanilla_F = np.zeros(1)
            extend_F = np.zeros(1)  # 1 is the same as input size
            extend_C = np.zeros(1)

            vanilla_F[0] = u[0] + lam[1] + 2. * raw[0] * u[0]
            extend_F[0] = -0.01 + 2. * raw[0] * dummy_u[0]
            extend_C[0] = u[0]**2 + dummy_u[0]**2 - \
                NonlinearSampleSystemConfigModule.INPUT_LOWER_BOUND**2

            F = np.concatenate([vanilla_F, extend_F, extend_C])

        elif len(x.shape) == 2:
            pred_len, _ = u.shape
            vanilla_F = np.zeros((pred_len, 1))
            extend_F = np.zeros((pred_len, 1))  # 1 is the same as input size
            extend_C = np.zeros((pred_len, 1))

            for i in range(pred_len):
                vanilla_F[i, 0] = \
                    u[i, 0] + lam[i, 1] + 2. * raw[i, 0] * u[i, 0]
                extend_F[i, 0] = -0.01 + 2. * raw[i, 0] * dummy_u[i, 0]
                extend_C[i, 0] = u[i, 0]**2 + dummy_u[i, 0]**2 - \
                    NonlinearSampleSystemConfigModule.INPUT_LOWER_BOUND**2

            F = np.concatenate([vanilla_F, extend_F, extend_C], axis=1)

        return F
