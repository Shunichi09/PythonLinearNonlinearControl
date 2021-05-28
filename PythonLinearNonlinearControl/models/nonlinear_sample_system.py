import numpy as np

from .model import Model
from ..common.utils import update_state_with_Runge_Kutta


class NonlinearSampleSystemModel(Model):
    """ nonlinear sample system model
    """

    def __init__(self, config):
        """
        """
        super(NonlinearSampleSystemModel, self).__init__()
        self.dt = config.DT
        self.gradient_hamiltonian_state = config.gradient_hamiltonian_state
        self.gradient_hamiltonian_input = config.gradient_hamiltonian_input
        self.gradient_cost_fn_state = config.gradient_cost_fn_state

    def predict_next_state(self, curr_x, u):
        """ predict next state

        Args:
            curr_x (numpy.ndarray): current state, shape(state_size, ) or
                shape(pop_size, state_size)
            u (numpy.ndarray): input, shape(input_size, ) or
                shape(pop_size, input_size)
        Returns:
            next_x (numpy.ndarray): next state, shape(state_size, ) or
                shape(pop_size, state_size)
        """
        if len(u.shape) == 1:
            func_1 = self._func_x_1
            func_2 = self._func_x_2
            functions = [func_1, func_2]
            next_x = update_state_with_Runge_Kutta(
                curr_x, u, functions, batch=False, dt=self.dt)
            return next_x

        elif len(u.shape) == 2:
            def func_1(xs, us): return self._func_x_1(xs, us, batch=True)
            def func_2(xs, us): return self._func_x_2(xs, us, batch=True)
            functions = [func_1, func_2]
            next_x = update_state_with_Runge_Kutta(
                curr_x, u, functions, batch=True, dt=self.dt)

            return next_x

    def x_dot(self, curr_x, u):
        """
        Args:
            curr_x (numpy.ndarray): current state, shape(state_size, )
            u (numpy.ndarray): input, shape(input_size, )
        Returns:
            x_dot (numpy.ndarray): next state, shape(state_size, )
        """
        state_size = curr_x.shape[0]
        x_dot = np.zeros(state_size)
        x_dot[0] = self._func_x_1(curr_x, u)
        x_dot[1] = self._func_x_2(curr_x, u)
        return x_dot

    def predict_adjoint_state(self, lam, x, u, g_x=None):
        """ predict adjoint states

        Args:
            lam (numpy.ndarray): adjoint state, shape(state_size, )
            x (numpy.ndarray): state, shape(state_size, )
            u (numpy.ndarray): input, shape(input_size, )
            goal (numpy.ndarray): goal state, shape(state_size, )
        Returns:
            prev_lam (numpy.ndarrya): previous adjoint state,
                shape(state_size, )
        """
        if len(u.shape) == 1:
            delta_lam = self.dt * \
                self.gradient_hamiltonian_state(x, lam, u, g_x)
            prev_lam = lam + delta_lam
            return prev_lam

        elif len(u.shape) == 2:
            raise ValueError

    def predict_terminal_adjoint_state(self, terminal_x, terminal_g_x=None):
        """ predict terminal adjoint state

        Args:
            terminal_x (numpy.ndarray): terminal state, shape(state_size, )
            terminal_g_x (numpy.ndarray): terminal goal state,
                shape(state_size, )
        Returns:
            terminal_lam (numpy.ndarray): terminal adjoint state,
                shape(state_size, )
        """
        terminal_lam = self.gradient_cost_fn_state(
            terminal_x, terminal_g_x, terminal=True)  # return in shape[1, state_size]
        return terminal_lam[0]

    def _func_x_1(self, x, u, batch=False):
        if not batch:
            x_dot = x[1]
        else:
            x_dot = x[:, 1]
        return x_dot

    def _func_x_2(self, x, u, batch=False):
        if not batch:
            x_dot = (1. - x[0]**2 - x[1]**2) * x[1] - x[0] + u[0]
        else:
            x_dot = (1. - x[:, 0]**2 - x[:, 1]**2) * \
                x[:, 1] - x[:, 0] + u[:, 0]
        return x_dot

    def calc_f_x(self, xs, us, dt):
        """ gradient of model with respect to the state in batch form
        Args:
            xs (numpy.ndarray): state, shape(pred_len+1, state_size)
            us (numpy.ndarray): input, shape(pred_len, input_size,)

        Return:
            f_x (numpy.ndarray): gradient of model with respect to x,
                shape(pred_len, state_size, state_size)

        Notes:
            This should be discrete form !!
        """
        # get size
        (_, state_size) = xs.shape
        (pred_len, _) = us.shape

        f_x = np.zeros((pred_len, state_size, state_size))
        f_x[:, 0, 1] = 1.
        f_x[:, 1, 0] = 2. * xs[:, 0] * xs[:, 1] - 1.
        f_x[:, 1, 1] = - 2. * xs[:, 1] * xs[:, 1] + \
            (1. - xs[:, 0]**2 - xs[:, 1]**2)

        return f_x * dt + np.eye(state_size)  # to discrete form

    def calc_f_u(self, xs, us, dt):
        """ gradient of model with respect to the input in batch form
        Args:
            xs (numpy.ndarray): state, shape(pred_len+1, state_size)
            us (numpy.ndarray): input, shape(pred_len, input_size,)

        Return:
            f_u (numpy.ndarray): gradient of model with respect to x,
                shape(pred_len, state_size, input_size)

        Notes:
            This should be discrete form !!
        """
        # get size
        (_, state_size) = xs.shape
        (pred_len, input_size) = us.shape

        f_u = np.zeros((pred_len, state_size, input_size))

        f_u[:, 1, 0] = 1.

        return f_u * dt  # to discrete form

    def calc_f_xx(self, xs, us, dt):
        """ hessian of model with respect to the state in batch form

        Args:
            xs (numpy.ndarray): state, shape(pred_len+1, state_size)
            us (numpy.ndarray): input, shape(pred_len, input_size,)

        Return:
            f_xx (numpy.ndarray): gradient of model with respect to x,
                shape(pred_len, state_size, state_size, state_size)
        """
        # get size
        (_, state_size) = xs.shape
        (pred_len, _) = us.shape

        f_xx = np.zeros((pred_len, state_size, state_size, state_size))

        raise NotImplementedError

    def calc_f_ux(self, xs, us, dt):
        """ hessian of model with respect to state and input in batch form

        Args:
            xs (numpy.ndarray): state, shape(pred_len+1, state_size)
            us (numpy.ndarray): input, shape(pred_len, input_size,)

        Return:
            f_ux (numpy.ndarray): gradient of model with respect to x,
                shape(pred_len, state_size, input_size, state_size)
        """
        # get size
        (_, state_size) = xs.shape
        (pred_len, input_size) = us.shape

        f_ux = np.zeros((pred_len, state_size, input_size, state_size))

        raise NotImplementedError

    def calc_f_uu(self, xs, us, dt):
        """ hessian of model with respect to input in batch form

        Args:
            xs (numpy.ndarray): state, shape(pred_len+1, state_size)
            us (numpy.ndarray): input, shape(pred_len, input_size,)

        Return:
            f_uu (numpy.ndarray): gradient of model with respect to x,
                shape(pred_len, state_size, input_size, input_size)
        """
        # get size
        (_, state_size) = xs.shape
        (pred_len, input_size) = us.shape

        f_uu = np.zeros((pred_len, state_size, input_size, input_size))

        raise NotImplementedError
