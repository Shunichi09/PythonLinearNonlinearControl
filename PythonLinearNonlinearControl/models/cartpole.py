import numpy as np

from .model import Model


class CartPoleModel(Model):
    """ cartpole model
    """

    def __init__(self, config):
        """
        """
        super(CartPoleModel, self).__init__()
        self.dt = config.DT
        self.mc = config.MC
        self.mp = config.MP
        self.l = config.L
        self.g = config.G

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
            # x
            d_x0 = curr_x[1]
            # v_x
            d_x1 = (u[0] + self.mp * np.sin(curr_x[2])
                    * (self.l * (curr_x[3]**2)
                       + self.g * np.cos(curr_x[2]))) \
                / (self.mc + self.mp * (np.sin(curr_x[2])**2))
            # theta
            d_x2 = curr_x[3]
            # v_theta
            d_x3 = (-u[0] * np.cos(curr_x[2])
                    - self.mp * self.l * (curr_x[3]**2)
                    * np.cos(curr_x[2]) * np.sin(curr_x[2])
                    - (self.mc + self.mp) * self.g * np.sin(curr_x[2])) \
                / (self.l * (self.mc + self.mp * (np.sin(curr_x[2])**2)))

            next_x = curr_x +\
                np.array([d_x0, d_x1, d_x2, d_x3]) * self.dt

            return next_x

        elif len(u.shape) == 2:
            # x
            d_x0 = curr_x[:, 1]
            # v_x
            d_x1 = (u[:, 0] + self.mp * np.sin(curr_x[:, 2])
                    * (self.l * (curr_x[:, 3]**2)
                       + self.g * np.cos(curr_x[:, 2]))) \
                / (self.mc + self.mp * (np.sin(curr_x[:, 2])**2))
            # theta
            d_x2 = curr_x[:, 3]
            # v_theta
            d_x3 = (-u[:, 0] * np.cos(curr_x[:, 2])
                    - self.mp * self.l * (curr_x[:, 3]**2)
                    * np.cos(curr_x[:, 2]) * np.sin(curr_x[:, 2])
                    - (self.mc + self.mp) * self.g * np.sin(curr_x[:, 2])) \
                / (self.l * (self.mc + self.mp * (np.sin(curr_x[:, 2])**2)))

            next_x = curr_x +\
                np.stack((d_x0, d_x1, d_x2, d_x3), axis=1) * self.dt

            return next_x

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

        # f_x_dot
        f_x[:, 0, 1] = np.ones(pred_len)

        # f_theta
        tmp = ((self.mc + self.mp * np.sin(xs[:, 2])**2)**(-2)) \
            * self.mp * 2. * np.sin(xs[:, 2]) * np.cos(xs[:, 2])
        tmp2 = 1. / (self.mc + self.mp * (np.sin(xs[:, 2])**2))

        f_x[:, 1, 2] = - us[:, 0] * tmp \
                       - tmp * (self.mp * np.sin(xs[:, 2])
                                * (self.l * xs[:, 3]**2
                                   + self.g * np.cos(xs[:, 2]))) \
            + tmp2 * (self.mp * np.cos(xs[:, 2]) * self.l
                      * xs[:, 3]**2
                      + self.mp * self.g * (np.cos(xs[:, 2])**2
                                            - np.sin(xs[:, 2])**2))
        f_x[:, 3, 2] = - 1. / self.l * tmp \
            * (-us[:, 0] * np.cos(xs[:, 2])
               - self.mp * self.l * (xs[:, 3]**2)
               * np.cos(xs[:, 2]) * np.sin(xs[:, 2])
               - (self.mc + self.mp) * self.g * np.sin(xs[:, 2])) \
            + 1. / self.l * tmp2 \
            * (us[:, 0] * np.sin(xs[:, 2])
               - self.mp * self.l * xs[:, 3]**2
               * (np.cos(xs[:, 2])**2 - np.sin(xs[:, 2])**2)
               - (self.mc + self.mp)
               * self.g * np.cos(xs[:, 2]))

        # f_theta_dot
        f_x[:, 1, 3] = tmp2 * (self.mp * np.sin(xs[:, 2])
                               * self.l * 2 * xs[:, 3])
        f_x[:, 2, 3] = np.ones(pred_len)
        f_x[:, 3, 3] = 1. / self.l * tmp2 \
            * (-2. * self.mp * self.l * xs[:, 3]
               * np.cos(xs[:, 2]) * np.sin(xs[:, 2]))

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

        f_u[:, 1, 0] = 1. / (self.mc + self.mp * (np.sin(xs[:, 2])**2))

        f_u[:, 3, 0] = -np.cos(xs[:, 2]) \
            / (self.l * (self.mc
                         + self.mp * (np.sin(xs[:, 2])**2)))

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
