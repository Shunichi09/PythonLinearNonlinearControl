import numpy as np


class Model():
    """ base class of model
    """

    def __init__(self):
        """
        """
        pass

    def predict_traj(self, curr_x, us):
        """ predict trajectories

        Args:
            curr_x (numpy.ndarray): current state, shape(state_size, )
            us (numpy.ndarray): inputs,
                shape(pred_len, input_size) 
                or shape(pop_size, pred_len, input_size)
        Returns:
            pred_xs (numpy.ndarray): predicted state,
                shape(pred_len+1, state_size) including current state
                or shape(pop_size, pred_len+1, state_size)
        """
        if len(us.shape) == 3:
            pred_xs = self._predict_traj_alltogether(curr_x, us)
        elif len(us.shape) == 2:
            pred_xs = self._predict_traj(curr_x, us)
        else:
            raise ValueError("Invalid us")

        return pred_xs

    def _predict_traj(self, curr_x, us):
        """ predict trajectories

        Args:
            curr_x (numpy.ndarray): current state, shape(state_size, )
            us (numpy.ndarray): inputs, shape(pred_len, input_size)
        Returns:
            pred_xs (numpy.ndarray): predicted state,
                shape(pred_len+1, state_size) including current state
        """
        # get size
        pred_len = us.shape[0]
        # initialze
        x = curr_x
        pred_xs = curr_x[np.newaxis, :]

        for t in range(pred_len):
            next_x = self.predict_next_state(x, us[t])
            # update
            pred_xs = np.concatenate((pred_xs, next_x[np.newaxis, :]), axis=0)
            x = next_x

        return pred_xs

    def _predict_traj_alltogether(self, curr_x, us):
        """ predict trajectories for all samples

        Args:
            curr_x (numpy.ndarray): current state, shape(pop_size, state_size)
            us (numpy.ndarray): inputs, shape(pop_size, pred_len, input_size)
        Returns:
            pred_xs (numpy.ndarray): predicted state,
                shape(pop_size, pred_len+1, state_size) including current state
        """
        # get size
        (pop_size, pred_len, _) = us.shape
        us = np.transpose(us, (1, 0, 2))  # to (pred_len, pop_size, input_size)
        # initialze
        x = np.tile(curr_x, (pop_size, 1))
        pred_xs = x[np.newaxis, :, :]  # (1, pop_size, state_size)

        for t in range(pred_len):
            # next_x.shape = (pop_size, state_size)
            next_x = self.predict_next_state(x, us[t])
            # update
            pred_xs = np.concatenate((pred_xs, next_x[np.newaxis, :, :]),
                                     axis=0)
            x = next_x

        return np.transpose(pred_xs, (1, 0, 2))

    def predict_next_state(self, curr_x, u):
        """ predict next state
        """
        raise NotImplementedError("Implement the model")

    def x_dot(self, curr_x, u):
        """ compute x dot
        """
        raise NotImplementedError("Implement the model")

    def predict_adjoint_traj(self, xs, us, g_xs):
        """
        Args:
            xs (numpy.ndarray): states trajectory, shape(pred_len+1, state_size)
            us (numpy.ndarray): inputs, shape(pred_len, input_size)
            g_xs (numpy.ndarray): goal states, shape(pred_len+1, state_size)
        Returns:
            lams (numpy.ndarray): adjoint state, shape(pred_len, state_size),
                adjoint size is the same as state_size
        Notes:
            Adjoint trajectory be computed by backward path.
            Usually, we should -\dot{lam} but in backward path case, we can use \dot{lam} directry 
        """
        # get size
        (pred_len, input_size) = us.shape
        # pred final adjoint state
        lam = self.predict_terminal_adjoint_state(xs[-1],
                                                  terminal_g_x=g_xs[-1])
        lams = lam[np.newaxis, :]

        for t in range(pred_len-1, 0, -1):
            prev_lam = \
                self.predict_adjoint_state(lam, xs[t], us[t],
                                           g_x=g_xs[t])
            # update
            lams = np.concatenate((prev_lam[np.newaxis, :], lams), axis=0)
            lam = prev_lam

        return lams

    def predict_adjoint_state(self, lam, x, u, g_x=None, t=None):
        """ predict adjoint states

        Args:
            lam (numpy.ndarray): adjoint state, shape(state_size, )
            x (numpy.ndarray): state, shape(state_size, )
            u (numpy.ndarray): input, shape(input_size, )
            g_x (numpy.ndarray): goal state, shape(state_size, )
        Returns:
            prev_lam (numpy.ndarrya): previous adjoint state,
                shape(state_size, )
        """
        raise NotImplementedError("Implement the adjoint model")

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
        raise NotImplementedError("Implement terminal adjoint state")

    @staticmethod
    def calc_f_x(xs, us, dt):
        """ gradient of model with respect to the state in batch form
        """
        raise NotImplementedError("Implement gradient of model \
                                   with respect to the state")

    @staticmethod
    def calc_f_u(xs, us, dt):
        """ gradient of model with respect to the input in batch form
        """
        raise NotImplementedError("Implement gradient of model \
                                   with respect to the input")

    @staticmethod
    def calc_f_xx(xs, us, dt):
        """ hessian of model with respect to the state in batch form
        """
        raise NotImplementedError("Implement hessian of model \
                                   with respect to the state")

    @staticmethod
    def calc_f_ux(xs, us, dt):
        """ hessian of model with respect to the input in batch form
        """
        raise NotImplementedError("Implement hessian of model \
                                   with respect to the input and state")

    @staticmethod
    def calc_f_uu(xs, us, dt):
        """ hessian of model with respect to the state in batch form
        """
        raise NotImplementedError("Implement hessian of model \
                                   with respect to the input")


class LinearModel(Model):
    """ discrete linear model, x[k+1] = Ax[k] + Bu[k]

    Attributes:
        A (numpy.ndarray): shape(state_size, state_size)
        B (numpy.ndarray): shape(state_size, input_size)
    """

    def __init__(self, A, B):
        """
        """
        super(LinearModel, self).__init__()
        self.A = A
        self.B = B

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
            next_x = np.matmul(self.A, curr_x[:, np.newaxis]) \
                + np.matmul(self.B, u[:, np.newaxis])

            return next_x.flatten()

        elif len(u.shape) == 2:
            next_x = np.matmul(curr_x, self.A.T) + np.matmul(u, self.B.T)

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
        (pred_len, _) = us.shape

        return np.tile(self.A, (pred_len, 1, 1))

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
        (pred_len, input_size) = us.shape

        return np.tile(self.B, (pred_len, 1, 1))

    @staticmethod
    def calc_f_xx(xs, us, dt):
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

        return f_xx

    @staticmethod
    def calc_f_ux(xs, us, dt):
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

        return f_ux

    @staticmethod
    def calc_f_uu(xs, us, dt):
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

        return f_uu
