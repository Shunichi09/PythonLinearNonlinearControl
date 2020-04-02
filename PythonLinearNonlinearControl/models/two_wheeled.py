import numpy as np

from .model import Model

class TwoWheeledModel(Model):
    """ two wheeled model
    """
    def __init__(self, config):
        """
        """
        super(TwoWheeledModel, self).__init__()
        self.dt = config.DT

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
            B = np.array([[np.cos(curr_x[-1]), 0.],
                          [np.sin(curr_x[-1]), 0.],
                          [0., 1.]])
            # calc dot
            x_dot = np.matmul(B, u[:, np.newaxis])
            # next state
            next_x = x_dot.flatten() * self.dt + curr_x

            return next_x

        elif len(u.shape) == 2:
            (pop_size, state_size) = curr_x.shape
            (_, input_size) = u.shape
            # B.shape = (pop_size, state_size, input_size)
            B = np.zeros((pop_size, state_size, input_size))
            # insert
            B[:, 0, 0] = np.cos(curr_x[:, -1])
            B[:, 1, 0] = np.sin(curr_x[:, -1])
            B[:, 2, 1] = np.ones(pop_size)

            # x_dot.shape = (pop_size, state_size, 1)
            x_dot = np.matmul(B, u[:, :, np.newaxis])
            # next state
            next_x = x_dot[:, :, 0] * self.dt + curr_x

            return next_x
        