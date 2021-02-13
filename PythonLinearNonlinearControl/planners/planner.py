import numpy as np


class Planner():
    """
    """

    def __init__(self):
        """
        """
        pass

    def plan(self, curr_x):
        """
        Args:
            curr_x (numpy.ndarray): current state, shape(state_size)
        Returns:
            g_xs (numpy.ndarrya): goal state, shape(pred_len, state_size)
        """
        raise NotImplementedError("Implement plan func")
