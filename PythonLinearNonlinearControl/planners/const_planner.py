import numpy as np
from .planner import Planner


class ConstantPlanner(Planner):
    """ This planner make constant goal state
    """

    def __init__(self, config):
        """
        """
        super(ConstantPlanner, self).__init__()
        self.pred_len = config.PRED_LEN
        self.state_size = config.STATE_SIZE

    def plan(self, curr_x, g_x=None):
        """
        Args:
            curr_x (numpy.ndarray): current state, shape(state_size)
            g_x (numpy.ndarray): goal state, shape(state_size),
                this state should be obtained from env
        Returns:
            g_xs (numpy.ndarrya): goal state, shape(pred_len, state_size)
        """
        return np.tile(g_x, (self.pred_len+1, 1))
