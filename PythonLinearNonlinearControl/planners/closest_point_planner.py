import numpy as np
from .planner import Planner


class ClosestPointPlanner(Planner):
    """ This planner make goal state according to goal path
    """

    def __init__(self, config):
        """
        """
        super(ClosestPointPlanner, self).__init__()
        self.pred_len = config.PRED_LEN
        self.state_size = config.STATE_SIZE
        self.n_ahead = config.N_AHEAD

    def plan(self, curr_x, g_traj):
        """
        Args:
            curr_x (numpy.ndarray): current state, shape(state_size)
            g_x (numpy.ndarray): goal state, shape(state_size),
                this state should be obtained from env
        Returns:
            g_xs (numpy.ndarrya): goal state, shape(pred_len+1, state_size)
        """
        min_idx = np.argmin(np.linalg.norm(curr_x[:-1] - g_traj[:, :-1],
                                           axis=1))

        start = (min_idx+self.n_ahead)
        if start > len(g_traj):
            start = len(g_traj)

        end = min_idx+self.n_ahead+self.pred_len+1

        if (min_idx+self.n_ahead+self.pred_len+1) > len(g_traj):
            end = len(g_traj)

        if abs(start - end) != self.pred_len + 1:
            return np.tile(g_traj[-1], (self.pred_len+1, 1))

        return g_traj[start:end]
