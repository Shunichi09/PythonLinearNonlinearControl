import numpy as np

from ..envs.cost import calc_cost


class Controller():
    """ Controller class
    """

    def __init__(self, config, model):
        """
        """
        self.config = config
        self.model = model

        # get cost func
        self.state_cost_fn = config.state_cost_fn
        self.terminal_state_cost_fn = config.terminal_state_cost_fn
        self.input_cost_fn = config.input_cost_fn

    def obtain_sol(self, curr_x, g_xs):
        """ calculate the optimal inputs
        Args:
            curr_x (numpy.ndarray): current state, shape(state_size, )
            g_xs (numpy.ndarrya): goal trajectory, shape(plan_len, state_size)
        Returns:
            opt_input (numpy.ndarray): optimal input, shape(input_size, )
        """
        raise NotImplementedError("Implement the algorithm to \
                                   get optimal input")

    def calc_cost(self, curr_x, samples, g_xs):
        """ calculate the cost of input samples

        Args:
            curr_x (numpy.ndarray): shape(state_size),
                current robot position
            samples (numpy.ndarray): shape(pop_size, opt_dim), 
                input samples
            g_xs (numpy.ndarray): shape(pred_len, state_size),
                goal states
        Returns:
            costs (numpy.ndarray): shape(pop_size, )
        """
        # get size
        pop_size = samples.shape[0]
        g_xs = np.tile(g_xs, (pop_size, 1, 1))

        # calc cost, pred_xs.shape = (pop_size, pred_len+1, state_size)
        pred_xs = self.model.predict_traj(curr_x, samples)

        # get particle cost
        costs = calc_cost(pred_xs, samples, g_xs,
                          self.state_cost_fn, self.input_cost_fn,
                          self.terminal_state_cost_fn)

        return costs
