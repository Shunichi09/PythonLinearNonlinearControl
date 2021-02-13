import numpy as np


class Env():
    """ Environments class
    Attributes:

        curr_x (numpy.ndarray): current state 
        history_x (list[numpy.ndarray]): historty of state, shape(step_count*state_size)
        step_count (int): step count
    """

    def __init__(self, config):
        """
        """
        self.config = config
        self.curr_x = None
        self.goal_state = None
        self.history_x = []
        self.history_g_x = []
        self.step_count = None

    def reset(self, init_x=None):
        """ reset state
        Returns:
            init_x (numpy.ndarray): initial state, shape(state_size, )  
            info (dict): information
        """
        self.step_count = 0

        self.curr_x = np.zeros(self.config["state_size"])

        if init_x is not None:
            self.curr_x = init_x

        # clear memory
        self.history_x = []
        self.history_g_x = []

        return self.curr_x, {}

    def step(self, u):
        """
        Args:
            u (numpy.ndarray) : input, shape(input_size, )
        Returns:
            next_x (numpy.ndarray): next state, shape(state_size, ) 
            cost (float): costs
            done (bool): end the simulation or not
            info (dict): information 
        """
        raise NotImplementedError("Implement step function")

    def __repr__(self):
        """
        """
        return self.config
