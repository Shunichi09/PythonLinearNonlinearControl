import numpy as np

from .env import Env

class CartPoleEnv(Env):
    """ Cartpole Environment

    Ref : 
        https://ocw.mit.edu/courses/
        electrical-engineering-and-computer-science/
        6-832-underactuated-robotics-spring-2009/readings/
        MIT6_832s09_read_ch03.pdf
    """
    def __init__(self):
        """
        """
        self.config = {"state_size" : 4,\
                       "input_size" : 1,\
                       "dt" : 0.02,\
                       "max_step" : 1000,\
                       "input_lower_bound": None,\
                       "input_upper_bound": None,
                       }

        super(CartPoleEnv, self).__init__(self.config)
    
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

        # goal
        self.g_x = np.array([0., 0., np.pi, 0.])
        
        # clear memory
        self.history_x = []
        self.history_g_x = []

        return self.curr_x, {"goal_state": self.g_x}

    def step(self, u):
        """ step environments

        Args:
            u (numpy.ndarray) : input, shape(input_size, )
        Returns:
            next_x (numpy.ndarray): next state, shape(state_size, ) 
            cost (float): costs
            done (bool): end the simulation or not
            info (dict): information 
        """
        # clip action
        if self.config["input_lower_bound"] is not None:
            u = np.clip(u,
                        self.config["input_lower_bound"],
                        self.config["input_upper_bound"])

        # step
        next_x = np.zeros(self.config["state_size"])

        # TODO: costs
        costs = 0.
        costs += 0.1 * np.sum(u**2)
        costs += np.sum((self.curr_x - self.g_x)**2)


        # save history
        self.history_x.append(next_x.flatten())
        self.history_g_x.append(self.g_x.flatten())
        
        # update
        self.curr_x = next_x.flatten()
        # update costs
        self.step_count += 1

        return next_x.flatten(), costs, \
               self.step_count > self.config["max_step"], \
               {"goal_state" : self.g_x}