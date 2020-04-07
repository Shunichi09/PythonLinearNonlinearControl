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
        self.config = {"state_size" : 4,
                       "input_size" : 1,
                       "dt" : 0.02,
                       "max_step" : 500,
                       "input_lower_bound": [-3.],
                       "input_upper_bound": [3.],
                       "mp": 0.2,
                       "mc": 1.,
                       "l": 0.5,
                       "g": 9.81,
                       }

        super(CartPoleEnv, self).__init__(self.config)
    
    def reset(self, init_x=None):
        """ reset state

        Returns:
            init_x (numpy.ndarray): initial state, shape(state_size, )  
            info (dict): information
        """
        self.step_count = 0
        
        self.curr_x = np.array([0., 0., 0., 0.])

        if init_x is not None:
            self.curr_x = init_x

        # goal
        self.g_x = np.array([0., 0., -np.pi, 0.])
        
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
        # x
        d_x0 = self.curr_x[1]
        # v_x
        d_x1 = (u[0] + self.config["mp"] * np.sin(self.curr_x[2]) \
               * (self.config["l"] * (self.curr_x[3]**2) \
                  + self.config["g"] * np.cos(self.curr_x[2]))) \
               / (self.config["mc"] + self.config["mp"] \
                  * (np.sin(self.curr_x[2])**2))
        # theta
        d_x2 = self.curr_x[3]
        
        # v_theta
        d_x3 = (-u[0] * np.cos(self.curr_x[2]) \
                - self.config["mp"] * self.config["l"] * (self.curr_x[3]**2) \
                  * np.cos(self.curr_x[2]) * np.sin(self.curr_x[2]) \
                - (self.config["mc"] + self.config["mp"]) * self.config["g"] \
                   * np.sin(self.curr_x[2])) \
               / (self.config["l"] * (self.config["mc"] + self.config["mp"] \
                                      * (np.sin(self.curr_x[2])**2)))
        
        next_x = self.curr_x +\
                 np.array([d_x0, d_x1, d_x2, d_x3]) * self.config["dt"] 

        # TODO: costs
        costs = 0.
        costs += 0.1 * np.sum(u**2)
        costs += 6. * self.curr_x[0]**2 \
                 + 12. * (np.cos(self.curr_x[2]) + 1.)**2 \
                 + 0.1 * self.curr_x[1]**2 \
                 + 0.1 * self.curr_x[3]**2

        # save history
        self.history_x.append(next_x.flatten())
        self.history_g_x.append(self.g_x.flatten())
        
        # update
        self.curr_x = next_x.flatten().copy()
        # update costs
        self.step_count += 1

        return next_x.flatten(), costs, \
               self.step_count > self.config["max_step"], \
               {"goal_state" : self.g_x}