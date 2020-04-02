import numpy as np
import scipy
from scipy import integrate
from .env import Env

class TwoWheeledConstEnv(Env):
    """ Two wheeled robot with constant goal Env
    """
    def __init__(self):
        """
        """
        self.config = {"state_size" : 3,\
                       "input_size" : 2,\
                       "dt" : 0.01,\
                       "max_step" : 500,\
                       "input_lower_bound": [-1.5, -3.14],\
                       "input_upper_bound": [1.5, 3.14],
                       }

        super(TwoWheeledEnv, self).__init__(self.config)
    
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
        self.goal_state = np.array([0., 0, -2., 3.])
        
        # clear memory
        self.history_x = []
        self.history_g_x = []

        return self.curr_x, {"goal_state": self.goal_state}

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
        # clip action
        u = np.clip(u,
                    self.config["input_lower_bound"],
                    self.config["input_lower_bound"])

        # step
        next_x = np.matmul(self.A, self.curr_x[:, np.newaxis]) \
                 + np.matmul(self.B, u[:, np.newaxis])

        # TODO: implement costs

        # save history
        self.history_x.append(next_x.flatten())
        self.history_g_x.append(self.goal_state.flatten())
        
        # update
        self.curr_x = next_x.flatten()
        # update costs
        self.step_count += 1

        return next_x.flatten(), 0., self.step_count > self.config["max_step"], {"goal_state" : self.goal_state}
    
class TwoWheeledEnv(Env):
    """ Two wheeled robot Env
    """
    def __init__(self):
        """
        """
        self.config = {"state_size" : 3,\
                       "input_size" : 2,\
                       "dt" : 0.01,\
                       "max_step" : 500,\
                       "input_lower_bound": [-1.5, -3.14],\
                       "input_upper_bound": [1.5, 3.14],
                       }

        super(TwoWheeledEnv, self).__init__(self.config)
    
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
        self.goal_state = np.array([0., 0, -2., 3.])
        
        # clear memory
        self.history_x = []
        self.history_g_x = []

        return self.curr_x, {"goal_state": self.goal_state}

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
        # clip action
        u = np.clip(u,
                    self.config["input_lower_bound"],
                    self.config["input_lower_bound"])

        # step
        next_x = np.matmul(self.A, self.curr_x[:, np.newaxis]) \
                 + np.matmul(self.B, u[:, np.newaxis])

        # TODO: implement costs

        # save history
        self.history_x.append(next_x.flatten())
        self.history_g_x.append(self.goal_state.flatten())
        
        # update
        self.curr_x = next_x.flatten()
        # update costs
        self.step_count += 1

        return next_x.flatten(), 0., self.step_count > self.config["max_step"], {"goal_state" : self.goal_state}
    
