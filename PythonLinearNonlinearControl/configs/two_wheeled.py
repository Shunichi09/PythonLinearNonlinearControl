import numpy as np

class TwoWheeledConfigModule():
    # parameters
    ENV_NAME = "TwoWheeled-v0"
    TYPE = "Nonlinear"
    TASK_HORIZON = 1000
    PRED_LEN = 10
    STATE_SIZE = 3
    INPUT_SIZE = 2
    DT = 0.01
    # cost parameters
    R = np.eye(INPUT_SIZE)
    Q = np.eye(STATE_SIZE)
    Sf = np.eye(STATE_SIZE)
    # bounds
    INPUT_LOWER_BOUND = np.array([-1.5, 3.14])
    INPUT_UPPER_BOUND = np.array([1.5, 3.14])

    def __init__(self):
        """ 
        """
        # opt configs
        self.opt_config = {
            "Random": {
                "popsize": 5000
            },
            "CEM": {
                "popsize": 500,
                "num_elites": 50,
                "max_iters": 15,
                "alpha": 0.3,
                "init_var":1.,
                "threshold":0.001
            },
            "MPPI":{
                "beta" : 0.6,
                "popsize": 5000,
                "kappa": 0.9,
                "noise_sigma": 0.5,
            },
           "iLQR":{
           },
           "NMPC-CGMRES":{
           },
           "NMPC-Newton":{
           },
        }   

    @staticmethod
    def input_cost_fn(u):
        """ input cost functions
        Args:
            u (numpy.ndarray): input, shape(input_size, )
                or shape(pop_size, input_size)
        Returns:
            cost (numpy.ndarray): cost of input, none or shape(pop_size, )
        """
        return (u**2) * np.diag(TwoWheeledConfigModule.R) * 0.1
    
    @staticmethod
    def state_cost_fn(x, g_x):
        """ state cost function
        Args:
            x (numpy.ndarray): state, shape(pred_len, state_size)
                or shape(pop_size,  pred_len, state_size)
            g_x (numpy.ndarray): goal state, shape(state_size, )
                or shape(pop_size, state_size)
        Returns:
            cost (numpy.ndarray): cost of state, none or shape(pop_size, )
        """
        return ((x - g_x)**2) * np.diag(TwoWheeledConfigModule.Q)

    @staticmethod
    def terminal_state_cost_fn(terminal_x, terminal_g_x):
        """
        Args:
            terminal_x (numpy.ndarray): terminal state,
                shape(state_size, ) or shape(pop_size, state_size)
            terminal_g_x (numpy.ndarray): terminal goal state,
                shape(state_size, ) or shape(pop_size, state_size)
        Returns:
            cost (numpy.ndarray): cost of state, none or shape(pop_size, )
        """
        return ((terminal_x - terminal_g_x)**2) \
                * np.diag(TwoWheeledConfigModule.Sf)