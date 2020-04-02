import numpy as np

class FirstOrderLagConfigModule():
    # parameters
    ENV_NAME = "FirstOrderLag-v0"
    TYPE = "Linear"
    TASK_HORIZON = 1000
    PRED_LEN = 10
    STATE_SIZE = 4
    INPUT_SIZE = 2
    DT = 0.05
    # cost parameters
    R = np.eye(INPUT_SIZE)
    Q = np.eye(STATE_SIZE)
    Sf = np.eye(STATE_SIZE)
    # bounds
    INPUT_LOWER_BOUND = np.array([-0.5, -0.5])
    INPUT_UPPER_BOUND = np.array([0.5, 0.5])
    # DT_INPUT_LOWER_BOUND = np.array([-0.5 * DT, -0.5 * DT])
    # DT_INPUT_UPPER_BOUND = np.array([0.25 * DT, 0.25 * DT])
    DT_INPUT_LOWER_BOUND = None
    DT_INPUT_UPPER_BOUND = None

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
           "cgmres-NMPC":{
           },
           "newton-NMPC":{
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
        return (u**2) * np.diag(FirstOrderLagConfigModule.R)
    
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
        return ((x - g_x)**2) * np.diag(FirstOrderLagConfigModule.Q)

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
                * np.diag(FirstOrderLagConfigModule.Sf)