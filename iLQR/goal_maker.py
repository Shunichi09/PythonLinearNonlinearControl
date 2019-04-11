import numpy as np
import math
import matplotlib.pyplot as plt


def make_trajectory(goal_type, N):
    """
    Parameters
    -------------
    goal_type : str
        goal type
    N : int
        length of reference trajectory
    Returns
    -----------
    ref_trajectory : numpy.ndarray, shape(N, STATE_SIZE)

    Notes
    ---------
    this function can only deal with the 
    """
    
    if goal_type == "const":
        ref_trajectory = np.array([[5., 3., 0.]])

        return ref_trajectory
    
    if goal_type == "sin":






class GoalMaker():
    """
    Attributes
    -----------

    """

    def __init_(self, goal_type="const", N=500, dt=0.01):
        """
        """
        self.N = N
        self.goal_type = goal_type
        self.dt = dt

        self.ref_traj = None
        
    def preprocess(self):
        """preprocess of make goal

        """
        goal_type_list = ["const", "sin"]

        if self.goal_type not in goal_type_list:
            raise ValueError("{0} is not in implemented goal type. please implement that!!".format(self.goal_type))

        self.ref_traj = make_trajectory(self.goal_type)

    def calc_goal(self, x):
        """
        """



        return goal


